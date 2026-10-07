import os
import json
from pypdf import PdfReader
import lancedb
import re

BASE_DIR = r"F:\RSNA\coco_data"
REF_DIR = os.path.join(BASE_DIR, "sources", "references")
DB_URI = os.path.join(BASE_DIR, "lancedb")
PDF_PATH = r"F:\RSNA\Science Research Writing_ For Native And Non-native Speakers Of English, Second Edition - PDF Room.pdf"

UNITS = [
    ("UNIT_1_How_to_Write_the_Introduction", ["UNIT 1", "How to Write the Introduction"]),
    ("UNIT_2_How_to_Write_about_Methods", ["UNIT 2", "How to Write about Methods"]),
    ("UNIT_3_How_to_Write_about_Results", ["UNIT 3", "How to Write about Results"]),
    ("UNIT_4_How_to_Write_the_Discussion", ["UNIT 4", "How to Write the Discussion"]),
    ("UNIT_5_How_to_Write_the_Conclusion", ["UNIT 5", "How to Write the Conclusion"]),
    ("UNIT_6_Writing_the_Abstract", ["UNIT 6", "Writing the Abstract"]),
    ("UNIT_7_Writing_the_Title", ["UNIT 7", "Writing the Title", "Unit 7: Writing the Title"]),
]

def extract_pdf_text():
    reader = PdfReader(PDF_PATH)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

def segment_by_units(full_text):
    segments = {}
    text = full_text
    # Build regex for unit anchors
    anchors = []
    for unit_name, keywords in UNITS:
        for kw in keywords:
            anchors.append((unit_name, kw))
    # Find positions
    positions = []
    for unit_name, kw in anchors:
        for m in re.finditer(re.escape(kw), text, flags=re.IGNORECASE):
            positions.append((m.start(), unit_name))
    positions.sort()
    if not positions:
        # Fallback: all in Unit 1
        segments[UNITS[0][0]] = text
        return segments
    # Build ranges
    for i, (start, unit_name) in enumerate(positions):
        end = positions[i+1][0] if i+1 < len(positions) else len(text)
        seg = text[start:end]
        # Append segments per unit (merge multiple anchors for same unit)
        if unit_name in segments:
            segments[unit_name] += "\n" + seg
        else:
            segments[unit_name] = seg
    return segments

def chunk_text(unit_name, text, chunk_size=1000, overlap=200):
    chunks = []
    i = 0
    L = len(text)
    idx = 0
    while i < L:
        end = min(i + chunk_size, L)
        chunk = text[i:end]
        chunk = chunk.strip()
        if chunk:
            chunks.append({
                "unit": unit_name,
                "text": chunk,
                "chunk_id": f"{unit_name}_{idx}"
            })
            idx += 1
        if end >= L:
            break
        i = end - overlap
        if i < 0:
            i = 0
    return chunks

def ensure_dirs():
    os.makedirs(REF_DIR, exist_ok=True)
    os.makedirs(DB_URI, exist_ok=True)

def write_unit_files(segments):
    for unit_name, text in segments.items():
        path = os.path.join(REF_DIR, f"{unit_name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

def ingest_to_lancedb(rows):
    db = lancedb.connect(DB_URI)
    try:
        names = db.table_names()
        if "reference_knowledge" in names:
            tbl = db.open_table("reference_knowledge")
            tbl.add(rows)
            return
    except Exception:
        pass
    tbl = db.create_table("reference_knowledge", rows)

def main():
    ensure_dirs()
    full_text = extract_pdf_text()
    segments = segment_by_units(full_text)
    write_unit_files(segments)
    rows = []
    for unit, text in segments.items():
        rows.extend(chunk_text(unit, text))
    ingest_to_lancedb(rows)
    print(json.dumps({"status": "ok", "units": list(segments.keys()), "chunks": len(rows)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
