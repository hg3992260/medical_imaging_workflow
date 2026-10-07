import os
import shutil
import time

import lancedb


def main():
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    base = os.path.abspath("coco_data/lancedb_tmp/_neigh_test")
    shutil.rmtree(base, ignore_errors=True)
    os.makedirs(base, exist_ok=True)

    from src.ai.project_context_indexer import _load_embedding_model

    model = _load_embedding_model()
    assert model is not None, "embedding model missing"

    texts = [
        "[DICOM_SESSION]\nsession_id: s1\nscope: method\nkey_dicom_fields:\n- SliceThickness: 0.6 mm\n\nMore details about protocol A.\n",
        "[DICOM_SESSION]\nsession_id: s1\nscope: method\nkey_dicom_fields:\n- SliceThickness: 0.6 mm\n- PixelSpacing: 0.4\\0.4\n\nAcquisition parameters and reconstruction kernel.\n",
        "[DICOM_SESSION]\nsession_id: s1\nscope: method\nkey_dicom_fields:\n- SliceThickness: 0.6 mm\n\nEnd of protocol A with contrast details.\n",
    ]
    vecs = model.encode(texts, show_progress_bar=False)

    db = lancedb.connect(base)
    rows = []
    for i, t in enumerate(texts):
        rows.append(
            {
                "id": f"p:dicom:s1:h:{i}",
                "project_id": "p",
                "user_id": "u1",
                "source_type": "dicom",
                "source_id": "s1",
                "chunk_index": i,
                "content_hash": "h",
                "updated_at": str(int(time.time())),
                "text": t,
                "vector": vecs[i].tolist(),
            }
        )
    db.create_table("project_context", data=rows)

    from src.ai.knowledge_base import KnowledgeBase

    kb = KnowledgeBase(db_uri=base, init_reference=False)
    out = kb.query_project("slice thickness protocol", project_id="p", limit=2)
    print(out)
    assert "chunk_index" not in out
    assert "SliceThickness" in out
    assert "More details" in out
    assert "End of protocol A" in out


if __name__ == "__main__":
    main()
