import sys, re, json
sys.path.insert(0, "F:/RSNA/medical_imaging_workflow")
from src.ai.section_reviewer import _extract_json_block

tests = [
    ('Pure JSON', '{"score": 8, "passed": true, "issues": [], "missing_data": [], "hallucination_flags": [], "suggested_edits": []}'),
    ('Thinking prefix', '\x01>\n{"score": 7.5, "passed": false, "issues": ["bad"], "missing_data": [], "hallucination_flags": [], "suggested_edits": []}'),
    ('Markdown fence', '```json\n{"score": 9, "passed": true, "issues": [], "missing_data": [], "hallucination_flags": [], "suggested_edits": []}\n```'),
    ('Key:value', 'score: 9.5\npassed: true\nissues: ["minor"]'),
    ('Narrative text', 'This is just some narrative review with no structured patterns.'),
    ('Empty', ''),
]

for name, text in tests:
    r = _extract_json_block(text)
    sc = r.get("score") if r else "N/A"
    print(f"[{name}] score={sc}")
