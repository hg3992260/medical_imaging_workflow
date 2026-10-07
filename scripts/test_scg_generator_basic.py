import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai.scientific_claim_generator import ScientificClaimGenerator


def main():
    ctx = """
[TABULAR_BRANCH_MINING]
[TABULAR_TREE] Project=LungCTStudy  Samples=8  Branches=10
[INTER_BRANCH_CORRELATION]
- mean_intensity ~ volume: pearson_r=0.901  n_samples=8
- mean_intensity ~ entropy: pearson_r=0.700  n_samples=8

[DATA_DENSITY]
- P/S1/a: density=0.10  weight=0.10
""".strip()
    claims, block = ScientificClaimGenerator().generate("LungCTStudy", ctx, structured_data={}, max_claims=10)
    assert claims
    assert any(c.get("tone_level") == 5 for c in claims)
    assert any(isinstance(c.get("confidence_score"), float) for c in claims)
    assert "[SCG_CLAIM_OBJECTS_JSON]" in block


if __name__ == "__main__":
    main()
