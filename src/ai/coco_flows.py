from __future__ import annotations

import os
import datetime
from typing import List, Optional, Any, Callable
import dataclasses
from src.core.path_config import get_coco_data_dir

try:
    import cocoindex  # type: ignore
    import cocoindex.targets.lancedb as coco_lancedb  # type: ignore
except Exception:
    cocoindex = None
    coco_lancedb = None

# Compatibility guard:
# Some cocoindex builds raise ValueError at decorator time for string-annotated
# DataSlice types (e.g. "cocoindex.DataSlice[str]"). If so, disable cocoindex
# flows and keep pure-Python fallback path alive.
if cocoindex is not None:
    try:
        @cocoindex.transform_flow()
        def _cocoindex_probe(x: "cocoindex.DataSlice[str]") -> "cocoindex.DataSlice[str]":
            return x
        del _cocoindex_probe
    except Exception:
        cocoindex = None
        coco_lancedb = None


def _noop_decorator(*args: Any, **kwargs: Any) -> Callable:
    def _inner(fn: Callable) -> Callable:
        return fn

    return _inner


transform_flow = cocoindex.transform_flow if cocoindex else _noop_decorator

def text_to_embedding(text: Any) -> Any:
    """Fallback symbol for imports when cocoindex is unavailable."""
    return text

# Paths
BASE_DIR = str(get_coco_data_dir())
RAW_SOURCE_DIR = os.path.join(BASE_DIR, "sources", "raw_papers")
REF_SOURCE_DIR = os.path.join(BASE_DIR, "sources", "references")
PROJ_SOURCE_DIR = os.path.join(BASE_DIR, "sources", "projects")
LANCEDB_URI = os.path.join(BASE_DIR, "lancedb")

# Ensure directories exist
os.makedirs(RAW_SOURCE_DIR, exist_ok=True)
os.makedirs(REF_SOURCE_DIR, exist_ok=True)
os.makedirs(PROJ_SOURCE_DIR, exist_ok=True)
os.makedirs(LANCEDB_URI, exist_ok=True)

# Placeholder for text_to_embedding if it's needed by knowledge_base
# Assuming it's a transform function
if cocoindex:
    @transform_flow()
    def text_to_embedding(text: "cocoindex.DataSlice[str]") -> "cocoindex.DataSlice[List[float]]":
        return text.transform(
            cocoindex.functions.SentenceTransformerEmbed(model="all-MiniLM-L6-v2")
        )

@dataclasses.dataclass
class ResearchOutline:
    """Structure for the global research outline."""
    title: str
    core_claims: List[str]
    primary_hypothesis: str
    null_hypothesis: str
    methodology_highlights: str
    key_results: List[str]
    target_audience: str

@dataclasses.dataclass
class UnitDraft:
    """Structure for a drafted unit."""
    unit_id: str
    content: str
    key_points: List[str]
    critique: Optional[dict] = None

@dataclasses.dataclass
class UnitSummary:
    """Structure for a compressed unit summary."""
    unit_id: str
    summary_text: str
    significance: str

if cocoindex:
    @transform_flow()
    def generate_outline(raw_text: "cocoindex.DataSlice[str]") -> "cocoindex.DataSlice[ResearchOutline]":
        return raw_text.transform(
            cocoindex.functions.ExtractByLlm(
                llm_spec=cocoindex.LlmSpec(
                    api_type=cocoindex.LlmApiType.OPENAI,
                    model="gpt-4o",
                ),
                output_type=ResearchOutline,
                instruction=(
                    "Analyze the provided raw research notes/data and generate a structured outline "
                    "for a medical imaging paper. Identify core claims, methodology highlights, "
                    "and key expected results."
                ),
            )
        )

if cocoindex:
    def draft_unit_parallel(
        raw_text: "cocoindex.DataSlice[str]",
        outline: "cocoindex.DataSlice[ResearchOutline]",
        unit_name: str,
    ) -> "cocoindex.DataSlice[UnitDraft]":
        return raw_text.transform(
            cocoindex.functions.ExtractByLlm(
                llm_spec=cocoindex.LlmSpec(
                    api_type=cocoindex.LlmApiType.OPENAI,
                    model="gpt-4o",
                ),
                output_type=UnitDraft,
                instruction=f"""
                Write a draft for {unit_name} of the paper.
                Use the provided Global Outline to ensure consistency in claims and terminology.
                Use the Raw Data for specific details.
                """,
            ),
            outline=outline,
        )

if cocoindex:
    @transform_flow()
    def compress_unit(draft: "cocoindex.DataSlice[UnitDraft]") -> "cocoindex.DataSlice[UnitSummary]":
        return draft.transform(
            cocoindex.functions.ExtractByLlm(
                llm_spec=cocoindex.LlmSpec(
                    api_type=cocoindex.LlmApiType.OPENAI,
                    model="gpt-3.5-turbo",
                ),
                output_type=UnitSummary,
                instruction=(
                    "Summarize the provided unit draft. Focus on statistical significance, "
                    "trends, and key arguments. Ignore verbose descriptions. "
                    "This summary will be used to write the Discussion section."
                ),
            )
        )

if cocoindex:
    @transform_flow()
    def draft_discussion(
        intro_summary: "cocoindex.DataSlice[UnitSummary]",
        results_summary: "cocoindex.DataSlice[UnitSummary]",
        outline: "cocoindex.DataSlice[ResearchOutline]",
    ) -> "cocoindex.DataSlice[UnitDraft]":
        return intro_summary.transform(
            cocoindex.functions.ExtractByLlm(
                llm_spec=cocoindex.LlmSpec(
                    api_type=cocoindex.LlmApiType.OPENAI,
                    model="gpt-4o",
                ),
                output_type=UnitDraft,
                instruction="""
                Write the Discussion section (Unit 4).
                Synthesize the claims from the Introduction and the findings from the Results.
                Discuss implications and limitations.
                """,
            ),
            results_summary=results_summary,
            outline=outline,
        )

    @cocoindex.flow_def(name="PaperWritingFlow")
    def paper_writing_flow(
        flow_builder: "cocoindex.FlowBuilder", data_scope: "cocoindex.DataScope"
    ) -> None:
        data_scope["papers"] = flow_builder.add_source(
            cocoindex.sources.LocalFile(path=RAW_SOURCE_DIR),
            refresh_interval=datetime.timedelta(seconds=30),
        )

        collector = data_scope.add_collector()

        with data_scope["papers"].row() as paper:
            paper["outline"] = generate_outline(paper["content"])
            paper["u1_intro"] = draft_unit_parallel(
                paper["content"], paper["outline"], "Unit 1: Introduction"
            )
            paper["u2_methods"] = draft_unit_parallel(
                paper["content"], paper["outline"], "Unit 2: Methods"
            )
            paper["u3_results"] = draft_unit_parallel(
                paper["content"], paper["outline"], "Unit 3: Results"
            )

            paper["u1_summary"] = compress_unit(paper["u1_intro"])
            paper["u3_summary"] = compress_unit(paper["u3_results"])

            paper["u4_discussion"] = draft_discussion(
                paper["u1_summary"], paper["u3_summary"], paper["outline"]
            )

            paper["u4_summary"] = compress_unit(paper["u4_discussion"])
            paper["u5_conclusion"] = paper["u4_summary"].transform(
                cocoindex.functions.ExtractByLlm(
                    llm_spec=cocoindex.LlmSpec(
                        api_type=cocoindex.LlmApiType.OPENAI, model="gpt-4o"
                    ),
                    output_type=UnitDraft,
                    instruction="Write Conclusion based on Discussion summary.",
                )
            )

            paper["final_meta"] = paper["outline"].transform(
                cocoindex.functions.ExtractByLlm(
                    llm_spec=cocoindex.LlmSpec(
                        api_type=cocoindex.LlmApiType.OPENAI, model="gpt-4o"
                    ),
                    output_type=UnitDraft,
                    instruction="Generate Title and Abstract based on the full paper outline and conclusions.",
                ),
                conclusion_summary=compress_unit(paper["u5_conclusion"]),
            )

            collector.collect(
                id=cocoindex.GeneratedField.UUID,
                filename=paper["filename"],
                outline=paper["outline"],
                intro=paper["u1_intro"],
                methods=paper["u2_methods"],
                results=paper["u3_results"],
                discussion=paper["u4_discussion"],
                conclusion=paper["u5_conclusion"],
                abstract_title=paper["final_meta"],
            )

        collector.export(
            "generated_papers",
            coco_lancedb.LanceDB(db_uri=LANCEDB_URI, table_name="generated_papers"),
            primary_key_fields=["id"],
        )
