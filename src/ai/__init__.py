"""
AI模块初始化

包含逻辑链Agent、论文草稿生成Agent和核心循环执行器
"""

from src.ai.agent_core_loop import AgentCoreLoopExecutor
from src.ai.logic_chain_agent import (
    LogicChainAgent,
    LogicChainExtractor,
    InferenceChainBuilder,
    PaperDraftGenerator,
    LogicNode,
    LogicEdge,
    InferenceChain,
    PaperDraft,
    create_logic_chain_agent,
    process_patient_case
)
from src.ai.paper_draft_agent import (
    PaperDraftAgent,
    TemplateStyleParser,
    DataSourceAggregator,
    SectionContentBuilder,
    SectionContent,
    TemplateStyle,
    create_paper_draft_agent
)

__all__ = [
    'AgentCoreLoopExecutor',
    'LogicChainAgent',
    'LogicChainExtractor',
    'InferenceChainBuilder',
    'PaperDraftGenerator',
    'LogicNode',
    'LogicEdge',
    'InferenceChain',
    'PaperDraft',
    'create_logic_chain_agent',
    'process_patient_case',
    # PaperDraftAgent
    'PaperDraftAgent',
    'TemplateStyleParser',
    'DataSourceAggregator',
    'SectionContentBuilder',
    'SectionContent',
    'TemplateStyle',
    'create_paper_draft_agent',
]