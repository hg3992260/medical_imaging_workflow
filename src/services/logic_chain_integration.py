"""
Logic Chain Integration Service

将LogicChainAgent集成到现有AgentCoreLoop工作流中。

该服务提供:
1. 与AgentCoreLoopExecutor的无缝集成
2. 与DataAggregationService的数据对接
3. 与KnowledgeBase的RAG增强

Author: RSNA Medical Imaging Workflow
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict

from src.ai.logic_chain_agent import (
    LogicChainAgent,
    LogicNode,
    InferenceChain,
    PaperDraft,
    create_logic_chain_agent
)

logger = logging.getLogger(__name__)


@dataclass
class LogicChainResult:
    """逻辑链处理结果"""
    success: bool
    logic_chains: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    paper_draft: Dict[str, Any]
    key_findings: List[str]
    statistics: Dict[str, int]
    visualization_data: Dict[str, Any]
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LogicChainIntegration:
    """
    逻辑链集成服务

    连接DataAggregationService、KnowledgeBase和LogicChainAgent，
    提供端到端的逻辑链提取和论文生成能力。
    """

    def __init__(
        self,
        llm_service=None,
        model: str = "llama3",
        knowledge_base=None,
        db_service=None,
        progress_callback: Optional[Callable[[str], None]] = None,
        stream_callback: Optional[Callable[[str], None]] = None
    ):
        self.llm_service = llm_service
        self.model = model
        self.kb = knowledge_base
        self.db_service = db_service
        self.progress_cb = progress_callback
        self.stream_cb = stream_callback

        # 初始化LogicChainAgent
        self.agent = create_logic_chain_agent(
            llm_service=llm_service,
            model=model,
            knowledge_base=knowledge_base,
            progress_callback=progress_callback,
            stream_callback=stream_callback
        )

    def _progress(self, message: str):
        """报告进度"""
        if self.progress_cb:
            try:
                self.progress_cb(message)
            except:
                pass
        logger.info(message)

    def _stream(self, token: str):
        """流式输出"""
        if self.stream_cb:
            try:
                self.stream_cb(token)
            except:
                pass

    def process_project_data(
        self,
        project_id: str,
        patient_cases: List[Any] = None,
        project_context: str = "",
        style_guidelines: str = "",
        use_rag: bool = True
    ) -> LogicChainResult:
        """
        处理项目数据，提取逻辑链并生成论文草稿

        Args:
            project_id: 项目ID
            patient_cases: 患者病例列表(可选，如不提供则自动获取)
            project_context: 项目上下文描述
            style_guidelines: 写作风格指南
            use_rag: 是否使用RAG增强

        Returns:
            LogicChainResult对象
        """
        self._progress(f"Starting Logic Chain processing for project {project_id}")
        self._stream("\n> 🔬 [LogicChainIntegration]: 开始处理项目数据...\n")

        try:
            # 1. 获取患者数据
            if patient_cases is None:
                patient_cases = self._fetch_patient_cases(project_id)

            if not patient_cases:
                self._stream("> ⚠️ [LogicChainIntegration]: 未找到患者数据\n")
                return LogicChainResult(
                    success=False,
                    logic_chains=[],
                    nodes=[],
                    edges=[],
                    paper_draft={},
                    key_findings=[],
                    statistics={},
                    visualization_data={},
                    error="No patient cases found"
                )

            self._stream(f"> 📋 [LogicChainIntegration]: 找到 {len(patient_cases)} 个病例\n")

            # 2. 聚合数据
            all_ocr_data = []
            all_dicom_data = []
            all_text_data = []

            for case in patient_cases:
                if hasattr(case, 'ocr_data') and case.ocr_data:
                    all_ocr_data.extend(case.ocr_data)
                if hasattr(case, 'dicom_data') and case.dicom_data:
                    all_dicom_data.extend(case.dicom_data)
                if hasattr(case, 'text_data') and case.text_data:
                    all_text_data.extend(case.text_data)

            self._stream(f"> 📊 [LogicChainIntegration]: OCR数据: {len(all_ocr_data)}, DICOM数据: {len(all_dicom_data)}, 文本数据: {len(all_text_data)}\n")

            # 3. 使用RAG增强上下文
            if use_rag and self.kb:
                self._stream("> 🔍 [LogicChainIntegration]: 使用RAG增强上下文...\n")
                project_context = self._enhance_context_with_rag(
                    project_context,
                    all_ocr_data,
                    all_dicom_data,
                    all_text_data
                )

            # 4. 提取结构化数据
            structured_data = self._extract_structured_data(patient_cases)
            self._stream(f"> 📈 [LogicChainIntegration]: 提取了 {len(structured_data.get('statistical_findings', []))} 个统计发现\n")

            # 5. 执行逻辑链处理
            self._stream("> 🧠 [LogicChainIntegration]: 执行逻辑链提取...\n")
            result = self.agent.process(
                ocr_data=all_ocr_data,
                dicom_data=all_dicom_data,
                text_data=all_text_data,
                structured_data=structured_data,
                project_context=project_context,
                style_guidelines=style_guidelines
            )

            # 6. 生成可视化数据
            vis_data = self.agent.get_visualization_data(result)

            self._stream("> ✅ [LogicChainIntegration]: 处理完成\n")

            return LogicChainResult(
                success=True,
                logic_chains=result.get("logic_chains", []),
                nodes=result.get("nodes", []),
                edges=result.get("edges", []),
                paper_draft=result.get("paper_draft", {}),
                key_findings=result.get("key_findings", []),
                statistics=result.get("statistics", {}),
                visualization_data=vis_data
            )

        except Exception as e:
            logger.error(f"Logic chain processing failed: {e}", exc_info=True)
            self._stream(f"> ❌ [LogicChainIntegration]: 错误: {str(e)}\n")
            return LogicChainResult(
                success=False,
                logic_chains=[],
                nodes=[],
                edges=[],
                paper_draft={},
                key_findings=[],
                statistics={},
                visualization_data={},
                error=str(e)
            )

    def _fetch_patient_cases(self, project_id: str) -> List[Any]:
        """获取项目中的患者病例"""
        if not self.db_service:
            return []

        try:
            from src.services.data_aggregation_service import DataAggregationService
            aggregator = DataAggregationService(self.db_service)
            return aggregator.aggregate_project_data(project_id)
        except Exception as e:
            logger.error(f"Failed to fetch patient cases: {e}")
            return []

    def _enhance_context_with_rag(
        self,
        context: str,
        ocr_data: List[Dict],
        dicom_data: List[Dict],
        text_data: List[Dict]
    ) -> str:
        """使用知识库增强上下文"""
        enhanced_context = context

        # 从OCR数据中提取关键词进行查询
        keywords = self._extract_keywords(ocr_data, dicom_data, text_data)

        if keywords and self.kb:
            for keyword in keywords[:5]:  # 限制查询数量
                try:
                    query_result = self.kb.query_project(keyword, limit=3)
                    if query_result:
                        enhanced_context += f"\n\n[Reference for {keyword}]:\n{query_result[:500]}"
                except Exception as e:
                    logger.warning(f"RAG query failed for {keyword}: {e}")

        return enhanced_context

    def _extract_keywords(
        self,
        ocr_data: List[Dict],
        dicom_data: List[Dict],
        text_data: List[Dict]
    ) -> List[str]:
        """从数据中提取关键词"""
        import re
        keywords = []

        # 从OCR数据提取
        for item in ocr_data:
            text = item.get("text", "") or item.get("extracted_text", "") or ""
            # 提取可能的医学术语
            terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            keywords.extend(terms[:3])

        # 从DICOM元数据提取
        for item in dicom_data:
            roi_data = item.get("roi_data", [])
            for roi in roi_data:
                roi_type = roi.get("roi_type", "")
                if roi_type:
                    keywords.append(roi_type)
                roi_name = roi.get("roi_name", "")
                if roi_name:
                    keywords.append(roi_name)

        # 去重并返回
        return list(set(keywords))[:20]

    def _extract_structured_data(self, patient_cases: List[Any]) -> Dict[str, Any]:
        """从患者病例提取结构化数据"""
        structured = {
            "statistical_findings": [],
            "demographics": [],
            "potential_correlations": [],
            "sample_size": len(patient_cases)
        }

        for case in patient_cases:
            # 提取统计发现
            if hasattr(case, 'ocr_data'):
                for item in case.ocr_data:
                    text = item.get("text", "") or item.get("extracted_text", "") or ""
                    # 查找统计模式
                    import re
                    p_values = re.findall(r'p\s*[<>=]\s*[0-9.]+', text, re.IGNORECASE)
                    r_values = re.findall(r'r\s*=\s*[0-9.\-]+', text, re.IGNORECASE)
                    n_values = re.findall(r'n\s*=\s*[0-9]+', text, re.IGNORECASE)

                    for pv in p_values[:2]:
                        structured["statistical_findings"].append(f"P-value: {pv}")
                    for rv in r_values[:2]:
                        structured["statistical_findings"].append(f"Correlation: {rv}")
                    for nv in n_values[:2]:
                        structured["statistical_findings"].append(f"Sample size: {nv}")

            # 提取人口学信息
            if hasattr(case, 'display_name'):
                structured["demographics"].append(case.display_name)

        return structured

    def generate_with_core_loop(
        self,
        raw_content: str,
        project_id: str,
        core_loop_executor=None
    ) -> Dict[str, Any]:
        """
        与AgentCoreLoopExecutor集成，先进行逻辑链提取，再进行论文生成

        Args:
            raw_content: 原始内容
            project_id: 项目ID
            core_loop_executor: AgentCoreLoopExecutor实例

        Returns:
            整合后的结果字典
        """
        self._progress("Starting integrated generation pipeline...")
        self._stream("\n> 🚀 [Integration]: 启动集成生成管道...\n")

        # Phase 1: 逻辑链提取
        self._stream("> 📊 Phase 1: Logic Chain Extraction\n")
        logic_result = self.process_project_data(
            project_id=project_id,
            project_context=raw_content,
            use_rag=True
        )

        if not logic_result.success:
            self._stream(f"> ⚠️ [Integration]: 逻辑链提取失败: {logic_result.error}\n")
            # 继续使用原始内容

        # 增强上下文
        enhanced_context = raw_content
        if logic_result.success:
            # 添加逻辑链摘要
            for chain in logic_result.logic_chains[:3]:
                enhanced_context += f"\n\n[Logic Chain]\n{chain.get('narrative', '')}"

            # 添加关键发现
            enhanced_context += "\n\n[Key Findings]\n"
            for finding in logic_result.key_findings:
                enhanced_context += f"- {finding}\n"

        self._stream(f"> 📝 Phase 2: Paper Generation with enhanced context ({len(enhanced_context)} chars)\n")

        # Phase 2: 使用AgentCoreLoop生成论文
        if core_loop_executor:
            paper_result = core_loop_executor.execute_paper_writing(
                enhanced_context,
                project_id
            )

            # 合并结果
            return {
                "success": True,
                "paper_result": paper_result,
                "logic_chain_result": logic_result.to_dict(),
                "enhanced_context_length": len(enhanced_context),
                "logic_chains_count": len(logic_result.logic_chains),
                "key_findings": logic_result.key_findings
            }
        else:
            # 如果没有core_loop_executor，返回逻辑链结果
            return {
                "success": True,
                "paper_result": None,
                "logic_chain_result": logic_result.to_dict(),
                "paper_draft": logic_result.paper_draft,
                "key_findings": logic_result.key_findings
            }

    def export_logic_chain_visualization(
        self,
        result: LogicChainResult,
        format: str = "json"
    ) -> str:
        """
        导出逻辑链可视化数据

        Args:
            result: LogicChainResult实例
            format: 导出格式 ('json', 'mermaid', 'graphml')

        Returns:
            格式化的可视化数据
        """
        if format == "json":
            import json
            return json.dumps(result.visualization_data, indent=2, ensure_ascii=False)

        elif format == "mermaid":
            return self._to_mermaid(result)

        elif format == "graphml":
            return self._to_graphml(result)

        else:
            return str(result.visualization_data)

    def _to_mermaid(self, result: LogicChainResult) -> str:
        """转换为Mermaid图表格式"""
        lines = ["graph TD"]

        # 添加节点
        for node in result.nodes:
            node_id = node.get("node_id", "unknown").replace("-", "_")
            label = node.get("content", "")[:30].replace('"', "'")
            node_type = node.get("node_type", "unknown")
            shape = {
                "evidence": f"{node_id}[{label}]",
                "inference": f"{node_id}({label})",
                "conclusion": f"{node_id}{{{{{label}}}}}",
                "premise": f"{node_id}[{label}]"
            }.get(node_type, f"{node_id}[{label}]")
            lines.append(f"    {shape}")

        # 添加边
        for edge in result.edges:
            from_id = edge.get("from_node", "").replace("-", "_")
            to_id = edge.get("to_node", "").replace("-", "_")
            relation = edge.get("relation_type", "")
            lines.append(f"    {from_id} -->|{relation}| {to_id}")

        return "\n".join(lines)

    def _to_graphml(self, result: LogicChainResult) -> str:
        """转换为GraphML格式"""
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append('<graphml>')
        lines.append('  <graph id="logic_chain" edgedefault="directed">')

        # 添加节点
        for node in result.nodes:
            node_id = node.get("node_id", "unknown")
            lines.append(f'    <node id="{node_id}">')
            lines.append(f'      <data key="type">{node.get("node_type", "")}</data>')
            lines.append(f'      <data key="content">{node.get("content", "")[:100]}</data>')
            lines.append('    </node>')

        # 添加边
        for edge in result.edges:
            edge_id = edge.get("edge_id", "")
            from_id = edge.get("from_node", "")
            to_id = edge.get("to_node", "")
            lines.append(f'    <edge id="{edge_id}" source="{from_id}" target="{to_id}">')
            lines.append(f'      <data key="relation">{edge.get("relation_type", "")}</data>')
            lines.append(f'      <data key="strength">{edge.get("strength", 0)}</data>')
            lines.append('    </edge>')

        lines.append('  </graph>')
        lines.append('</graphml>')

        return "\n".join(lines)


# 便捷函数
def create_integration_service(
    llm_service=None,
    model: str = "llama3",
    knowledge_base=None,
    db_service=None,
    progress_callback=None,
    stream_callback=None
) -> LogicChainIntegration:
    """创建LogicChainIntegration实例"""
    return LogicChainIntegration(
        llm_service=llm_service,
        model=model,
        knowledge_base=knowledge_base,
        db_service=db_service,
        progress_callback=progress_callback,
        stream_callback=stream_callback
    )