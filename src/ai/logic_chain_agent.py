"""
Logic Chain Agent - 逻辑链提取与论文草稿生成Agent

该模块实现了从多源数据中提取逻辑链条、构建推理链、生成论文草稿的完整流程。

架构设计:
1. LogicChainExtractor - 从数据中提取逻辑关系
2. InferenceChainBuilder - 构建多跳推理链
3. PaperDraftGenerator - 生成论文草稿
4. LogicChainAgent - 整合所有组件的主Agent

Author: RSNA Medical Imaging Workflow
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class LogicNode:
    """逻辑节点 - 表示一个逻辑推理单元"""
    node_id: str
    node_type: str  # 'premise', 'evidence', 'inference', 'conclusion'
    content: str
    source_type: str = ""  # 'ocr', 'dicom', 'text', 'literature', 'inference'
    source_id: str = ""
    confidence: float = 1.0
    entity_context: str = ""  # 领域实体类型约束，例如 "Medical/CT" 或 "Environment/Water"
    evidence_type: str = "direct"  # "direct"(实验/病例直接证据), "background"(文献背景), "derived"(推理)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LogicEdge:
    """逻辑边 - 表示节点间的推理关系"""
    edge_id: str
    from_node: str
    to_node: str
    relation_type: str  # 'causes', 'implies', 'supports', 'contradicts', 'correlates'
    strength: float = 1.0  # 0-1, 推理强度
    evidence: str = ""  # 支持该推理的证据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceChain:
    """推理链 - 表示完整的推理路径"""
    chain_id: str
    nodes: List[LogicNode]
    edges: List[LogicEdge]
    chain_type: str = ""  # 'discovery', 'validation', 'comparison', 'conclusion'
    narrative: str = ""  # 自然语言叙述
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "chain_type": self.chain_type,
            "narrative": self.narrative,
            "confidence": self.confidence,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }


@dataclass
class PaperDraft:
    """论文草稿结构"""
    title: str
    abstract: str
    sections: Dict[str, str]  # section_name -> content
    logic_chains: List[InferenceChain]
    key_claims: List[Dict[str, Any]]
    references: List[Dict[str, str]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "sections": self.sections,
            "logic_chains": [lc.to_dict() for lc in self.logic_chains],
            "key_claims": self.key_claims,
            "references": self.references,
            "metadata": self.metadata,
        }


class MetadataAnalyzer:
    """
    Metadata Analyzer - 注入物理常识与逻辑约束
    根据预定义的物理规则（如管电流 mA、重建算法）为提取的数据打上明确的逻辑标签。
    """
    @staticmethod
    def analyze_dose_ranking(text: str, metadata: dict) -> dict:
        """
        物理真值表 (Physics Truth Table):
        例如: mA < 200 -> Low Dose; algorithm == CV -> Baseline
        """
        # 尝试从文本中提取 mA
        import re
        ma_match = re.search(r'(\d+)\s*mA', text, re.IGNORECASE)
        if ma_match:
            ma_val = int(ma_match.group(1))
            if ma_val < 200:
                metadata['dose_context'] = f"{ma_val}mA (Low Dose)"
                metadata['priority'] = "Low Dose"
            else:
                metadata['dose_context'] = f"{ma_val}mA (Standard Dose)"
                metadata['priority'] = "Standard Dose"
                
        # 尝试提取重建算法 CV / CI
        if re.search(r'\bCV\b', text, re.IGNORECASE):
            metadata['algorithm'] = "CV"
            metadata['role'] = "Baseline"
        elif re.search(r'\bCI\b', text, re.IGNORECASE):
            metadata['algorithm'] = "CI"
            metadata['role'] = "Experimental"
            
        return metadata

def normalize_medical_term(raw_term: str) -> Dict[str, str]:
    """
    将原始 Excel 表头或文本片段映射到 RadLex 类别
    例如: "SD肝" -> {"anatomy": "Liver", "metric": "Image_Noise"}
    """
    result = {"anatomy": "Unknown", "metric": "Unknown"}
    
    raw_lower = str(raw_term).lower()
    
    # 解剖映射
    if any(k in raw_lower for k in ["肝", "liver"]): result["anatomy"] = "Liver"
    if any(k in raw_lower for k in ["脾", "spleen"]): result["anatomy"] = "Spleen"
    if any(k in raw_lower for k in ["肾", "kidney"]): result["anatomy"] = "Kidney"
    if any(k in raw_lower for k in ["胰", "pancreas"]): result["anatomy"] = "Pancreas"
    if any(k in raw_lower for k in ["腹主", "aorta"]): result["anatomy"] = "Abdominal Aorta"
    if any(k in raw_lower for k in ["脑", "brain", "neuro"]): result["anatomy"] = "Brain"
    if any(k in raw_lower for k in ["肺", "lung"]): result["anatomy"] = "Lung"
    if any(k in raw_lower for k in ["骨", "bone", "msk"]): result["anatomy"] = "Bone"
    
    # 指标映射
    if "sd" in raw_lower or "standard deviation" in raw_lower: result["metric"] = "Image_Noise"
    if "ct" in raw_lower or "hu" in raw_lower or "hounsfield" in raw_lower: result["metric"] = "CT_Attenuation"
    if "snr" in raw_lower: result["metric"] = "Signal_to_Noise_Ratio"
    if "cnr" in raw_lower: result["metric"] = "Contrast_to_Noise_Ratio"
    if any(k in raw_lower for k in ["adc", "apparent diffusion"]): result["metric"] = "Apparent_Diffusion_Coefficient"
    if "ivim" in raw_lower: result["metric"] = "IVIM"
    if "dki" in raw_lower: result["metric"] = "DKI"
    if any(k in raw_lower for k in ["t1", "t2", "tr", "te"]): result["metric"] = "MR_Relaxation"
    
    return result


class LogicScopeValidator:
    @staticmethod
    def _normalize_scope(scope: str) -> str:
        s = str(scope or "").strip().lower()
        if not s:
            return "General"
        if "pcct" in s or "photon counting" in s:
            return "Medical/PCCT"
        if "mri" in s:
            return "Medical/MRI"
        if any(k in s for k in ["ct", "dicom", "radiology", "hounsfield", "hu"]):
            return "Medical/CT"
        if any(k in s for k in ["water", "environment"]):
            return "Environment/Water"
        return "General"

    @staticmethod
    def validate_connection(from_node: LogicNode, to_node: LogicNode, project_scope: str) -> bool:
        fctx = str(from_node.entity_context or "General")
        tctx = str(to_node.entity_context or "General")
        pscope = LogicScopeValidator._normalize_scope(project_scope)

        # 硬阻断跨域连接
        if fctx != "General" and tctx != "General" and fctx != tctx:
            return False

        # inference 节点需与项目范围兼容
        if to_node.node_type == "inference" and pscope != "General":
            if pscope.startswith("Medical/") and not str(tctx).startswith("Medical/"):
                return False
            if pscope.startswith("Environment/") and not str(tctx).startswith("Environment/"):
                return False
        return True

class LogicChainExtractor:
    """
    从多源数据中提取逻辑关系

    支持:
    - OCR结构化数据提取
    - DICOM影像特征提取
    - 临床文本关系提取
    - 统计数据因果推断
    """

    # 关键词模式 - 用于识别逻辑关系
    CAUSAL_PATTERNS = [
        r"because\s+of",
        r"due\s+to",
        r"leads?\s+to",
        r"causes?",
        r"results?\s+in",
        r"因为",
        r"导致",
        r"引起",
        r"使得",
    ]

    CORRELATION_PATTERNS = [
        r"correlates?\s+with",
        r"associated\s+with",
        r"related\s+to",
        r"相关",
        r"关联",
        r"与.*有关",
    ]

    CONTRADICTION_PATTERNS = [
        r"however",
        r"but",
        r"contrary\s+to",
        r"inconsistent\s+with",
        r"但是",
        r"然而",
        r"相反",
        r"与.*矛盾",
    ]

    STATISTICAL_PATTERNS = [
        r"p\s*[<>=]\s*[0-9.]+",
        r"r\s*=\s*[0-9.\-]+",
        r"n\s*=\s*[0-9]+",
        r"confidence\s+interval",
        r"显著性",
        r"置信区间",
    ]

    def __init__(self, llm_service=None):
        self.llm_service = llm_service

    def extract_from_ocr(self, ocr_data: List[Dict[str, Any]]) -> List[LogicNode]:
        """从OCR数据提取逻辑节点"""
        nodes = []

        for idx, item in enumerate(ocr_data):
            text = item.get("text", "") or item.get("extracted_text", "") or ""
            if not text:
                continue

            # 尝试解析JSON结构化数据
            structured_data = None
            try:
                if text.strip().startswith("{") or text.strip().startswith("["):
                    structured_data = json.loads(text)
            except:
                pass

            if structured_data:
                # 处理结构化数据
                nodes.extend(self._extract_from_structured(
                    structured_data,
                    source_type="ocr",
                    source_id=item.get("image_name", f"ocr_{idx}"),
                    confidence=item.get("confidence", 1.0)
                ))
            else:
                # 处理非结构化文本
                extracted_nodes = self._extract_from_text(
                    text,
                    source_type="ocr",
                    source_id=item.get("image_name", f"ocr_{idx}"),
                    confidence=item.get("confidence", 1.0)
                )
                nodes.extend(extracted_nodes)

        return nodes

    def extract_from_dicom(self, dicom_data: List[Dict[str, Any]]) -> List[LogicNode]:
        """从DICOM数据提取逻辑节点"""
        nodes = []

        for idx, item in enumerate(dicom_data):
            source_id = item.get("file_name", f"dicom_{idx}")

            # 提取DICOM元数据作为证据节点
            dicom_info = item.get("dicom_info", "")
            if dicom_info:
                try:
                    info = json.loads(dicom_info) if isinstance(dicom_info, str) else dicom_info
                    for key, value in info.items():
                        if value:
                            nodes.append(LogicNode(
                                node_id=f"dicom_meta_{idx}_{key}",
                                node_type="evidence",
                                content=f"{key}: {value}",
                                source_type="dicom",
                                source_id=source_id,
                                confidence=1.0,
                                evidence_type="direct",
                                metadata={"key": key, "value": value}
                            ))
                except:
                    pass

            # 提取ROI数据
            roi_data = item.get("roi_data", [])
            for roi_idx, roi in enumerate(roi_data):
                roi_type = roi.get("roi_type", "unknown")
                roi_name = roi.get("roi_name", "unnamed")
                area = roi.get("area", "N/A")

                nodes.append(LogicNode(
                    node_id=f"dicom_roi_{idx}_{roi_idx}",
                    node_type="evidence",
                    content=f"ROI {roi_name} ({roi_type}): Area={area}",
                    source_type="dicom",
                    source_id=source_id,
                    confidence=1.0,
                    evidence_type="direct",
                    metadata={
                        "roi_type": roi_type,
                        "roi_name": roi_name,
                        "area": area,
                        "perimeter": roi.get("perimeter"),
                    }
                ))

        return nodes

    def extract_from_text(self, text_data: List[Dict[str, Any]]) -> List[LogicNode]:
        """从文本数据提取逻辑节点"""
        nodes = []

        for idx, item in enumerate(text_data):
            content = item.get("content", "")
            if not content:
                continue
            file_name = str(item.get("file_name", f"text_{idx}") or "").lower()
            source_type = "literature" if any(k in file_name for k in ["literature", "reference", "pubmed", "template", "guide", "docs"]) else "text"

            extracted = self._extract_from_text(
                content,
                source_type=source_type,
                source_id=item.get("file_name", f"text_{idx}"),
                confidence=1.0
            )
            nodes.extend(extracted)

        return nodes

    def _extract_from_structured(
        self,
        data: Any,
        source_type: str,
        source_id: str,
        confidence: float
    ) -> List[LogicNode]:
        """从结构化数据(如JSON)提取逻辑节点"""
        nodes = []
        
        # 简单判定 entity_context
        # 实际项目中可以调用更强大的分类器，这里用简易词频判定
        def _guess_entity_context(text: str) -> str:
            text_lower = str(text).lower()
            if any(k in text_lower for k in ['ct', 'mri', 'hu', 'hounsfield', 'dose', 'dicom', 'roi']):
                return "Medical/CT"
            if any(k in text_lower for k in ['water', 'ph', 'turbidity', 'river', 'lake']):
                return "Environment/Water"
            return "General"

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    nodes.extend(self._extract_from_structured(
                        value, source_type, f"{source_id}_{key}", confidence
                    ))
                elif value:
                    metadata={"key": key}
                    
                    MetadataAnalyzer.analyze_dose_ranking(f"{key}: {value}", metadata)
                    
                    # 注入 medical_meta 
                    med_meta = normalize_medical_term(str(key))
                    metadata["medical_meta"] = med_meta
                    
                    content_str = f"{key}: {value}"
                    node = LogicNode(
                        node_id=f"{source_type}_{source_id}_{key}",
                        node_type="evidence",
                        content=content_str,
                        source_type=source_type,
                        source_id=source_id,
                        confidence=confidence,
                        entity_context=_guess_entity_context(content_str),
                        evidence_type="direct" if source_type in ["ocr", "dicom", "text"] else "background",
                        metadata=metadata
                    )
                    nodes.append(node)
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                nodes.extend(self._extract_from_structured(
                    item, source_type, f"{source_id}_{idx}", confidence
                ))

        return nodes

    def _extract_from_text(
        self,
        text: str,
        source_type: str,
        source_id: str,
        confidence: float
    ) -> List[LogicNode]:
        """从非结构化文本提取逻辑节点"""
        nodes = []

        def _guess_entity_context(text: str) -> str:
            text_lower = str(text).lower()
            if any(k in text_lower for k in ['ct', 'mri', 'hu', 'hounsfield', 'dose', 'dicom', 'roi']):
                return "Medical/CT"
            if any(k in text_lower for k in ['water', 'ph', 'turbidity', 'river', 'lake']):
                return "Environment/Water"
            return "General"

        # 提取统计显著性声明
        for pattern in self.STATISTICAL_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                metadata = {"type": "statistical", "value": match}
                MetadataAnalyzer.analyze_dose_ranking(text, metadata)
                content_str = f"Statistical finding: {match}"
                node = LogicNode(
                    node_id=f"{source_type}_{source_id}_stat_{len(nodes)}",
                    node_type="evidence",
                    content=content_str,
                    source_type=source_type,
                    source_id=source_id,
                    confidence=confidence,
                    entity_context=_guess_entity_context(content_str),
                    evidence_type="background" if source_type == "literature" else "direct",
                    metadata=metadata
                )
                nodes.append(node)

        # 提取因果关系声明
        for pattern in self.CAUSAL_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # 提取因果关系周围上下文
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()

                metadata = {"type": "causal", "context": context}
                MetadataAnalyzer.analyze_dose_ranking(context, metadata)
                content_str = f"Causal relation: {context}"
                node = LogicNode(
                    node_id=f"{source_type}_{source_id}_causal_{len(nodes)}",
                    node_type="inference",
                    content=content_str,
                    source_type=source_type,
                    source_id=source_id,
                    confidence=confidence * 0.8,  # 因果关系置信度略低
                    entity_context=_guess_entity_context(content_str),
                    evidence_type="derived",
                    metadata=metadata
                )
                nodes.append(node)

        # 提取相关性声明
        for pattern in self.CORRELATION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()

                metadata = {"type": "correlation", "context": context}
                MetadataAnalyzer.analyze_dose_ranking(context, metadata)
                content_str = f"Correlation: {context}"
                node = LogicNode(
                    node_id=f"{source_type}_{source_id}_corr_{len(nodes)}",
                    node_type="inference",
                    content=content_str,
                    source_type=source_type,
                    source_id=source_id,
                    confidence=confidence * 0.7,
                    entity_context=_guess_entity_context(content_str),
                    evidence_type="derived",
                    metadata=metadata
                )
                nodes.append(node)

        # 提取矛盾/对比声明
        for pattern in self.CONTRADICTION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()

                metadata = {"type": "contradiction", "context": context}
                MetadataAnalyzer.analyze_dose_ranking(context, metadata)
                content_str = f"Contradiction: {context}"
                node = LogicNode(
                    node_id=f"{source_type}_{source_id}_contra_{len(nodes)}",
                    node_type="inference",
                    content=content_str,
                    source_type=source_type,
                    source_id=source_id,
                    confidence=confidence * 0.8,
                    entity_context=_guess_entity_context(content_str),
                    evidence_type="derived",
                    metadata=metadata
                )
                nodes.append(node)

        return nodes

    def extract_all(
        self,
        ocr_data: List[Dict] = None,
        dicom_data: List[Dict] = None,
        text_data: List[Dict] = None,
        structured_data: Dict = None
    ) -> Tuple[List[LogicNode], List[LogicEdge]]:
        """从所有数据源提取逻辑节点和边"""
        all_nodes = []
        all_edges = []

        # 提取各数据源的节点
        if ocr_data:
            all_nodes.extend(self.extract_from_ocr(ocr_data))
        if dicom_data:
            all_nodes.extend(self.extract_from_dicom(dicom_data))
        if text_data:
            all_nodes.extend(self.extract_from_text(text_data))

        # 从结构化数据中提取统计关系作为边
        if structured_data:
            edges = self._extract_edges_from_structured(structured_data, all_nodes)
            all_edges.extend(edges)

        # 基于节点相似性构建潜在边
        similarity_edges = self._build_similarity_edges(all_nodes)
        all_edges.extend(similarity_edges)

        return all_nodes, all_edges

    def _extract_edges_from_structured(
        self,
        data: Dict[str, Any],
        nodes: List[LogicNode]
    ) -> List[LogicEdge]:
        """从结构化统计结果提取逻辑边"""
        edges = []

        # 提取相关性关系
        correlations = data.get("potential_correlations") or data.get("correlations") or []
        for idx, corr in enumerate(correlations):
            if isinstance(corr, str):
                edge = LogicEdge(
                    edge_id=f"corr_edge_{idx}",
                    from_node="",
                    to_node="",
                    relation_type="correlates",
                    strength=0.7,
                    evidence=corr,
                    metadata={"source": "statistical_analysis"}
                )
                edges.append(edge)

        return edges

    def _build_similarity_edges(self, nodes: List[LogicNode]) -> List[LogicEdge]:
        """基于节点内容相似性构建边"""
        edges = []

        # 简单的关键词匹配来识别相关节点
        for i, node_a in enumerate(nodes):
            for j, node_b in enumerate(nodes):
                if i >= j:
                    continue

                # 检查是否有共同关键词
                words_a = set(re.findall(r'\b\w{3,}\b', node_a.content.lower()))
                words_b = set(re.findall(r'\b\w{3,}\b', node_b.content.lower()))

                overlap = words_a & words_b
                if len(overlap) >= 2:
                    # 有足够关键词重叠，创建关联边
                    edge = LogicEdge(
                        edge_id=f"sim_edge_{i}_{j}",
                        from_node=node_a.node_id,
                        to_node=node_b.node_id,
                        relation_type="supports",
                        strength=0.5 + 0.1 * len(overlap),
                        evidence=f"Shared terms: {', '.join(list(overlap)[:5])}",
                        metadata={"overlap_count": len(overlap)}
                    )
                    edges.append(edge)

        return edges


class InferenceChainBuilder:
    """
    构建多跳推理链

    支持构建:
    - 发现链 (discovery chain)
    - 验证链 (validation chain)
    - 比较链 (comparison chain)
    - 结论链 (conclusion chain)
    """

    def __init__(self, llm_service=None, model: str = "llama3"):
        self.llm_service = llm_service
        self.model = model

    def build_chains(
        self,
        nodes: List[LogicNode],
        edges: List[LogicEdge],
        max_chain_length: int = 5,
        project_scope: str = "",
        strict_mode: bool = False,
    ) -> List[InferenceChain]:
        """构建推理链"""
        chains = []

        # 构建邻接表
        adjacency = defaultdict(list)
        node_map = {n.node_id: n for n in nodes}

        for edge in edges:
            if edge.from_node and edge.to_node:
                # Check entity_context compatibility before allowing the edge
                from_n = node_map.get(edge.from_node)
                to_n = node_map.get(edge.to_node)
                if from_n and to_n:
                    if not LogicScopeValidator.validate_connection(from_n, to_n, project_scope):
                        logger.warning(
                            f"Blocked edge by ScopeValidator: {from_n.entity_context} -> {to_n.entity_context}, scope={project_scope}"
                        )
                        continue
                
                adjacency[edge.from_node].append((edge.to_node, edge))

        # 从证据节点开始构建链
        evidence_nodes = [n for n in nodes if n.node_type == "evidence"]

        for start_node in evidence_nodes[:10]:  # 限制起始节点数量
            # 构建发现链
            chain = self._build_single_chain(
                start_node,
                adjacency,
                node_map,
                max_length=max_chain_length
            )
            if chain and len(chain.nodes) >= 2:
                chains.append(chain)

        # 使用LLM增强推理链
        if self.llm_service and chains:
            chains = self._enhance_chains_with_llm(chains, node_map)
            chains = self._detect_and_correct_contradictions(chains)

        if strict_mode:
            chains = self._strict_validate_inference_chains(chains, project_scope)

        return chains

    def _strict_validate_inference_chains(self, chains: List[InferenceChain], project_scope: str) -> List[InferenceChain]:
        if not chains:
            return chains
        validated = []
        for chain in chains:
            ok = True
            # 规则级检查
            for n in chain.nodes:
                if n.node_type == "inference":
                    if not LogicScopeValidator.validate_connection(chain.nodes[0], n, project_scope):
                        ok = False
                        break
            # LLM 二次检查：是否仅由 premise/evidence 导出
            if ok and self.llm_service:
                premise = "\n".join([f"- {n.content}" for n in chain.nodes if n.node_type in ["premise", "evidence"]][:8])
                infer = "\n".join([f"- {n.content}" for n in chain.nodes if n.node_type == "inference"][:6])
                prompt = f"""
你是严格推理审计器。判断以下 inference 是否【仅由】给定 premise/evidence 导出，且未使用外部常识。
Project Scope: {project_scope}

Premise/Evidence:
{premise}

Inference:
{infer}

仅回复 PASS 或 FAIL: <reason>。
"""
                try:
                    res = self.llm_service.generate(self.model, prompt, options={"temperature": 0.0})
                    if res.get("success"):
                        msg = str(res.get("response", "")).strip().upper()
                        if not msg.startswith("PASS"):
                            ok = False
                except Exception:
                    pass
            if ok:
                validated.append(chain)
            else:
                logger.warning(f"Strict mode removed chain: {chain.chain_id}")
        return validated

    def _enhance_chains_with_llm(self, chains: List[InferenceChain], node_map: Dict[str, LogicNode]) -> List[InferenceChain]:
        """使用大语言模型增强推理链的描述"""
        if not chains:
            return chains

        for chain in chains[:5]:  # 只增强前5个关键链
            try:
                # 收集链节点信息
                node_contents = [f"- {n.node_type}: {n.content} (Context: {n.entity_context})" for n in chain.nodes]
                nodes_text = "\n".join(node_contents)

                prompt = f"""
你目前处于【严格上下文模式】。
TERMINOLOGY GUARD: 你的分析必须仅限于 RSNA RadLex 定义的医学和物理参数范畴（如 CT_Attenuation, Signal_to_Noise_Ratio, Lung, Liver 等）。
DOMAIN FILTER: 如果你在以下节点中发现诸如 Chlorophyll (叶绿素)、Water Quality (水质) 等非医学词汇，请将其视为噪声并直接丢弃或拒绝生成推理链。

根据以下逻辑节点序列，生成一段连贯的医学研究逻辑推理。
逻辑节点序列:
{nodes_text}

请输出一段100字左右的连贯推理段落，说明这些节点之间的逻辑关联。
禁止使用任何未在输入节点中出现的外部知识。若无法从节点导出推理，请回复 "DOMAIN_REJECTED"。
"""
                res = self.llm_service.generate(self.model, prompt, options={"temperature": 0.3})
                if res.get("success"):
                    resp = res.get("response", "").strip()
                    if "DOMAIN_REJECTED" in resp.upper():
                        logger.warning(f"Chain rejected by Domain Filter LLM: {chain.chain_id}")
                        chain.narrative = "Rejected due to domain drift."
                    else:
                        chain.narrative = resp
            except Exception as e:
                logger.error(f"Failed to enhance chain with LLM: {e}")

        # 过滤掉被拒绝的链
        return [c for c in chains if "Rejected" not in c.narrative]

    def _detect_and_correct_contradictions(self, chains: List[InferenceChain]) -> List[InferenceChain]:
        """
        矛盾检测逻辑:
        如果证据节点（Evidence）显示图像噪声（SD）下降等正面指标，
        但推理/结论写成了“质量下降”，触发Self-Correction修正。
        """
        if not self.llm_service:
            return chains
            
        corrected_chains = []
        for chain in chains:
            prompt = f"""
请作为逻辑审计员，检查以下逻辑链是否存在内在矛盾（特别是数据与结论相反的情况）。
例如：如果数据提到"SD下降"或"SNR提升"，但结论是"图像质量下降"，则存在矛盾。

逻辑链内容:
{chain.narrative}

如果存在矛盾，请输出修正后的逻辑链内容。如果不存在矛盾，请回复"NO_CONTRADICTION"。
只输出结果，不要输出多余解释。
"""
            try:
                res = self.llm_service.generate(self.model, prompt, options={"temperature": 0.1})
                if res.get("success"):
                    resp = res.get("response", "").strip()
                    if "NO_CONTRADICTION" not in resp.upper() and len(resp) > 10:
                        logger.warning(f"Detected and corrected contradiction in chain: {chain.chain_id}")
                        chain.narrative = resp
            except Exception as e:
                logger.error(f"Contradiction check failed: {e}")
            corrected_chains.append(chain)
        return corrected_chains

    def _build_single_chain(
        self,
        start_node: LogicNode,
        adjacency: Dict[str, List],
        node_map: Dict[str, LogicNode],
        max_length: int = 5
    ) -> Optional[InferenceChain]:
        """从起始节点构建单条推理链"""
        chain_nodes = [start_node]
        chain_edges = []
        visited = {start_node.node_id}
        current = start_node.node_id

        for _ in range(max_length - 1):
            neighbors = adjacency.get(current, [])
            if not neighbors:
                break

            # 选择最强连接的邻居
            best_neighbor = None
            best_edge = None
            best_strength = 0

            for neighbor_id, edge in neighbors:
                if neighbor_id not in visited:
                    if edge.strength > best_strength:
                        best_strength = edge.strength
                        best_neighbor = neighbor_id
                        best_edge = edge

            if not best_neighbor:
                break

            visited.add(best_neighbor)
            chain_nodes.append(node_map[best_neighbor])
            chain_edges.append(best_edge)
            current = best_neighbor

        if len(chain_nodes) < 2:
            return None

        return InferenceChain(
            chain_id=f"chain_{start_node.node_id}",
            nodes=chain_nodes,
            edges=chain_edges,
            chain_type="discovery",
            narrative=self._generate_narrative(chain_nodes, chain_edges),
            confidence=self._compute_chain_confidence(chain_nodes, chain_edges)
        )

    def _generate_narrative(self, nodes: List[LogicNode], edges: List[LogicEdge]) -> str:
        """生成推理链的自然语言叙述"""
        if not nodes:
            return ""

        parts = [f"观察到 {nodes[0].content[:100]}"]

        for i, edge in enumerate(edges):
            relation_word = {
                "causes": "导致",
                "implies": "意味着",
                "supports": "支持",
                "contradicts": "与...矛盾",
                "correlates": "与...相关"
            }.get(edge.relation_type, "关联到")

            if i + 1 < len(nodes):
                parts.append(f"{relation_word} {nodes[i+1].content[:80]}")

        return " -> ".join(parts)

    def _compute_chain_confidence(
        self,
        nodes: List[LogicNode],
        edges: List[LogicEdge]
    ) -> float:
        """计算推理链的置信度"""
        if not nodes:
            return 0.0

        # 节点置信度平均
        node_conf = sum(n.confidence for n in nodes) / len(nodes)

        # 边强度平均
        edge_conf = sum(e.strength for e in edges) / len(edges) if edges else 1.0

        # 长度惩罚
        length_penalty = 0.95 ** len(nodes)

        return node_conf * edge_conf * length_penalty

    def _enhance_chains_with_llm(
        self,
        chains: List[InferenceChain],
        node_map: Dict[str, LogicNode]
    ) -> List[InferenceChain]:
        """使用LLM增强推理链"""
        # 如果LLM服务不可用，直接返回原始链
        if not self.llm_service:
            return chains

        enhanced_chains = []

        for chain in chains[:5]:  # 限制处理数量
            prompt = self._build_enhancement_prompt(chain)

            try:
                result = self.llm_service.generate(
                    self.model,
                    prompt,
                    options={"num_predict": 512}
                )

                if result.get("success"):
                    response = result.get("response", "")
                    # 解析LLM响应增强叙述
                    if response:
                        chain.narrative = response[:500]
                        enhanced_chains.append(chain)
                        continue
            except Exception as e:
                logger.warning(f"LLM enhancement failed: {e}")

            enhanced_chains.append(chain)

        return enhanced_chains

    def _build_enhancement_prompt(self, chain: InferenceChain) -> str:
        """构建LLM增强提示"""
        nodes_desc = "\n".join([
            f"- {n.node_type}: {n.content[:100]}"
            for n in chain.nodes
        ])

        prompt = f"""你目前处于【严格上下文模式】。
请根据以下逻辑节点序列，生成一个简洁的中文推理叙述(不超过100字):

逻辑节点:
{nodes_desc}

要求:
1. 保持因果关系清晰
2. 使用学术语言
3. 不要编造信息
4. 用 -> 表示推理方向
5. 输出只能是 LogicChain 中 evidence/inference 节点重组
6. 若节点中未显示相关性，严禁使用“因此/表明/证明”等强因果词

推理叙述:"""

        return prompt

    def build_conclusion_chain(
        self,
        chains: List[InferenceChain],
        key_findings: List[str]
    ) -> InferenceChain:
        """从多条推理链构建结论链"""
        # 收集所有节点
        all_nodes = []
        all_edges = []

        for chain in chains:
            all_nodes.extend(chain.nodes)
            all_edges.extend(chain.edges)

        # 创建结论节点
        conclusion_node = LogicNode(
            node_id="final_conclusion",
            node_type="conclusion",
            content="; ".join(key_findings[:3]),
            source_type="inference",
            source_id="chain_synthesis",
            confidence=min(c.confidence for c in chains) if chains else 0.5,
            evidence_type="derived",
        )
        all_nodes.append(conclusion_node)

        # 创建从关键发现到结论的边
        for finding_idx, finding in enumerate(key_findings[:3]):
            edge = LogicEdge(
                edge_id=f"to_conclusion_{finding_idx}",
                from_node=f"finding_{finding_idx}",
                to_node="final_conclusion",
                relation_type="supports",
                strength=0.8,
                evidence=finding[:100]
            )
            all_edges.append(edge)

        return InferenceChain(
            chain_id="conclusion_chain",
            nodes=all_nodes[-5:] + [conclusion_node],
            edges=all_edges[-3:],
            chain_type="conclusion",
            narrative=f"综合分析得出: {conclusion_node.content}",
            confidence=conclusion_node.confidence
        )


class PaperDraftGenerator:
    """
    论文草稿生成器

    基于推理链和逻辑节点生成结构化论文草稿
    """

    def __init__(self, llm_service=None, model: str = "llama3"):
        self.llm_service = llm_service
        self.model = model

    def generate_draft(
        self,
        logic_chains: List[InferenceChain],
        nodes: List[LogicNode],
        project_context: str = "",
        style_guidelines: str = "",
        max_sections: int = 6
    ) -> PaperDraft:
        """生成论文草稿"""

        # 提取关键主张
        key_claims = self._extract_key_claims(logic_chains, nodes)

        # 生成标题
        title = self._generate_title(key_claims)

        # 生成摘要
        abstract = self._generate_abstract(logic_chains, key_claims)

        # 生成各章节
        sections = self._generate_sections(
            logic_chains,
            nodes,
            key_claims,
            style_guidelines
        )

        # 构建引用
        references = self._extract_references(nodes)

        return PaperDraft(
            title=title,
            abstract=abstract,
            sections=sections,
            logic_chains=logic_chains,
            key_claims=key_claims,
            references=references,
            metadata={
                "total_chains": len(logic_chains),
                "total_nodes": len(nodes),
                "generation_method": "logic_chain_agent"
            }
        )

    def _extract_key_claims(
        self,
        chains: List[InferenceChain],
        nodes: List[LogicNode]
    ) -> List[Dict[str, Any]]:
        """提取关键主张"""
        claims = []

        # 从结论节点提取
        for node in nodes:
            if node.node_type == "conclusion":
                claims.append({
                    "claim_id": f"claim_{len(claims)}",
                    "content": node.content,
                    "type": "conclusion",
                    "confidence": node.confidence,
                    "evidence": node.source_id
                })

        # 从推理链提取
        for chain in chains:
            if chain.chain_type == "conclusion":
                claims.append({
                    "claim_id": f"claim_{len(claims)}",
                    "content": chain.narrative,
                    "type": "chain_synthesis",
                    "confidence": chain.confidence,
                    "evidence": [n.node_id for n in chain.nodes]
                })

        # 从统计证据节点提取
        for node in nodes:
            if node.node_type == "evidence" and "statistical" in node.metadata.get("type", ""):
                claims.append({
                    "claim_id": f"claim_{len(claims)}",
                    "content": node.content,
                    "type": "statistical",
                    "confidence": node.confidence,
                    "evidence": node.source_id
                })

        return claims[:10]  # 限制主张数量

    def _generate_title(self, key_claims: List[Dict]) -> str:
        """生成论文标题"""
        if not key_claims:
            return "Research Paper Draft"

        # 提取关键术语
        all_words = []
        for claim in key_claims[:3]:
            words = re.findall(r'\b[A-Z][a-z]+\b', claim.get("content", ""))
            all_words.extend(words)

        if all_words:
            # 取最常见的关键词
            from collections import Counter
            word_counts = Counter(all_words)
            top_words = [w for w, _ in word_counts.most_common(3)]

            if len(top_words) >= 2:
                return f"{' '.join(top_words[:2])}: A Systematic Analysis"

        return "Medical Imaging Research: Analysis and Findings"

    def _generate_abstract(
        self,
        chains: List[InferenceChain],
        key_claims: List[Dict]
    ) -> str:
        """生成摘要"""
        if not self.llm_service:
            # 回退到简单拼接
            parts = []
            for chain in chains[:3]:
                parts.append(chain.narrative)
            return " ".join(parts) if parts else "Abstract will be generated based on the full paper."

        prompt = self._build_abstract_prompt(chains, key_claims)

        try:
            result = self.llm_service.generate(
                self.model,
                prompt,
                options={"num_predict": 300}
            )

            if result.get("success"):
                return result.get("response", "")[:500]
        except Exception as e:
            logger.warning(f"Abstract generation failed: {e}")

        return "Abstract generation pending."

    def _build_abstract_prompt(
        self,
        chains: List[InferenceChain],
        key_claims: List[Dict]
    ) -> str:
        """构建摘要生成提示"""
        claims_text = "\n".join([
            f"- {c.get('content', '')}"
            for c in key_claims[:5]
        ])

        chains_text = "\n".join([
            f"- {c.narrative}"
            for c in chains[:3]
        ])

        prompt = f"""Based on the following key claims and reasoning chains, write a concise abstract (150-200 words) for a medical research paper:

Key Claims:
{claims_text}

Reasoning Chains:
{chains_text}

Requirements:
1. Include: Background, Methods, Results, Conclusion
2. Use formal academic language
3. Be concise and precise
4. Do not fabricate data

Abstract:"""

        return prompt

    def _generate_sections(
        self,
        chains: List[InferenceChain],
        nodes: List[LogicNode],
        key_claims: List[Dict],
        style_guidelines: str = ""
    ) -> Dict[str, str]:
        """生成各章节内容"""
        sections = {}

        # 按类型组织节点
        evidence_nodes = [n for n in nodes if n.node_type == "evidence"]
        direct_evidence_nodes = [n for n in evidence_nodes if n.evidence_type == "direct"]
        inference_nodes = [n for n in nodes if n.node_type == "inference"]

        # Introduction
        sections["introduction"] = self._generate_introduction(key_claims, style_guidelines)

        # Methods
        sections["methods"] = self._generate_methods(evidence_nodes, style_guidelines)

        # Results
        sections["results"] = self._generate_results(chains, direct_evidence_nodes, style_guidelines)

        # Discussion
        sections["discussion"] = self._generate_discussion(chains, inference_nodes, style_guidelines)

        # Conclusion
        sections["conclusion"] = self._generate_conclusion(key_claims, style_guidelines)

        return sections

    def _generate_introduction(self, key_claims: List[Dict], style: str) -> str:
        """生成引言"""
        if not self.llm_service:
            return "Introduction content will be generated based on research context."

        claims_text = "\n".join([f"- {c.get('content', '')}" for c in key_claims[:3]])

        prompt = f"""Write an Introduction section for a medical research paper with the following key claims:

{claims_text}

{f'Style Guidelines: {style}' if style else ''}

Requirements:
1. Provide background context
2. State the research gap
3. Present research objectives
4. Keep within 300-400 words

Introduction:"""

        try:
            result = self.llm_service.generate(self.model, prompt, options={"num_predict": 400})
            if result.get("success"):
                return result.get("response", "")[:600]
        except:
            pass

        return "Introduction to be expanded based on research context."

    def _generate_methods(self, evidence_nodes: List[LogicNode], style: str) -> str:
        """生成方法章节"""
        if not self.llm_service:
            # 回退到简单描述
            sources = set(n.source_type for n in evidence_nodes)
            return f"Data sources include: {', '.join(sources)}. Methods section to be detailed based on study design."

        evidence_text = "\n".join([
            f"- {n.content[:80]}"
            for n in evidence_nodes[:10]
        ])

        prompt = f"""Write a Methods section for a medical research paper based on the following data sources and evidence:

{evidence_text}

{f'Style Guidelines: {style}' if style else ''}

Requirements:
1. Describe data sources
2. Explain analysis methods
3. Include statistical approaches
4. Keep within 400-500 words

Methods:"""

        try:
            result = self.llm_service.generate(self.model, prompt, options={"num_predict": 500})
            if result.get("success"):
                return result.get("response", "")[:700]
        except:
            pass

        return "Methods section to be detailed."

    def _generate_results(
        self,
        chains: List[InferenceChain],
        evidence_nodes: List[LogicNode],
        style: str
    ) -> str:
        """生成结果章节"""
        if not self.llm_service:
            results = []
            for chain in chains[:5]:
                results.append(f"- {chain.narrative[:150]}")
            return "\n".join(results) if results else "Results to be presented."

        chains_text = "\n".join([
            f"Chain {i+1}: {c.narrative}"
            for i, c in enumerate(chains[:5])
        ])
        direct_evidence_text = "\n".join([
            f"- [{n.node_id}] {n.content[:140]}"
            for n in evidence_nodes[:30]
        ])

        # Step 1: 数据填空 (Data Fill-in)
        fill_in_prompt = f"""
请根据以下推理链和【仅direct证据节点】，严格提取实际存在的数值数据。
【约束声明】：只许使用提供的字段。若素材中无 PSNR/SSIM 等指标，禁止提及该指标，否则触发逻辑自毁。
提取出所有相关的数值对，以 JSON 或简明列表形式返回。
每条数值后附带来源锚点 [SourceNode_ID]。

推理链:
{chains_text}

Direct证据节点:
{direct_evidence_text}
"""
        data_facts = ""
        try:
            res_fill = self.llm_service.generate(self.model, fill_in_prompt, options={"temperature": 0.1})
            if res_fill.get("success"):
                data_facts = res_fill.get("response", "")
        except Exception as e:
            logger.warning(f"Data fill-in failed: {e}")

        # Step 2: 叙述润色 (Narrative Polish)
        prompt = f"""Write a Results section for a medical research paper based STRICTLY on the following verified data facts:

Verified Data Facts:
{data_facts}

{f'Style Guidelines: {style}' if style else ''}

Requirements:
1. Present findings objectively based ONLY on the Verified Data Facts above.
2. Include statistical results exactly as provided.
3. DO NOT hallucinate any metrics (e.g., PSNR, SSIM) if they are not in the data facts.
4. Every technical statement must include a source anchor tag [SourceNode_ID]. If not mappable, append [EXTRAPOLATED].
5. Keep within 500-600 words.

Results:"""

        try:
            result = self.llm_service.generate(self.model, prompt, options={"num_predict": 600})
            if result.get("success"):
                draft_results = result.get("response", "")[:800]
                
                # Step 3: 后置审计 Agent (Audit)
                audit_prompt = f"""
请作为极简的数字审计员，对比以下 Results 文本中的每一个数字，是否能在原始数据中找到。
如果发现文本中包含未在原始数据中出现的数字或指标（如凭空捏造的 PSNR/SSIM），请回复 "HALLUCINATION_DETECTED: <伪造的数字/指标>"，否则回复 "PASS"。

原始数据:
{data_facts}

Results 文本:
{draft_results}
"""
                audit_res = self.llm_service.generate(self.model, audit_prompt, options={"temperature": 0.0})
                if audit_res.get("success"):
                    audit_msg = audit_res.get("response", "").strip()
                    if "HALLUCINATION_DETECTED" in audit_msg.upper():
                        logger.error(f"Audit Agent triggered: {audit_msg}")
                        return f"[AUDIT FAILED - HALLUCINATED DATA REMOVED]\n{draft_results}\n\n(Audit Info: {audit_msg})"
                return draft_results
        except:
            pass

        return "Results section to be detailed based on data facts."

    def _generate_discussion(
        self,
        chains: List[InferenceChain],
        inference_nodes: List[LogicNode],
        style: str
    ) -> str:
        """生成讨论章节"""
        if not self.llm_service:
            return "Discussion to be generated based on results and literature context."

        inferences_text = "\n".join([
            f"- {n.content[:100]}"
            for n in inference_nodes[:8]
        ])

        prompt = f"""Write a Discussion section for a medical research paper based on the following inference chains:

{inferences_text}

{f'Style Guidelines: {style}' if style else ''}

Requirements:
1. Interpret key findings
2. Compare with literature
3. Discuss implications
4. Acknowledge limitations
5. Keep within 500-600 words

Discussion:"""

        try:
            result = self.llm_service.generate(self.model, prompt, options={"num_predict": 600})
            if result.get("success"):
                return result.get("response", "")[:800]
        except:
            pass

        return "Discussion section to be developed."

    def _generate_conclusion(self, key_claims: List[Dict], style: str) -> str:
        """生成结论"""
        claims_text = "; ".join([c.get("content", "")[:50] for c in key_claims[:3]])

        return f"In conclusion, this study demonstrates that {claims_text}. Further research is warranted to validate these findings."

    def _extract_references(self, nodes: List[LogicNode]) -> List[Dict[str, str]]:
        """从节点提取引用"""
        references = []

        for node in nodes:
            if node.source_type == "literature" and node.source_id:
                references.append({
                    "id": node.source_id,
                    "source": node.content[:100]
                })

        return references[:20]  # 限制引用数量


class LogicChainAgent:
    """
    逻辑链Agent主类

    整合所有组件，提供完整的逻辑链提取和论文生成流程
    """

    def __init__(
        self,
        llm_service=None,
        model: str = "llama3",
        knowledge_base=None,
        progress_callback=None,
        stream_callback=None
    ):
        self.llm_service = llm_service
        self.model = model
        self.kb = knowledge_base
        self.progress_cb = progress_callback
        self.stream_cb = stream_callback

        # 初始化子组件
        self.extractor = LogicChainExtractor(llm_service)
        self.chain_builder = InferenceChainBuilder(llm_service, model)
        self.draft_generator = PaperDraftGenerator(llm_service, model)

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

    def process(
        self,
        ocr_data: List[Dict] = None,
        dicom_data: List[Dict] = None,
        text_data: List[Dict] = None,
        structured_data: Dict = None,
        project_context: str = "",
        style_guidelines: str = "",
        strict_mode: bool = True,
        project_scope: str = "",
    ) -> Dict[str, Any]:
        """
        执行完整的逻辑链提取和论文生成流程

        Args:
            ocr_data: OCR识别数据列表
            dicom_data: DICOM影像数据列表
            text_data: 文本数据列表
            structured_data: 结构化统计结果
            project_context: 项目上下文描述
            style_guidelines: 写作风格指南

        Returns:
            包含推理链和论文草稿的字典
        """
        self._progress("Starting Logic Chain Agent...")
        self._stream("\n> 🧠 [LogicChainAgent]: 初始化逻辑链提取...\n")

        # Phase 1: 提取逻辑节点和边
        self._progress("Phase 1: Extracting logic nodes from data sources...")
        self._stream("> 📊 [LogicExtractor]: 从数据源提取逻辑节点...\n")

        nodes, edges = self.extractor.extract_all(
            ocr_data=ocr_data,
            dicom_data=dicom_data,
            text_data=text_data,
            structured_data=structured_data
        )

        self._stream(f"> 📊 [LogicExtractor]: 提取了 {len(nodes)} 个逻辑节点, {len(edges)} 条边\n")

        # Phase 2: 构建推理链
        self._progress("Phase 2: Building inference chains...")
        self._stream("> 🔗 [ChainBuilder]: 构建推理链...\n")

        effective_scope = project_scope or self._infer_project_scope(project_context, nodes)
        chains = self.chain_builder.build_chains(
            nodes,
            edges,
            project_scope=effective_scope,
            strict_mode=strict_mode,
        )
        self._stream(f"> 🔗 [ChainBuilder]: 构建了 {len(chains)} 条推理链\n")

        # Phase 3: 构建结论链
        self._progress("Phase 3: Synthesizing conclusion chain...")
        self._stream("> 🎯 [ChainBuilder]: 综合结论链...\n")

        key_findings = self._extract_key_findings(chains, nodes)
        conclusion_chain = self.chain_builder.build_conclusion_chain(chains, key_findings)
        chains.append(conclusion_chain)

        # Phase 4: 生成论文草稿
        self._progress("Phase 4: Generating paper draft...")
        self._stream("> 📝 [DraftGenerator]: 生成论文草稿...\n")

        draft = self.draft_generator.generate_draft(
            logic_chains=chains,
            nodes=nodes,
            project_context=project_context,
            style_guidelines=style_guidelines
        )

        self._stream(f"> 📝 [DraftGenerator]: 生成了标题: {draft.title}\n")

        # Phase 5: 返回结果
        self._progress("Logic Chain Agent completed successfully.")
        self._stream("> ✅ [LogicChainAgent]: 流程完成!\n")

        return {
            "success": True,
            "logic_chains": [c.to_dict() for c in chains],
            "nodes": [n.to_dict() for n in nodes],
            "edges": [e.to_dict() for e in edges],
            "paper_draft": draft.to_dict(),
            "key_findings": key_findings,
            "project_scope": effective_scope,
            "strict_mode": strict_mode,
            "statistics": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "total_chains": len(chains),
                "evidence_nodes": len([n for n in nodes if n.node_type == "evidence"]),
                "inference_nodes": len([n for n in nodes if n.node_type == "inference"])
            }
        }

    def _infer_project_scope(self, project_context: str, nodes: List[LogicNode]) -> str:
        t = str(project_context or "").lower()
        if "pcct" in t or "photon counting" in t:
            return "Medical/PCCT"
        if "mri" in t:
            return "Medical/MRI"
        if any(k in t for k in ["ct", "dicom", "hounsfield", "hu", "radiology"]):
            return "Medical/CT"
        for n in nodes[:80]:
            ec = str(n.entity_context or "")
            if ec and ec != "General":
                return ec
        return "Medical/CT"

    def _extract_key_findings(
        self,
        chains: List[InferenceChain],
        nodes: List[LogicNode]
    ) -> List[str]:
        """从推理链提取关键发现"""
        findings = []

        # 从结论链提取
        for chain in chains:
            if chain.chain_type == "conclusion":
                findings.append(chain.narrative)

        # 从高置信度节点提取
        for node in nodes:
            if node.confidence >= 0.9 and node.node_type == "evidence":
                findings.append(node.content[:100])

        return findings[:5]

    def get_visualization_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成可视化数据格式

        用于前端展示逻辑链图谱
        """
        nodes = result.get("nodes", [])
        edges = result.get("edges", [])

        # 转换为可视化格式
        vis_nodes = []
        for node in nodes:
            vis_nodes.append({
                "id": node.get("node_id"),
                "label": node.get("content", "")[:50],
                "type": node.get("node_type"),
                "source": node.get("source_type"),
                "confidence": node.get("confidence"),
            })

        vis_edges = []
        for edge in edges:
            vis_edges.append({
                "from": edge.get("from_node"),
                "to": edge.get("to_node"),
                "label": edge.get("relation_type"),
                "strength": edge.get("strength"),
            })

        return {
            "nodes": vis_nodes,
            "edges": vis_edges,
            "chains": result.get("logic_chains", [])
        }


# 便捷函数
def create_logic_chain_agent(
    llm_service=None,
    model: str = "llama3",
    knowledge_base=None,
    progress_callback=None,
    stream_callback=None
) -> LogicChainAgent:
    """创建LogicChainAgent实例"""
    return LogicChainAgent(
        llm_service=llm_service,
        model=model,
        knowledge_base=knowledge_base,
        progress_callback=progress_callback,
        stream_callback=stream_callback
    )


def process_patient_case(
    patient_case,
    llm_service=None,
    model: str = "llama3",
    project_context: str = ""
) -> Dict[str, Any]:
    """
    处理单个患者病例数据的便捷函数

    Args:
        patient_case: PatientCase实例
        llm_service: LLM服务
        model: 模型名称
        project_context: 项目上下文

    Returns:
        处理结果字典
    """
    agent = LogicChainAgent(llm_service=llm_service, model=model)

    return agent.process(
        ocr_data=patient_case.ocr_data if hasattr(patient_case, 'ocr_data') else None,
        dicom_data=patient_case.dicom_data if hasattr(patient_case, 'dicom_data') else None,
        text_data=patient_case.text_data if hasattr(patient_case, 'text_data') else None,
        project_context=project_context
    )
