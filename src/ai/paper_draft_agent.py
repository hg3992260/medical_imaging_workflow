"""
PaperDraftAgent - 论文草稿生成Agent

按模板风格和章节整合多个数据源内容，生成结构化论文草稿。

架构设计:
1. TemplateStyleParser   - 解析模板风格(section结构、引用格式、用词风格)
2. DataSourceAggregator  - 聚合多个数据源内容
3. SectionContentBuilder - 按章节构建内容
4. PaperDraftAgent       - 主Agent整合流程

Author: RSNA Medical Imaging Workflow
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from src.ai.pdf_template_parser import parse_pdf_sections_from_text, build_style_profile, summarize_pdf_template
from src.ai.agent_skills.template_fitting import TemplateFitter
from src.ai.logic_chain_agent import LogicChainExtractor, InferenceChainBuilder, PaperDraftGenerator, LogicChainAgent
from src.services.ollama_service import OllamaService

logger = logging.getLogger(__name__)


@dataclass
class SectionContent:
    """章节内容"""
    section_name: str
    content: str
    source_nodes: List[str] = field(default_factory=list)
    confidence: float = 1.0
    style_hints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemplateStyle:
    """模板风格配置"""
    journal_name: str = ""
    structure: List[str] = field(default_factory=list)
    citation_style: str = "Unknown"
    max_word_count: Dict[str, int] = field(default_factory=dict)
    style_profiles: Dict[str, Any] = field(default_factory=dict)
    raw_sections: Dict[str, str] = field(default_factory=dict)


class TemplateStyleParser:
    """
    解析模板风格 - 从PDF模板或文本中提取风格配置
    """

    def __init__(self, llm_service=None, model: str = "llama3"):
        self.llm_service = llm_service
        self.model = model

    def parse_from_pdf(self, pdf_text: str) -> TemplateStyle:
        """从PDF文本解析模板风格"""
        sections = parse_pdf_sections_from_text(pdf_text)
        summary = summarize_pdf_template(sections)

        return TemplateStyle(
            journal_name="Detected Template",
            structure=summary.get("detected_structure", []),
            citation_style=summary.get("citation_style_hint", "Unknown"),
            max_word_count={},
            style_profiles=summary.get("style_profiles", {}),
            raw_sections=sections
        )

    def parse_from_template_config(self, config: Dict[str, Any]) -> TemplateStyle:
        """从模板配置字典解析风格"""
        return TemplateStyle(
            journal_name=config.get("journal_name", ""),
            structure=config.get("structure", []),
            citation_style=config.get("citation_style", "Unknown"),
            max_word_count=config.get("max_word_count", {}),
            style_profiles={},
            raw_sections={}
        )

    def enhance_style_with_llm(self, style: TemplateStyle) -> TemplateStyle:
        """使用LLM增强风格描述"""
        if not self.llm_service or not style.structure:
            return style

        # 基于已有结构生成风格描述
        prompt = f"""分析以下论文模板的风格特征，生成详细的写作风格指南:

期刊/模板: {style.journal_name}
结构: {', '.join(style.structure)}
引用格式: {style.citation_style}

请生成:
1. 各章节的典型写作模式(被动/主动、学术用语特征)
2. 常见的句式结构特征
3. 数据呈现风格(详细vs简洁)
4. 结论部分的典型写法

保持简洁，用中文回答。"""

        try:
            result = self.llm_service.generate(
                self.model,
                prompt,
                options={"num_predict": 400}
            )
            if result.get("success"):
                style.style_hints = {
                    "llm_style_guide": result.get("response", "")[:800]
                }
        except Exception as e:
            logger.warning(f"Style enhancement failed: {e}")

        return style


class DataSourceAggregator:
    """
    聚合多个数据源内容
    支持: OCR、DICOM、文档、表格、ROI、文献
    """

    def __init__(self, project_context_indexer=None, db_service=None):
        self.indexer = project_context_indexer
        self.db_service = db_service

    def aggregate(
        self,
        project_id: str,
        source_types: List[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        聚合项目数据源

        Args:
            project_id: 项目ID
            source_types: 要聚合的数据源类型列表，为空则聚合全部

        Returns:
            按类型组织的原始数据字典
        """
        if source_types is None:
            source_types = ["ocr", "dicom", "document", "roi", "tabular", "sample"]

        aggregated = {}

        # 通过indexer获取各数据源
        if self.indexer:
            for source_type in source_types:
                data = self._fetch_by_source_type(project_id, source_type)
                if data:
                    aggregated[source_type] = data

        return aggregated

    def _fetch_by_source_type(self, project_id: str, source_type: str) -> List[Dict[str, Any]]:
        """根据类型获取数据"""
        # ProjectContextIndexer 返回的是迭代器
        # 这里简化处理，实际应该直接查询
        results = []

        try:
            if source_type == "ocr":
                rows = self.db_service.execute_query(
                    """SELECT r.result_id, r.recognized_text, r.confidence, r.created_at
                       FROM ocr_results r JOIN ocr_sessions s ON r.session_id=s.session_id
                       WHERE s.project_id = ? ORDER BY r.created_at DESC""",
                    (project_id,)
                )
                for r in rows or []:
                    results.append({
                        "source_type": "ocr",
                        "source_id": r.get("result_id", ""),
                        "text": r.get("recognized_text", ""),
                        "confidence": r.get("confidence", 1.0),
                        "created_at": str(r.get("created_at", ""))
                    })

            elif source_type == "dicom":
                rows = self.db_service.execute_query(
                    """SELECT session_id, user_id, dicom_info, file_path, created_at
                       FROM dicom_sessions WHERE project_id = ?""",
                    (project_id,)
                )
                for r in rows or []:
                    results.append({
                        "source_type": "dicom",
                        "source_id": r.get("session_id", ""),
                        "dicom_info": r.get("dicom_info", ""),
                        "file_path": r.get("file_path", ""),
                        "created_at": str(r.get("created_at", ""))
                    })

            elif source_type == "document":
                rows = self.db_service.execute_query(
                    """SELECT d.doc_id, d.file_name, d.content, d.analysis_result, d.created_at
                       FROM documents d JOIN text_sessions t ON d.session_id=t.session_id
                       WHERE t.project_id = ? ORDER BY d.created_at DESC""",
                    (project_id,)
                )
                for r in rows or []:
                    results.append({
                        "source_type": "document",
                        "source_id": r.get("doc_id", ""),
                        "file_name": r.get("file_name", ""),
                        "content": r.get("content", ""),
                        "analysis_result": r.get("analysis_result", ""),
                        "created_at": str(r.get("created_at", ""))
                    })

            elif source_type == "roi":
                rows = self.db_service.execute_query(
                    """SELECT rd.roi_id, rd.roi_type, rd.area, rd.perimeter, rd.ocr_text, rd.properties
                       FROM roi_data rd JOIN dicom_sessions ds ON rd.session_id=ds.session_id
                       WHERE ds.project_id = ? ORDER BY rd.created_at DESC""",
                    (project_id,)
                )
                for r in rows or []:
                    results.append({
                        "source_type": "roi",
                        "source_id": r.get("roi_id", ""),
                        "roi_type": r.get("roi_type", ""),
                        "area": r.get("area"),
                        "perimeter": r.get("perimeter"),
                        "ocr_text": r.get("ocr_text", ""),
                        "properties": r.get("properties", ""),
                    })

        except Exception as e:
            logger.warning(f"Failed to fetch {source_type}: {e}")

        return results

    def merge_to_text(self, aggregated: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, str]]:
        """将各数据源合并为文本列表，供后续处理"""
        texts = []

        for source_type, items in aggregated.items():
            for item in items:
                text = ""
                if source_type == "ocr":
                    text = item.get("text", "")
                elif source_type == "document":
                    text = item.get("content", "")
                elif source_type == "roi":
                    text = item.get("ocr_text", "")

                if text:
                    texts.append({
                        "content": text,
                        "source_type": source_type,
                        "source_id": item.get("source_id", "")
                    })

        return texts


class SectionContentBuilder:
    """
    按章节构建内容 - 将数据源内容分配到论文各章节
    """

    # 章节关键词映射
    SECTION_KEYWORDS = {
        "introduction": ["背景", "介绍", "研究背景", "目的", "objective", "background", "introduction"],
        "methods": ["方法", "材料", "方法学", "实验", "materials", "methods", "methodology"],
        "results": ["结果", "发现", "数据", "分析", "results", "findings", "data"],
        "discussion": ["讨论", "解释", "意义", "limitation", "discussion", "interpretation"],
        "conclusion": ["结论", "总结", "总结", "conclusion", "summary"]
    }

    def __init__(self, llm_service=None, model: str = "llama3"):
        self.llm_service = llm_service
        self.model = model

    def assign_content_to_sections(
        self,
        text_sources: List[Dict[str, str]],
        template_structure: List[str],
        style: TemplateStyle
    ) -> Dict[str, SectionContent]:
        """
        将内容分配到各章节

        Args:
            text_sources: 文本数据源列表
            template_structure: 目标模板结构
            style: 模板风格

        Returns:
            章节内容字典
        """
        sections = {}

        # 初始化各章节
        for sec in template_structure:
            sec_key = self._normalize_section(sec)
            sections[sec_key] = SectionContent(
                section_name=sec_key,
                content="",
                source_nodes=[]
            )

        # 使用LLM进行智能分配
        if self.llm_service:
            return self._llm_assign_content(text_sources, sections, style)

        # 回退: 基于关键词分配
        return self._keyword_assign_content(text_sources, sections)

    def _normalize_section(self, section: str) -> str:
        """标准化章节名称"""
        s = section.lower().strip()
        for key, aliases in self.SECTION_KEYWORDS.items():
            for alias in aliases:
                if alias in s:
                    return key
        return s

    def _llm_assign_content(
        self,
        text_sources: List[Dict[str, str]],
        sections: Dict[str, SectionContent],
        style: TemplateStyle
    ) -> Dict[str, SectionContent]:
        """使用LLM智能分配内容到章节"""

        # 构建内容摘要(避免token过多)
        content_summary = []
        for i, src in enumerate(text_sources[:20]):  # 限制数量
            content_summary.append({
                "index": i,
                "type": src.get("source_type", ""),
                "preview": src.get("content", "")[:200]
            })

        sections_list = list(sections.keys())
        prompt = f"""你是一个医学论文写作助手。将以下内容片段分配到合适的论文章节。

目标章节结构: {sections_list}
期刊风格: {style.journal_name}

内容片段:
{json.dumps(content_summary, ensure_ascii=False, indent=2)}

任务:
为每个章节选择最相关的3-5个内容片段索引，返回JSON格式:
{{
  "introduction": [0, 3, 5],
  "methods": [1, 2],
  ...
}}

只返回JSON，不要其他文字。"""

        try:
            result = self.llm_service.generate(
                self.model,
                prompt,
                options={"num_predict": 300}
            )

            if result.get("success"):
                response = result.get("response", "")
                # 解析JSON响应
                assignment = self._parse_json_response(response)
                if assignment:
                    return self._apply_assignment(text_sources, sections, assignment)
        except Exception as e:
            logger.warning(f"LLM assignment failed: {e}")

        return sections

    def _parse_json_response(self, response: str) -> Optional[Dict[str, List[int]]]:
        """解析LLM返回的JSON"""
        try:
            # 尝试直接解析
            return json.loads(response)
        except:
            pass

        # 尝试提取JSON块
        match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass

        return None

    def _apply_assignment(
        self,
        text_sources: List[Dict[str, str]],
        sections: Dict[str, SectionContent],
        assignment: Dict[str, List[int]]
    ) -> Dict[str, SectionContent]:
        """应用分配结果"""
        for sec_key, indices in assignment.items():
            if sec_key not in sections:
                continue

            contents = []
            node_ids = []
            for idx in indices:
                if 0 <= idx < len(text_sources):
                    src = text_sources[idx]
                    contents.append(src.get("content", ""))
                    node_ids.append(src.get("source_id", f"src_{idx}"))

            sections[sec_key].content = "\n\n".join(contents)
            sections[sec_key].source_nodes = node_ids

        return sections

    def _keyword_assign_content(
        self,
        text_sources: List[Dict[str, str]],
        sections: Dict[str, SectionContent]
    ) -> Dict[str, SectionContent]:
        """基于关键词的内容分配"""
        for src in text_sources:
            content = src.get("content", "").lower()
            best_section = None
            best_score = 0

            for sec, keywords in self.SECTION_KEYWORDS.items():
                score = sum(1 for kw in keywords if kw in content)
                if score > best_score and sec in sections:
                    best_score = score
                    best_section = sec

            if best_section and best_score > 0:
                sec = sections[best_section]
                sec.content += "\n\n" + src.get("content", "")
                sec.source_nodes.append(src.get("source_id", ""))

        return sections


class PaperDraftAgent:
    """
    论文草稿生成Agent主类

    整合模板风格、数据源聚合、章节构建，生成符合目标期刊要求的论文草稿
    """

    def __init__(
        self,
        llm_service: OllamaService = None,
        db_service = None,
        project_context_indexer = None,
        model: str = "llama3",
        progress_callback=None,
        stream_callback=None
    ):
        self.llm_service = llm_service
        self.db_service = db_service
        self.indexer = project_context_indexer
        self.model = model
        self.progress_cb = progress_callback
        self.stream_cb = stream_callback

        # 初始化子组件
        self.style_parser = TemplateStyleParser(llm_service, model)
        self.data_aggregator = DataSourceAggregator(project_context_indexer, db_service)
        self.section_builder = SectionContentBuilder(llm_service, model)
        self.template_fitter = TemplateFitter(llm_service, model) if llm_service else None

        # 复用LogicChainAgent的提取器
        self.logic_extractor = LogicChainExtractor(llm_service)
        self.chain_builder = InferenceChainBuilder(llm_service, model)
        self.draft_generator = PaperDraftGenerator(llm_service, model)

    def _progress(self, message: str):
        if self.progress_cb:
            try:
                self.progress_cb(message)
            except:
                pass
        logger.info(message)

    def _stream(self, token: str):
        if self.stream_cb:
            try:
                self.stream_cb(token)
            except:
                pass

    def generate(
        self,
        project_id: str,
        template_source: Any = None,
        template_type: str = "config",  # "pdf", "config", "journal"
        journal_name: str = "",
        custom_sections: List[str] = None,
        source_types: List[str] = None,
        project_context: str = ""
    ) -> Dict[str, Any]:
        """
        生成论文草稿

        Args:
            project_id: 项目ID
            template_source: 模板源(pdf文本、配置dict、或期刊名)
            template_type: 模板类型 ("pdf", "config", "journal")
            journal_name: 期刊名称(当template_type="journal"时使用)
            custom_sections: 自定义章节结构
            source_types: 要聚合的数据源类型
            project_context: 项目上下文描述

        Returns:
            包含论文草稿和元数据的字典
        """
        self._progress("Starting PaperDraftAgent...")
        self._stream("\n> 📄 [PaperDraftAgent]: 初始化论文生成...\n")

        # Phase 1: 解析模板风格
        self._progress("Phase 1: Parsing template style...")
        self._stream("> 🎨 [StyleParser]: 解析模板风格...\n")

        style = self._parse_template(template_source, template_type, journal_name)
        self._stream(f"> 🎨 [StyleParser]: 期刊: {style.journal_name}, 结构: {len(style.structure)} 个章节\n")

        # Phase 2: 聚合数据源
        self._progress("Phase 2: Aggregating data sources...")
        self._stream("> 📊 [DataAggregator]: 聚合数据源...\n")

        aggregated = self.data_aggregator.aggregate(project_id, source_types)
        text_sources = self.data_aggregator.merge_to_text(aggregated)
        self._stream(f"> 📊 [DataAggregator]: 聚合了 {len(text_sources)} 条文本内容\n")

        # Phase 3: 提取逻辑节点
        self._progress("Phase 3: Extracting logic nodes...")
        self._stream("> 🔍 [LogicExtractor]: 提取逻辑节点...\n")

        nodes, edges = self.logic_extractor.extract_all(
            text_data=[{"content": s.get("content", ""), "source_id": s.get("source_id", "")} for s in text_sources]
        )
        self._stream(f"> 🔍 [LogicExtractor]: 提取了 {len(nodes)} 个节点, {len(edges)} 条边\n")

        # Phase 4: 构建推理链
        self._progress("Phase 4: Building inference chains...")
        self._stream("> 🔗 [ChainBuilder]: 构建推理链...\n")

        chains = self.chain_builder.build_chains(nodes, edges)

        # Phase 5: 分配内容到章节
        self._progress("Phase 5: Assigning content to sections...")
        self._stream("> 📑 [SectionBuilder]: 构建章节内容...\n")

        target_sections = custom_sections or style.structure or [
            "introduction", "methods", "results", "discussion", "conclusion"
        ]

        sections = self.section_builder.assign_content_to_sections(
            text_sources, target_sections, style
        )

        # Phase 6: 生成论文草稿
        self._progress("Phase 6: Generating paper draft...")
        self._stream("> 📝 [DraftGenerator]: 生成论文...\n")

        draft = self.draft_generator.generate_draft(
            logic_chains=chains,
            nodes=nodes,
            project_context=project_context,
            style_guidelines=style.style_hints.get("llm_style_guide", "")
        )

        # 用章节内容替换
        for sec_key, sec_content in sections.items():
            if sec_content.content and sec_key in draft.sections:
                # 拼接: LLM生成内容 + 原始数据内容
                combined = self._combine_content(
                    draft.sections.get(sec_key, ""),
                    sec_content.content,
                    style
                )
                draft.sections[sec_key] = combined

        # Phase 7: 应用模板约束
        if self.template_fitter and journal_name:
            self._progress("Phase 7: Fitting to journal template...")
            self._stream(f"> 📐 [TemplateFitter]: 应用 {journal_name} 约束...\n")

            draft = self._apply_template_fit(draft, journal_name, style)

        self._progress("PaperDraftAgent completed successfully.")
        self._stream("> ✅ [PaperDraftAgent]: 生成完成!\n")

        return {
            "success": True,
            "draft": draft.to_dict(),
            "style": {
                "journal_name": style.journal_name,
                "structure": style.structure,
                "citation_style": style.citation_style,
                "style_hints": style.style_hints
            },
            "sections": {k: v.content for k, v in sections.items()},
            "statistics": {
                "total_nodes": len(nodes),
                "total_chains": len(chains),
                "content_sources": len(text_sources),
                "aggregated_sources": {k: len(v) for k, v in aggregated.items()}
            }
        }

    def _parse_template(
        self,
        template_source: Any,
        template_type: str,
        journal_name: str
    ) -> TemplateStyle:
        """解析模板"""
        if template_type == "pdf" and isinstance(template_source, str):
            return self.style_parser.parse_from_pdf(template_source)

        elif template_type == "config" and isinstance(template_source, dict):
            return self.style_parser.parse_from_template_config(template_source)

        elif template_type == "journal" or journal_name:
            target = journal_name or template_source
            if self.template_fitter:
                config = self.template_fitter.get_template_constraints(target)
                return self.style_parser.parse_from_template_config(config)

        # 默认模板
        return TemplateStyle(
            journal_name="General",
            structure=["introduction", "methods", "results", "discussion", "conclusion"],
            citation_style="Unknown"
        )

    def _combine_content(
        self,
        generated: str,
        raw: str,
        style: TemplateStyle
    ) -> str:
        """合并LLM生成内容和原始数据内容"""
        if not raw:
            return generated

        if not generated:
            return raw[:3000]  # 限制长度

        # 简单拼接，实际应该更智能地融合
        return f"{generated}\n\n---\n原始数据补充:\n{raw[:1000]}"

    def _apply_template_fit(
        self,
        draft,
        journal_name: str,
        style: TemplateStyle
    ):
        """应用模板约束"""
        # 合并所有章节内容
        full_text = draft.abstract + "\n\n"
        for sec_name, sec_content in draft.sections.items():
            full_text += f"\n## {sec_name.capitalize()}\n{sec_content}\n"

        # 使用TemplateFitter拟合
        result = self.template_fitter.fit_draft(full_text, journal_name)

        # 重新解析拟合后的文本(简化处理)
        fitted_text = result.get("fitted_text", full_text)

        # 更新draft
        parts = fitted_text.split("##")
        for part in parts:
            part = part.strip()
            if not part:
                continue

            lines = part.split("\n")
            if lines:
                sec_name = lines[0].strip().lower()
                sec_content = "\n".join(lines[1:]).strip()

                if sec_name in draft.sections:
                    draft.sections[sec_name] = sec_content

        return draft


def create_paper_draft_agent(
    llm_service: OllamaService = None,
    db_service = None,
    project_context_indexer = None,
    model: str = "llama3",
    progress_callback=None,
    stream_callback=None
) -> PaperDraftAgent:
    """创建PaperDraftAgent实例"""
    return PaperDraftAgent(
        llm_service=llm_service,
        db_service=db_service,
        project_context_indexer=project_context_indexer,
        model=model,
        progress_callback=progress_callback,
        stream_callback=stream_callback
    )
