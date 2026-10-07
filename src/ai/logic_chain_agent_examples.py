"""
Logic Chain Agent 使用示例

展示如何使用逻辑链Agent从多源数据提取逻辑链条并生成论文草稿

Author: RSNA Medical Imaging Workflow
"""

# 示例 1: 基本使用
def basic_usage_example():
    """
    基本使用示例 - 直接处理数据
    """
    from src.ai.logic_chain_agent import create_logic_chain_agent

    # 创建Agent
    agent = create_logic_chain_agent(
        llm_service=None,  # 可选：提供LLM服务增强推理
        model="llama3",
        progress_callback=lambda msg: print(f"[Progress] {msg}"),
        stream_callback=lambda token: print(token, end='', flush=True)
    )

    # 准备数据
    ocr_data = [
        {
            "image_name": "chart1.png",
            "text": "Patient survival rate: 85% (p<0.001). Treatment group showed significant improvement compared to control.",
            "confidence": 0.95
        },
        {
            "image_name": "table1.png",
            "text": '{"metric": "accuracy", "value": "92.5%", "confidence_interval": "[89.2%, 95.8%]"}',
            "confidence": 0.90
        }
    ]

    dicom_data = [
        {
            "file_name": "scan001.dcm",
            "dicom_info": '{"Modality": "CT", "SliceThickness": "1.5mm"}',
            "roi_data": [
                {"roi_type": "tumor", "roi_name": "lesion_1", "area": 125.5}
            ]
        }
    ]

    text_data = [
        {
            "file_name": "clinical_notes.txt",
            "content": "The patient underwent CT scanning. Results indicate that tumor size correlates with treatment response (r=0.75). However, some cases showed contradictory findings."
        }
    ]

    # 执行处理
    result = agent.process(
        ocr_data=ocr_data,
        dicom_data=dicom_data,
        text_data=text_data,
        project_context="Medical imaging study on tumor detection",
        style_guidelines="Use formal academic language. Include statistical evidence."
    )

    # 输出结果
    print("\n=== 处理结果 ===")
    print(f"成功: {result['success']}")
    print(f"逻辑链数量: {len(result['logic_chains'])}")
    print(f"节点数量: {len(result['nodes'])}")
    print(f"关键发现: {result['key_findings']}")

    # 打印论文草稿
    draft = result['paper_draft']
    print(f"\n标题: {draft['title']}")
    print(f"摘要: {draft['abstract'][:200]}...")

    return result


# 示例 2: 与数据聚合服务集成
def data_aggregation_example(project_id: str):
    """
    与DataAggregationService集成的示例
    """
    from src.services.database_service import DatabaseService
    from src.services.data_aggregation_service import DataAggregationService
    from src.ai.logic_chain_agent import LogicChainAgent

    # 初始化服务
    db_service = DatabaseService()
    aggregator = DataAggregationService(db_service)

    # 获取患者数据
    patient_cases = aggregator.aggregate_project_data(project_id)

    # 准备数据
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

    # 创建并运行Agent
    agent = LogicChainAgent()
    result = agent.process(
        ocr_data=all_ocr_data,
        dicom_data=all_dicom_data,
        text_data=all_text_data
    )

    return result


# 示例 3: 与AgentCoreLoop集成
def core_loop_integration_example(project_id: str):
    """
    与AgentCoreLoopExecutor集成的示例
    """
    from src.services.database_service import DatabaseService
    from src.ai.agent_core_loop import AgentCoreLoopExecutor
    from src.ai.knowledge_base import KnowledgeBase

    # 初始化
    db_service = DatabaseService()
    kb = KnowledgeBase()

    # 创建执行器
    executor = AgentCoreLoopExecutor(
        kb=kb,
        db_service=db_service,
        progress_callback=lambda msg: print(f"[Progress] {msg}")
    )

    # 使用逻辑链增强生成
    result = executor.execute_with_logic_chain(
        raw_content="研究背景：医学影像AI分析...",
        project_id=project_id
    )

    print(f"论文结果: {result.get('outline', {}).get('title')}")
    print(f"逻辑链数量: {result.get('logic_chains_count', 0)}")

    return result


# 示例 4: 可视化数据导出
def visualization_export_example():
    """
    导出可视化数据的示例
    """
    from src.services.logic_chain_integration import LogicChainIntegration

    # 创建集成服务
    integration = LogicChainIntegration()

    # 处理项目数据 (假设已有结果)
    result = integration.process_project_data(project_id="demo_project")

    if result.success:
        # 导出为Mermaid图表
        mermaid_chart = integration.export_logic_chain_visualization(result, format="mermaid")
        print("Mermaid 图表:")
        print(mermaid_chart)

        # 导出为GraphML (可用于Gephi等工具)
        graphml = integration.export_logic_chain_visualization(result, format="graphml")
        with open("logic_chain.graphml", "w", encoding="utf-8") as f:
            f.write(graphml)

        # 导出为JSON (可用于D3.js等可视化库)
        json_data = integration.export_logic_chain_visualization(result, format="json")
        with open("logic_chain.json", "w", encoding="utf-8") as f:
            f.write(json_data)


# 示例 5: 使用RAG增强
def rag_enhanced_example(project_id: str):
    """
    使用知识库RAG增强的示例
    """
    from src.ai.knowledge_base import KnowledgeBase
    from src.ai.logic_chain_agent import LogicChainAgent

    # 初始化知识库
    kb = KnowledgeBase()

    # 创建带RAG的Agent
    agent = LogicChainAgent(
        knowledge_base=kb,
        progress_callback=lambda msg: print(f"[Progress] {msg}")
    )

    # 处理数据时会自动使用知识库增强
    result = agent.process(
        ocr_data=[...],
        dicom_data=[...],
        text_data=[...],
        project_context="Context enhanced by RAG"
    )

    return result


if __name__ == "__main__":
    print("Logic Chain Agent 使用示例")
    print("=" * 50)

    # 运行基本示例
    result = basic_usage_example()

    print("\n" + "=" * 50)
    print("示例完成！")
