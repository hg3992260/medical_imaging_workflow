import os

from flask import Blueprint, jsonify, request

from src.ai.agent_core_loop import AgentCoreLoopExecutor
from src.services.database_service import DatabaseService
from src.services.project_service import ProjectService
from src.services.project_storage_service import ProjectStorageService
from src.utils.logger import get_logger


agent_bp = Blueprint("agent_bp", __name__)
logger = get_logger(__name__)


@agent_bp.route("/agent/paper", methods=["POST"])
def generate_paper():
    try:
        data = request.json or {}
        project_id = str(data.get("project_id", "") or "").strip()
        raw_content = str(data.get("raw_content", "") or "")
        enable_global_docs = bool(data.get("enable_global_docs", False))
        global_docs_dir = str(data.get("global_docs_dir", "") or "").strip()
        judge = data.get("judge", None)
        workspace = str(data.get("workspace", "") or "").strip()
        enable_python_tool = bool(data.get("enable_python_tool", True))

        if not project_id:
            return jsonify({"success": False, "message": "project_id is required"}), 400

        db_service = DatabaseService()
        storage = ProjectStorageService(db_service)
        storage.create_project_storage(project_id)
        if not workspace:
            try:
                proj = ProjectService(db_service).get_project_by_id(project_id)
                meta = (getattr(proj, "metadata", None) or {}) if proj else {}
                batch_folder = str((meta or {}).get("batch_folder", "") or "").strip()
                if batch_folder and os.path.isdir(batch_folder):
                    workspace = batch_folder
                else:
                    workspace = str(storage.get_project_storage_path(project_id))
            except Exception:
                workspace = str(storage.get_project_storage_path(project_id))

        executor = AgentCoreLoopExecutor(
            kb=None,
            db_service=db_service,
            progress_cb=None,
            stream_cb=None,
            enable_global_docs=enable_global_docs,
            global_docs_dir=global_docs_dir or r"f:\RSNA\medical_imaging_workflow\src\docs",
        )
        result = executor.execute_paper_writing(
            raw_content=raw_content,
            project_id=project_id,
            judge_config=judge if isinstance(judge, dict) else None,
            workspace=workspace,
            enable_python_tool=enable_python_tool,
        )
        return jsonify({"success": True, "data": result})
    except Exception as e:
        logger.error(f"agent/paper failed: {e}")
        return jsonify({"success": False, "message": str(e)}), 500
