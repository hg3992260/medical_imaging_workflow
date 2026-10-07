import sqlite3
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from src.services.database_service import DatabaseService

logger = logging.getLogger(__name__)

@dataclass
class PatientCase:
    """Represents a complete view of a patient's data within a project."""
    user_id: str
    display_name: str
    project_id: str
    ocr_data: List[Dict[str, Any]]
    dicom_data: List[Dict[str, Any]]
    text_data: List[Dict[str, Any]]
    
    def to_dict(self):
        return asdict(self)

    def get_summary_text(self) -> str:
        """Generates a text summary of the case for LLM consumption."""
        summary = f"Patient ID: {self.display_name} ({self.user_id})\n"
        
        if self.ocr_data:
            summary += "\n[OCR Extracted Data]:\n"
            for item in self.ocr_data:
                summary += f"- Source: {item.get('image_name', 'Unknown')}\n"
                
                # Check if extracted_text is JSON (Sheet structure)
                import json
                text_content = item.get('text', '')
                try:
                    if text_content.strip().startswith('{') or text_content.strip().startswith('['):
                        json_data = json.loads(text_content)
                        # 尝试将 JSON 数据转为 Markdown 表格视图
                        if isinstance(json_data, list) and len(json_data) > 0 and isinstance(json_data[0], dict):
                            import pandas as pd
                            df = pd.DataFrame(json_data)
                            md_table = df.to_markdown(index=False)
                            summary += f"  Structured Sheet Data (DataFrame View):\n{md_table}\n"
                            summary += "  (Column Explanations: e.g., CT_Liver = 肝脏平均CT值, SD = 图像噪声)\n"
                        else:
                            summary += f"  Structured Sheet Data: {json.dumps(json_data, ensure_ascii=False)[:1000]}...\n"
                    else:
                        summary += f"  Text: {text_content[:500]}...\n"
                except Exception as e:
                    summary += f"  Text: {text_content[:500]}...\n"

                if item.get('confidence'):
                    summary += f"  Confidence: {item.get('confidence'):.2f}\n"

        if self.dicom_data:
            summary += "\n[Imaging Features (DICOM)]:\n"
            for item in self.dicom_data:
                summary += f"- Scan: {item.get('file_name', 'Unknown')}\n"
                if item.get('roi_data'):
                    for roi in item['roi_data']:
                        summary += f"  ROI ({roi.get('roi_type')}): Area={roi.get('area', 'N/A')}, Name={roi.get('roi_name', 'N/A')}\n"

        if self.text_data:
            summary += "\n[Clinical Notes]:\n"
            for item in self.text_data:
                summary += f"- Doc: {item.get('file_name', 'Unknown')}\n"
                summary += f"  Content: {item.get('content', '')[:500]}...\n"
        
        return summary

class DataAggregationService:
    """
    Service to aggregate all data points (OCR, DICOM, Text) for patients within a project.
    Acts as the 'Data Mining' layer before the AI Agent loop.
    """

    def __init__(self, db_service: DatabaseService):
        self.db = db_service

    def aggregate_project_data(self, project_id: str) -> List[PatientCase]:
        """
        Aggregates all available data for a given project, grouped by user_id.
        """
        logger.info(f"Starting data aggregation for project: {project_id}")
        
        patients = []
        
        try:
            # 1. Get all users (patients/samples) in the project
            users = self._get_project_users(project_id)
            logger.info(f"Found {len(users)} patients/samples in project.")

            for user in users:
                user_id = user['user_id']
                display_name = user['display_name']
                
                # 2. Fetch OCR Data
                ocr_data = self._get_ocr_data(project_id, user_id)
                
                # 3. Fetch DICOM Data (including ROIs)
                dicom_data = self._get_dicom_data(project_id, user_id)
                
                # 4. Fetch Text Data
                text_data = self._get_text_data(project_id, user_id)
                
                # Only create case if there is ANY data
                if ocr_data or dicom_data or text_data:
                    case = PatientCase(
                        user_id=user_id,
                        display_name=display_name,
                        project_id=project_id,
                        ocr_data=ocr_data,
                        dicom_data=dicom_data,
                        text_data=text_data
                    )
                    patients.append(case)
            
            # Domain Consistency Guard
            # We skip this if there is no data
            if patients:
                self._check_domain_consistency(project_id, patients)

            logger.info(f"Aggregation complete. Created {len(patients)} patient cases.")
            return patients

        except Exception as e:
            logger.error(f"Failed to aggregate project data: {e}", exc_info=True)
            return []

    def _check_domain_consistency(self, project_id: str, patients: List[PatientCase]):
        """
        Topic-Evidence Alignment: Check if the evidence data roughly matches the project topic.
        If we find high frequency of medical terms (CT, Hounsfield, ROI) but the topic is something else (like Water Quality), we warn.
        """
        # Fetch project info to get the topic/title
        # Note: In projects table, the columns are project_id, name, description. We must query 'name' not 'project_name'.
        project_info = self.db.execute_query("SELECT name, description FROM projects WHERE project_id = ?", (project_id,))
        if not project_info:
            return
            
        project_name = str(project_info[0].get('name', '')).lower()
        project_desc = str(project_info[0].get('description', '')).lower()
        topic_text = f"{project_name} {project_desc}"
        
        # If the project title itself contains medical keywords, it's fine.
        medical_keywords = ['ct', 'mri', 'hounsfield', 'hu', 'dose', 'radiologist', 'scan', 'dicom', 'roi', 'lesion', 'tumor']
        topic_is_medical = any(kw in topic_text for kw in medical_keywords)
        
        # Check evidence
        evidence_text = ""
        for p in patients[:5]: # Check first 5 patients
            evidence_text += p.get_summary_text().lower()
            
        evidence_is_medical = sum(evidence_text.count(kw) for kw in medical_keywords) > 5
        
        if evidence_is_medical and not topic_is_medical:
            logger.warning("DOMAIN DRIFT DETECTED: The project topic does not seem medical, but the evidence contains high-frequency medical terms (CT, HU, etc.).")
            # In a real system, we might raise an Exception or return a warning flag to the UI.
            # For now, we log a critical warning that will be visible in the console.

    def _get_project_users(self, project_id: str) -> List[Dict]:
        query = """
            SELECT user_id, display_name 
            FROM user_ids 
            WHERE project_id = ? AND is_active = 1
        """
        return self.db.execute_query(query, (project_id,))

    def _get_ocr_data(self, project_id: str, user_id: str) -> List[Dict]:
        """Fetches OCR results joined with session info."""
        # Fix: Use image_path instead of image_name as per schema
        query = """
            SELECT 
                r.result_id,
                s.image_path as image_name, 
                COALESCE(r.recognized_text, r.extracted_text) as text, 
                r.confidence,
                r.extracted_text,
                r.created_at
            FROM ocr_sessions s
            LEFT JOIN ocr_results r ON s.session_id = r.session_id
            WHERE s.project_id = ? AND s.user_id = ?
            AND (r.recognized_text IS NOT NULL OR r.extracted_text IS NOT NULL)
        """
        rows = self.db.execute_query(query, (project_id, user_id)) or []
        for r in rows:
            rid = str((r or {}).get("result_id") or "")
            if rid:
                merged = self._load_chunked_text("ocr_result", rid, str((r or {}).get("text") or ""))
                r["text"] = merged
        return rows


    def _get_dicom_data(self, project_id: str, user_id: str) -> List[Dict]:
        """Fetches DICOM sessions and their associated ROIs."""
        # First get sessions
        session_query = """
            SELECT session_id, file_name, dicom_info
            FROM dicom_sessions
            WHERE project_id = ? AND user_id = ?
        """
        sessions = self.db.execute_query(session_query, (project_id, user_id))
        
        for session in sessions:
            # Get ROIs for each session
            roi_query = """
                SELECT roi_type, roi_name, area, perimeter, properties
                FROM roi_data
                WHERE session_id = ?
            """
            session['roi_data'] = self.db.execute_query(roi_query, (session['session_id'],))
            
        return sessions

    def _get_text_data(self, project_id: str, user_id: str) -> List[Dict]:
        """Fetches text documents."""
        query = """
            SELECT d.doc_id, d.file_name, d.content, d.analysis_result
            FROM text_sessions s
            JOIN documents d ON s.session_id = d.session_id
            WHERE s.project_id = ? AND s.user_id = ?
        """
        rows = self.db.execute_query(query, (project_id, user_id)) or []
        for r in rows:
            did = str((r or {}).get("doc_id") or "")
            if did:
                merged = self._load_chunked_text("document", did, str((r or {}).get("content") or ""))
                r["content"] = merged
        return rows

    def _load_chunked_text(self, source_type: str, source_id: str, fallback_text: str = "") -> str:
        st = str(source_type or "").strip()
        sid = str(source_id or "").strip()
        if not st or not sid:
            return str(fallback_text or "")
        try:
            rows = self.db.execute_query(
                """
                SELECT text_content
                FROM content_chunks
                WHERE source_type = ? AND source_id = ?
                ORDER BY chunk_index ASC
                """,
                (st, sid),
            )
            if rows:
                merged = "".join([str((x or {}).get("text_content") or "") for x in rows])
                if merged:
                    return merged
        except Exception:
            pass
        return str(fallback_text or "")
