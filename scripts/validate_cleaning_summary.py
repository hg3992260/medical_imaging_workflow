import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.services.database_service import DatabaseService
from src.services.data_cleaning_service import DataCleaningService
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    db = DatabaseService()
    with db.get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT project_id FROM projects ORDER BY created_at DESC LIMIT 1")
        row = cur.fetchone()
        project_id = row[0] if row else "default-project-001"
    svc = DataCleaningService()
    summary = svc.get_project_sample_summary(project_id)
    logger.info(f"Summary for project {project_id}: {summary}")
    print(summary)

if __name__ == "__main__":
    main()
