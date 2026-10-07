import os
import shutil
import sqlite3
import json
import zipfile
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

class BackupService:
    """备份恢复服务
    
    负责项目数据的备份和恢复功能，包括：
    - 项目数据的完整备份
    - 增量备份
    - 数据恢复
    - 备份文件管理
    """
    
    def __init__(self, database_service, storage_service):
        self.db_service = database_service
        self.storage_service = storage_service
        self.backup_path = Path("storage/backups")
        self.backup_path.mkdir(parents=True, exist_ok=True)
    
    def create_full_backup(self, project_id: str, backup_name: str = None) -> Optional[str]:
        """创建项目完整备份
        
        Args:
            project_id: 项目ID
            backup_name: 备份名称，为空则自动生成
            
        Returns:
            str: 备份ID，失败返回None
        """
        try:
            if not backup_name:
                backup_name = f"full_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            backup_id = self._generate_backup_id()
            backup_dir = self.backup_path / backup_id
            backup_dir.mkdir(exist_ok=True)
            
            # 备份项目文件
            project_storage_path = self.storage_service.get_project_storage_path(project_id)
            if project_storage_path.exists():
                shutil.copytree(
                    str(project_storage_path),
                    str(backup_dir / "files"),
                    dirs_exist_ok=True
                )
            
            # 备份数据库数据
            self._backup_project_database_full(project_id, backup_dir / "database.json")
            
            # 创建备份元数据
            metadata = {
                'backup_id': backup_id,
                'project_id': project_id,
                'backup_name': backup_name,
                'backup_type': 'full',
                'created_at': datetime.now().isoformat(),
                'file_count': self._count_files(backup_dir / "files"),
                'total_size': self._calculate_directory_size(backup_dir)
            }
            
            with open(backup_dir / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # 压缩备份
            backup_file = self.backup_path / f"{backup_id}.zip"
            self._create_zip_archive(backup_dir, backup_file)
            
            # 删除临时目录
            shutil.rmtree(backup_dir)
            
            # 记录备份信息到数据库
            self._record_backup(project_id, backup_id, backup_name, 'full', 
                              str(backup_file), metadata['total_size'])
            
            return backup_id
            
        except Exception as e:
            print(f"创建完整备份失败: {e}")
            return None
    
    def create_incremental_backup(self, project_id: str, last_backup_id: str = None) -> Optional[str]:
        """创建增量备份
        
        Args:
            project_id: 项目ID
            last_backup_id: 上次备份ID，为空则查找最近的备份
            
        Returns:
            str: 备份ID，失败返回None
        """
        try:
            if not last_backup_id:
                last_backup_id = self._get_latest_backup_id(project_id)
                if not last_backup_id:
                    # 没有历史备份，创建完整备份
                    return self.create_full_backup(project_id)
            
            backup_name = f"incremental_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_id = self._generate_backup_id()
            backup_dir = self.backup_path / backup_id
            backup_dir.mkdir(exist_ok=True)
            
            # 获取上次备份时间
            last_backup_time = self._get_backup_time(last_backup_id)
            if not last_backup_time:
                return None
            
            # 备份变更的文件
            changed_files = self._get_changed_files(project_id, last_backup_time)
            if changed_files:
                files_dir = backup_dir / "files"
                files_dir.mkdir(exist_ok=True)
                
                for file_info in changed_files:
                    src_path = Path(file_info['file_path'])
                    if src_path.exists():
                        rel_path = src_path.relative_to(
                            self.storage_service.get_project_storage_path(project_id)
                        )
                        dst_path = files_dir / rel_path
                        dst_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(src_path), str(dst_path))
            
            # 备份变更的数据库数据
            self._backup_project_database_incremental(
                project_id, backup_dir / "database.json", last_backup_time
            )
            
            # 创建备份元数据
            metadata = {
                'backup_id': backup_id,
                'project_id': project_id,
                'backup_name': backup_name,
                'backup_type': 'incremental',
                'base_backup_id': last_backup_id,
                'created_at': datetime.now().isoformat(),
                'file_count': len(changed_files) if changed_files else 0,
                'total_size': self._calculate_directory_size(backup_dir)
            }
            
            with open(backup_dir / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # 压缩备份
            backup_file = self.backup_path / f"{backup_id}.zip"
            self._create_zip_archive(backup_dir, backup_file)
            
            # 删除临时目录
            shutil.rmtree(backup_dir)
            
            # 记录备份信息到数据库
            self._record_backup(project_id, backup_id, backup_name, 'incremental',
                              str(backup_file), metadata['total_size'])
            
            return backup_id
            
        except Exception as e:
            print(f"创建增量备份失败: {e}")
            return None
    
    def restore_backup(self, project_id: str, backup_id: str, 
                      restore_files: bool = True, restore_database: bool = True) -> bool:
        """恢复备份
        
        Args:
            project_id: 项目ID
            backup_id: 备份ID
            restore_files: 是否恢复文件
            restore_database: 是否恢复数据库
            
        Returns:
            bool: 恢复是否成功
        """
        try:
            backup_file = self.backup_path / f"{backup_id}.zip"
            if not backup_file.exists():
                print(f"备份文件不存在: {backup_file}")
                return False
            
            # 解压备份文件
            restore_dir = self.backup_path / f"restore_{backup_id}"
            restore_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(backup_file, 'r') as zip_ref:
                zip_ref.extractall(restore_dir)
            
            # 读取备份元数据
            metadata_file = restore_dir / "metadata.json"
            if not metadata_file.exists():
                print("备份元数据文件不存在")
                return False
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 恢复文件
            if restore_files:
                files_dir = restore_dir / "files"
                if files_dir.exists():
                    project_storage_path = self.storage_service.get_project_storage_path(project_id)
                    
                    # 备份当前数据（以防恢复失败）
                    current_backup_dir = self.backup_path / f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    if project_storage_path.exists():
                        shutil.copytree(str(project_storage_path), str(current_backup_dir))
                    
                    try:
                        # 清空当前存储目录
                        if project_storage_path.exists():
                            shutil.rmtree(project_storage_path)
                        
                        # 恢复文件
                        shutil.copytree(str(files_dir), str(project_storage_path))
                        
                        # 删除预备份
                        if current_backup_dir.exists():
                            shutil.rmtree(current_backup_dir)
                            
                    except Exception as e:
                        # 恢复失败，回滚
                        if current_backup_dir.exists():
                            if project_storage_path.exists():
                                shutil.rmtree(project_storage_path)
                            shutil.copytree(str(current_backup_dir), str(project_storage_path))
                            shutil.rmtree(current_backup_dir)
                        raise e
            
            # 恢复数据库
            if restore_database:
                database_file = restore_dir / "database.json"
                if database_file.exists():
                    self._restore_project_database(project_id, database_file)
            
            # 清理临时目录
            shutil.rmtree(restore_dir)
            
            # 记录恢复操作
            self._record_restore_operation(project_id, backup_id)
            
            return True
            
        except Exception as e:
            print(f"恢复备份失败: {e}")
            return False
    
    def list_backups(self, project_id: str) -> List[Dict]:
        """列出项目备份
        
        Args:
            project_id: 项目ID
            
        Returns:
            List[Dict]: 备份列表
        """
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
            
            cursor.execute("""
                SELECT backup_id, backup_name, backup_type, backup_path, 
                       backup_size, created_at
                FROM backup_records
                WHERE project_id = ?
                ORDER BY created_at DESC
            """, (project_id,))
            
            results = cursor.fetchall()
            backups = []
            
            for row in results:
                backups.append({
                    'backup_id': row[0],
                    'backup_name': row[1],
                    'backup_type': row[2],
                    'backup_path': row[3],
                    'backup_size': row[4],
                    'created_at': row[5],
                    'size_mb': round(row[4] / (1024 * 1024), 2) if row[4] else 0
                })
            
            return backups
            
        except Exception as e:
            print(f"列出备份失败: {e}")
            return []
    
    def delete_backup(self, backup_id: str) -> bool:
        """删除备份
        
        Args:
            backup_id: 备份ID
            
        Returns:
            bool: 删除是否成功
        """
        try:
            # 删除备份文件
            backup_file = self.backup_path / f"{backup_id}.zip"
            if backup_file.exists():
                backup_file.unlink()
            
            # 从数据库删除记录
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM backup_records WHERE backup_id = ?
                """, (backup_id,))
                
                conn.commit()
            return True
            
        except Exception as e:
            print(f"删除备份失败: {e}")
            return False
    
    def create_project_backup(self, project_id: str, backup_name: str = None) -> Optional[Dict]:
        """创建项目备份（别名方法）
        
        Args:
            project_id: 项目ID
            backup_name: 备份名称
            
        Returns:
            Optional[Dict]: 备份结果字典，包含success、backup_id、backup_path等信息
        """
        backup_id = self.create_full_backup(project_id, backup_name)
        if backup_id:
            backup_file = self.backup_path / f"{backup_id}.zip"
            return {
                'success': True,
                'backup_id': backup_id,
                'backup_path': str(backup_file)
            }
        else:
            return {
                'success': False,
                'backup_id': None,
                'backup_path': None
             }
    
    def get_project_backups(self, project_id: str) -> List[Dict]:
        """获取项目备份列表（别名方法）
        
        Args:
            project_id: 项目ID
            
        Returns:
            List[Dict]: 备份列表
        """
        return self.list_backups(project_id)
    
    def _generate_backup_id(self) -> str:
        """生成备份ID"""
        return f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
    
    def _backup_project_database_full(self, project_id: str, output_file: Path):
        """备份项目数据库数据（完整备份）"""
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
            
                # 备份相关表的数据
                tables_to_backup = [
                    'dicom_sessions', 'roi_data', 'ocr_sessions', 'ocr_results',
                    'text_sessions', 'documents', 'storage_files'
                ]
                
                backup_data = {}
                
                for table in tables_to_backup:
                    # 根据表结构使用不同的查询策略
                    if table == 'storage_files':
                        cursor.execute(f"""
                            SELECT sf.* FROM {table} sf
                            JOIN project_storage ps ON sf.storage_id = ps.storage_id
                            WHERE ps.project_id = ?
                        """, (project_id,))
                    elif table == 'documents':
                        cursor.execute(f"""
                            SELECT d.* FROM {table} d
                            JOIN text_sessions ts ON d.session_id = ts.session_id
                            WHERE ts.project_id = ?
                        """, (project_id,))
                    elif table == 'roi_data':
                        cursor.execute(f"""
                            SELECT r.* FROM {table} r
                            JOIN dicom_sessions ds ON r.session_id = ds.session_id
                            WHERE ds.project_id = ?
                        """, (project_id,))
                    elif table == 'ocr_results':
                        cursor.execute(f"""
                            SELECT o.* FROM {table} o
                            JOIN ocr_sessions os ON o.session_id = os.session_id
                            WHERE os.project_id = ?
                        """, (project_id,))
                    else:
                        # 对于有project_id列的表
                        cursor.execute(f"SELECT * FROM {table} WHERE project_id = ?", (project_id,))
                    
                    columns = [description[0] for description in cursor.description]
                    rows = cursor.fetchall()
                    
                    backup_data[table] = {
                        'columns': columns,
                        'rows': rows
                    }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(backup_data, f, indent=2, ensure_ascii=False, default=str)

                
        except Exception as e:
            print(f"备份数据库数据失败: {e}")
    
    def _backup_project_database_incremental(self, project_id: str, output_file: Path, since_time: str):
        """备份项目数据库数据（增量备份）"""
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
            
                tables_to_backup = [
                    'dicom_sessions', 'roi_data', 'ocr_sessions', 'ocr_results',
                    'text_sessions', 'documents', 'storage_files'
                ]
                
                backup_data = {}
                
                for table in tables_to_backup:
                    # 根据表结构使用不同的查询策略进行增量备份
                    if table == 'storage_files':
                        cursor.execute(f"""
                            SELECT sf.* FROM {table} sf
                            JOIN project_storage ps ON sf.storage_id = ps.storage_id
                            WHERE ps.project_id = ? AND sf.created_at > ?
                        """, (project_id, since_time))
                    elif table == 'documents':
                        cursor.execute(f"""
                            SELECT d.* FROM {table} d
                            JOIN text_sessions ts ON d.session_id = ts.session_id
                            WHERE ts.project_id = ? AND (d.created_at > ? OR d.modified_at > ?)
                        """, (project_id, since_time, since_time))
                    elif table == 'roi_data':
                        cursor.execute(f"""
                            SELECT r.* FROM {table} r
                            JOIN dicom_sessions ds ON r.session_id = ds.session_id
                            WHERE ds.project_id = ? AND r.created_at > ?
                        """, (project_id, since_time))
                    elif table == 'ocr_results':
                        cursor.execute(f"""
                            SELECT o.* FROM {table} o
                            JOIN ocr_sessions os ON o.session_id = os.session_id
                            WHERE os.project_id = ? AND o.created_at > ?
                        """, (project_id, since_time))
                    else:
                        # 对于有project_id列的表
                        cursor.execute(f"""
                            SELECT * FROM {table} 
                            WHERE project_id = ? AND 
                                  (created_at > ? OR updated_at > ?)
                        """, (project_id, since_time, since_time))
                    
                    columns = [description[0] for description in cursor.description]
                    rows = cursor.fetchall()
                    
                    if rows:
                        backup_data[table] = {
                            'columns': columns,
                            'rows': rows
                        }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(backup_data, f, indent=2, ensure_ascii=False, default=str)

                
        except Exception as e:
            print(f"增量备份数据库数据失败: {e}")
    
    def _restore_project_database(self, project_id: str, database_file: Path):
        """恢复项目数据库数据"""
        try:
            with open(database_file, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
            
                # 清空现有数据
                tables_to_clear = [
                    'storage_files', 'documents', 'text_sessions', 'ocr_results',
                    'ocr_sessions', 'roi_data', 'dicom_sessions'
                ]
                
                for table in tables_to_clear:
                    # 根据表结构使用不同的删除策略
                    if table == 'storage_files':
                        cursor.execute(f"""
                            DELETE FROM {table} WHERE storage_id IN (
                                SELECT storage_id FROM project_storage WHERE project_id = ?
                            )
                        """, (project_id,))
                    elif table == 'documents':
                        cursor.execute(f"""
                            DELETE FROM {table} WHERE session_id IN (
                                SELECT session_id FROM text_sessions WHERE project_id = ?
                            )
                        """, (project_id,))
                    elif table == 'roi_data':
                        cursor.execute(f"""
                            DELETE FROM {table} WHERE session_id IN (
                                SELECT session_id FROM dicom_sessions WHERE project_id = ?
                            )
                        """, (project_id,))
                    elif table == 'ocr_results':
                        cursor.execute(f"""
                            DELETE FROM {table} WHERE session_id IN (
                                SELECT session_id FROM ocr_sessions WHERE project_id = ?
                            )
                        """, (project_id,))
                    else:
                        # 对于有project_id列的表
                        cursor.execute(f"DELETE FROM {table} WHERE project_id = ?", (project_id,))
                
                # 恢复数据
                for table, data in backup_data.items():
                    columns = data['columns']
                    rows = data['rows']
                    
                    if rows:
                        placeholders = ','.join(['?' for _ in columns])
                        cursor.executemany(
                            f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
                            rows
                        )
                
                conn.commit()

            
        except Exception as e:
            print(f"恢复数据库数据失败: {e}")
    
    def _get_changed_files(self, project_id: str, since_time: str) -> List[Dict]:
        """获取变更的文件"""
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
            
            cursor.execute("""
                SELECT sf.file_id, sf.file_path, sf.file_type, sf.created_at
                FROM storage_files sf
                JOIN project_storage ps ON sf.storage_id = ps.storage_id
                WHERE ps.project_id = ? AND sf.created_at > ?
                ORDER BY sf.created_at
            """, (project_id, since_time))
            
            results = cursor.fetchall()
            return [{
                'file_id': row[0],
                'file_path': row[1],
                'file_type': row[2],
                'created_at': row[3]
            } for row in results]
            
        except Exception as e:
            print(f"获取变更文件失败: {e}")
            return []
    
    def _get_latest_backup_id(self, project_id: str) -> Optional[str]:
        """获取最新备份ID"""
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
            
            cursor.execute("""
                SELECT backup_id FROM backup_records
                WHERE project_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (project_id,))
            
            result = cursor.fetchone()
            return result[0] if result else None
            
        except Exception as e:
            print(f"获取最新备份ID失败: {e}")
            return None
    
    def _get_backup_time(self, backup_id: str) -> Optional[str]:
        """获取备份时间"""
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
            
            cursor.execute("""
                SELECT created_at FROM backup_records
                WHERE backup_id = ?
            """, (backup_id,))
            
            result = cursor.fetchone()
            return result[0] if result else None
            
        except Exception as e:
            print(f"获取备份时间失败: {e}")
            return None
    
    def _record_backup(self, project_id: str, backup_id: str, backup_name: str,
                      backup_type: str, backup_path: str, backup_size: int):
        """记录备份信息"""
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO backup_records 
                    (project_id, backup_id, backup_name, backup_type, backup_path, 
                     backup_size, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    project_id, backup_id, backup_name, backup_type, backup_path,
                    backup_size, datetime.now().isoformat()
                ))
                
                conn.commit()
                
        except Exception as e:
            print(f"记录备份信息失败: {e}")
    
    def _record_restore_operation(self, project_id: str, backup_id: str):
        """记录恢复操作"""
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE backup_records 
                SET last_restored_at = ?
                WHERE backup_id = ?
            """, (datetime.now().isoformat(), backup_id))
            
            conn.commit()
            
        except Exception as e:
            print(f"记录恢复操作失败: {e}")
    
    def _count_files(self, directory: Path) -> int:
        """统计目录中的文件数量"""
        if not directory.exists():
            return 0
        
        count = 0
        for item in directory.rglob('*'):
            if item.is_file():
                count += 1
        return count
    
    def _calculate_directory_size(self, directory: Path) -> int:
        """计算目录大小"""
        if not directory.exists():
            return 0
        
        total_size = 0
        for item in directory.rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size
        return total_size
    
    def _create_zip_archive(self, source_dir: Path, output_file: Path):
        """创建ZIP压缩包"""
        with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in source_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_dir)
                    zipf.write(file_path, arcname)