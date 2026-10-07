from flask import Blueprint, jsonify, request
from src.services.project_service import ProjectService
from src.models.project import Project
from src.utils.logger import get_logger

project_bp = Blueprint('project_bp', __name__)
logger = get_logger(__name__)
project_service = ProjectService()

@project_bp.route('/projects', methods=['GET'])
def get_projects():
    """Get all projects"""
    try:
        status = request.args.get('status')
        projects = project_service.get_all_projects(status)
        return jsonify({
            'success': True,
            'data': [p.to_dict() for p in projects]
        })
    except Exception as e:
        logger.error(f"Error getting projects: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@project_bp.route('/projects', methods=['POST'])
def create_project():
    """Create a new project"""
    try:
        data = request.json
        name = data.get('name')
        description = data.get('description', '')
        
        if not name:
            return jsonify({'success': False, 'message': 'Project name is required'}), 400
            
        new_project = Project.create_new(name, description)
        if project_service.create_project(new_project):
            return jsonify({
                'success': True,
                'data': new_project.to_dict()
            }), 201
        else:
            return jsonify({'success': False, 'message': 'Failed to create project'}), 500
            
    except Exception as e:
        logger.error(f"Error creating project: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@project_bp.route('/projects/<project_id>', methods=['GET'])
def get_project(project_id):
    """Get project details"""
    try:
        project = project_service.get_project_by_id(project_id)
        if project:
            return jsonify({
                'success': True,
                'data': project.to_dict()
            })
        else:
            return jsonify({'success': False, 'message': 'Project not found'}), 404
    except Exception as e:
        logger.error(f"Error getting project {project_id}: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@project_bp.route('/projects/<project_id>', methods=['PUT'])
def update_project(project_id):
    """Update project"""
    try:
        data = request.json
        project = project_service.get_project_by_id(project_id)
        
        if not project:
            return jsonify({'success': False, 'message': 'Project not found'}), 404
            
        project.update_info(
            name=data.get('name'),
            description=data.get('description'),
            metadata=data.get('metadata')
        )
        
        if project_service.update_project(project):
            return jsonify({
                'success': True,
                'data': project.to_dict()
            })
        else:
            return jsonify({'success': False, 'message': 'Failed to update project'}), 500
            
    except Exception as e:
        logger.error(f"Error updating project {project_id}: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@project_bp.route('/projects/<project_id>', methods=['DELETE'])
def delete_project(project_id):
    """Delete (archive) project"""
    try:
        if project_service.delete_project(project_id):
            return jsonify({'success': True, 'message': 'Project deleted successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to delete project'}), 500
    except Exception as e:
        logger.error(f"Error deleting project {project_id}: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@project_bp.route('/projects/<project_id>/archive', methods=['POST'])
def archive_project(project_id):
    """Archive project"""
    try:
        if project_service.archive_project(project_id):
            return jsonify({'success': True, 'message': 'Project archived successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to archive project'}), 500
    except Exception as e:
        logger.error(f"Error archiving project {project_id}: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@project_bp.route('/projects/<project_id>/activate', methods=['POST'])
def activate_project(project_id):
    """Activate project"""
    try:
        if project_service.activate_project(project_id):
            return jsonify({'success': True, 'message': 'Project activated successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to activate project'}), 500
    except Exception as e:
        logger.error(f"Error activating project {project_id}: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@project_bp.route('/projects/<project_id>/stats', methods=['GET'])
def get_project_stats(project_id):
    """Get project statistics"""
    try:
        stats = project_service.get_project_stats(project_id)
        return jsonify({
            'success': True,
            'data': stats
        })
    except Exception as e:
        logger.error(f"Error getting project stats {project_id}: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500
