from flask import Blueprint, request, jsonify
import os
import json
import time
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest
from src.services.dicom_service import DicomService
from src.services.database_service import DatabaseService
from src.utils.logger import get_logger

dicom_bp = Blueprint('dicom_bp', __name__)
logger = get_logger(__name__)
db_service = DatabaseService()
dicom_service = DicomService(db_service)

ALLOWED_EXTENSIONS = {'dcm', 'dicom'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@dicom_bp.route('/dicom/upload', methods=['POST'])
def upload_dicom():
    """Upload and process DICOM file"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'message': 'Invalid file type'}), 400

        project_id = request.form.get('project_id', 'default')
        user_id = request.form.get('user_id', 'default_user')

        # Save temporarily
        filename = secure_filename(file.filename)
        temp_dir = os.path.join('temp', 'dicom_uploads')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)

        try:
            # Process DICOM
            # 1. Load and Extract Info
            dicom_data = dicom_service.load_dicom_file(temp_path)
            if not dicom_data:
                return jsonify({'success': False, 'message': 'Failed to load DICOM file'}), 400
                
            dicom_info = dicom_service.extract_dicom_info(dicom_data)
            
            # 2. Convert to Image
            image_array = dicom_service.convert_to_image(dicom_data)
            if image_array is None:
                return jsonify({'success': False, 'message': 'Failed to convert DICOM to image'}), 500

            # 3. Detect ROI
            # Default parameters
            roi_regions = dicom_service.detect_roi_regions(
                image_array,
                threshold=0.5,
                method='otsu',
                min_area=100,
                dicom_data=dicom_data
            )

            # 4. Create Session
            session_id = dicom_service.create_dicom_session(
                user_id=user_id,
                file_path=temp_path,
                dicom_info=dicom_info,
                roi_data=roi_regions,
                project_id=project_id
            )

            if not session_id:
                return jsonify({'success': False, 'message': 'Failed to create session'}), 500

            return jsonify({
                'success': True,
                'session_id': session_id,
                'dicom_info': dicom_info,
                'roi_count': len(roi_regions),
                'rois': roi_regions
            })

        finally:
            # Cleanup temp file if needed (create_dicom_session moves/copies it usually)
            # But here create_dicom_session uses storage service which likely copies.
            # We can remove temp if we are sure.
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

    except Exception as e:
        logger.error(f"Error processing DICOM: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@dicom_bp.route('/dicom/sessions', methods=['GET'])
def get_sessions():
    """Get DICOM sessions"""
    try:
        project_id = request.args.get('project_id', 'default')
        limit = int(request.args.get('limit', 50))
        sessions = dicom_service.get_dicom_sessions(project_id=project_id, limit=limit)
        return jsonify({'success': True, 'data': sessions})
    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@dicom_bp.route('/dicom/sessions/<session_id>', methods=['GET'])
def get_session_details(session_id):
    """Get session details including ROIs"""
    try:
        details = dicom_service.get_dicom_session_details(session_id)
        if details:
            return jsonify({'success': True, 'data': details})
        return jsonify({'success': False, 'message': 'Session not found'}), 404
    except Exception as e:
        logger.error(f"Error getting session details: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500
