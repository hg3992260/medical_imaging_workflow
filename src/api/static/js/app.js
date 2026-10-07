// Vanilla JS Implementation (Replacing Vue.js)

document.addEventListener('DOMContentLoaded', () => {
    // State
    const state = {
        currentView: 'project_management',
        projects: [],
        currentProject: null,
        newProject: {
            name: '',
            description: ''
        }
    };

    // Elements
    const elements = {
        sidebarLinks: document.querySelectorAll('.sidebar .nav-link'),
        projectListContainer: document.getElementById('project-list-container'),
        currentProjectName: document.getElementById('current-project-name'),
        currentProjectContainer: document.getElementById('current-project-container'),
        noProjectContainer: document.getElementById('no-project-container'),
        createProjectBtn: document.getElementById('btn-create-project'),
        createProjectForm: document.getElementById('form-create-project'),
        viewProjectManagement: document.getElementById('view-project-management'),
        viewPlaceholders: document.getElementById('view-placeholders'),
        placeholderText: document.getElementById('placeholder-text')
    };

    // Initialization
    init();

    function init() {
        setupNavigation();
        fetchProjects();
        fetchCurrentProject();
        setupModals();
    }

    // Navigation
    function setupNavigation() {
        elements.sidebarLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                // Update active state
                elements.sidebarLinks.forEach(l => l.classList.remove('active'));
                link.classList.add('active');

                // Update view state
                const viewName = link.getAttribute('data-view');
                state.currentView = viewName;
                renderView();
            });
        });
    }

    function renderView() {
        // Hide all views first
        elements.viewProjectManagement.style.display = 'none';
        document.getElementById('view-dicom-roi').style.display = 'none';
        elements.viewPlaceholders.style.display = 'none';

        if (state.currentView === 'project_management') {
            elements.viewProjectManagement.style.display = 'block';
        } else if (state.currentView === 'dicom_roi') {
            document.getElementById('view-dicom-roi').style.display = 'block';
            fetchDicomSessions(); // Fetch history when view is active
        } else {
            elements.viewPlaceholders.style.display = 'flex';
            elements.placeholderText.textContent = `当前视图: ${state.currentView}`;
        }
    }

    // DICOM Functions
    async function uploadDicom(e) {
        e.preventDefault();
        const fileInput = document.getElementById('input-dicom-file');
        const btn = document.getElementById('btn-upload-dicom');
        const spinner = btn.querySelector('.spinner-border');
        
        if (!fileInput.files.length) return alert('请选择文件');
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        if (state.currentProject) {
            formData.append('project_id', state.currentProject.project_id);
        }

        btn.disabled = true;
        spinner.classList.remove('d-none');

        try {
            const response = await fetch('/api/dicom/upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (data.success) {
                renderDicomResult(data);
                fetchDicomSessions(); // Refresh history
            } else {
                alert('处理失败: ' + data.message);
            }
        } catch (error) {
            console.error('Error uploading DICOM:', error);
            alert('上传失败');
        } finally {
            btn.disabled = false;
            spinner.classList.add('d-none');
        }
    }

    async function fetchDicomSessions() {
        if (!state.currentProject) return;
        try {
            const response = await fetch(`/api/dicom/sessions?project_id=${state.currentProject.project_id}`);
            const data = await response.json();
            if (data.success) {
                renderDicomHistory(data.data);
            }
        } catch (error) {
            console.error('Error fetching DICOM sessions:', error);
        }
    }

    function renderDicomHistory(sessions) {
        const container = document.getElementById('dicom-history-list');
        container.innerHTML = '';
        
        sessions.forEach(session => {
            const item = document.createElement('button');
            item.className = 'list-group-item list-group-item-action';
            item.innerHTML = `
                <div class="d-flex w-100 justify-content-between">
                    <h6 class="mb-1 text-truncate" style="max-width: 150px;">${session.file_path.split(/[\\/]/).pop()}</h6>
                    <small>${new Date(session.created_at).toLocaleDateString()}</small>
                </div>
                <small class="text-muted">ID: ${session.session_id.substring(0, 8)}...</small>
            `;
            item.onclick = () => loadDicomSession(session.session_id);
            container.appendChild(item);
        });
    }

    async function loadDicomSession(sessionId) {
        try {
            const response = await fetch(`/api/dicom/sessions/${sessionId}`);
            const data = await response.json();
            if (data.success) {
                renderDicomResult(data.data);
            }
        } catch (error) {
            console.error('Error loading session:', error);
        }
    }

    function renderDicomResult(data) {
        document.getElementById('dicom-placeholder').classList.add('d-none');
        document.getElementById('dicom-result-container').classList.remove('d-none');
        
        document.getElementById('roi-count-badge').textContent = `${data.roi_data ? data.roi_data.length : data.roi_count || 0} ROIs`;
        
        const info = data.dicom_info || {};
        document.getElementById('dicom-patient-id').textContent = info.patient_id || 'N/A';
        document.getElementById('dicom-study-date').textContent = info.study_date || 'N/A';
        
        const roiList = document.getElementById('roi-details-list');
        const rois = data.roi_data || data.rois || [];
        
        roiList.innerHTML = rois.map((roi, idx) => `
            <div class="border-bottom pb-2 mb-2">
                <strong>ROI #${idx + 1}</strong> (${roi.roi_type || 'Detected'})<br>
                <small>Area: ${roi.area.toFixed(2)} px²</small><br>
                <small>Intensity: ${roi.statistics ? roi.statistics.mean_intensity.toFixed(2) : 'N/A'}</small>
                ${roi.ocr_text ? `<div class="mt-1 p-1 bg-light border rounded"><small>OCR: ${roi.ocr_text}</small></div>` : ''}
            </div>
        `).join('') || '<p class="text-muted">No ROIs detected</p>';
    }

    // API Calls
    async function fetchProjects() {
        try {
            const response = await fetch('/api/projects');
            const data = await response.json();
            if (data.success) {
                state.projects = data.data;
                renderProjectList();
            }
        } catch (error) {
            console.error('Error fetching projects:', error);
            alert('获取项目列表失败');
        }
    }

    async function fetchCurrentProject() {
        const storedId = localStorage.getItem('currentProjectId');
        if (storedId) {
            try {
                const response = await fetch(`/api/projects/${storedId}`);
                const data = await response.json();
                if (data.success) {
                    state.currentProject = data.data;
                    renderCurrentProjectInfo();
                }
            } catch (error) {
                console.error('Error fetching current project:', error);
            }
        } else {
            renderCurrentProjectInfo();
        }
    }

    async function createProject(e) {
        e.preventDefault();
        const nameInput = document.getElementById('input-project-name');
        const descInput = document.getElementById('input-project-desc');
        
        const payload = {
            name: nameInput.value,
            description: descInput.value
        };

        try {
            const response = await fetch('/api/projects', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            
            if (data.success) {
                state.projects.unshift(data.data);
                
                // Clear form
                nameInput.value = '';
                descInput.value = '';
                
                // Close modal
                const modalEl = document.getElementById('createProjectModal');
                const modal = bootstrap.Modal.getInstance(modalEl);
                modal.hide();
                
                // Auto select if first
                if (!state.currentProject) {
                    selectProject(data.data);
                }
                
                renderProjectList();
            }
        } catch (error) {
            console.error('Error creating project:', error);
            alert('创建项目失败');
        }
    }

    async function deleteProject(project) {
        if (!confirm(`确定要删除项目 "${project.name}" 吗?`)) return;

        try {
            const response = await fetch(`/api/projects/${project.project_id}`, {
                method: 'DELETE'
            });
            const data = await response.json();

            if (data.success) {
                state.projects = state.projects.filter(p => p.project_id !== project.project_id);
                if (state.currentProject && state.currentProject.project_id === project.project_id) {
                    state.currentProject = null;
                    localStorage.removeItem('currentProjectId');
                    renderCurrentProjectInfo();
                }
                renderProjectList();
            }
        } catch (error) {
            console.error('Error deleting project:', error);
            alert('删除项目失败');
        }
    }

    function selectProject(project) {
        state.currentProject = project;
        localStorage.setItem('currentProjectId', project.project_id);
        renderCurrentProjectInfo();
        renderProjectList(); // Re-render to update border highlight
    }

    // Rendering
    function renderProjectList() {
        const container = elements.projectListContainer;
        container.innerHTML = '';

        state.projects.forEach(project => {
            const col = document.createElement('div');
            col.className = 'col-md-4 mb-4';
            
            const isSelected = state.currentProject && state.currentProject.project_id === project.project_id;
            const borderClass = isSelected ? 'border-primary' : '';

            col.innerHTML = `
                <div class="card h-100 ${borderClass}">
                    <div class="card-body">
                        <h5 class="card-title">${escapeHtml(project.name)}</h5>
                        <h6 class="card-subtitle mb-2 text-muted">ID: ${project.project_id}</h6>
                        <p class="card-text">${escapeHtml(project.description || '无描述')}</p>
                        <p class="card-text"><small class="text-muted">更新时间: ${new Date(project.updated_at).toLocaleString()}</small></p>
                    </div>
                    <div class="card-footer bg-transparent border-top-0">
                        <button class="btn btn-outline-primary btn-sm me-2 btn-select">选择</button>
                        <button class="btn btn-outline-danger btn-sm btn-delete">删除</button>
                    </div>
                </div>
            `;

            // Bind events
            col.querySelector('.btn-select').addEventListener('click', () => selectProject(project));
            col.querySelector('.btn-delete').addEventListener('click', () => deleteProject(project));

            container.appendChild(col);
        });
    }

    function renderCurrentProjectInfo() {
        if (state.currentProject) {
            elements.currentProjectContainer.style.display = 'block';
            elements.noProjectContainer.style.display = 'none';
            elements.currentProjectName.textContent = state.currentProject.name;
            elements.currentProjectName.title = state.currentProject.name;
        } else {
            elements.currentProjectContainer.style.display = 'none';
            elements.noProjectContainer.style.display = 'block';
        }
    }

    function setupModals() {
        if (elements.createProjectBtn) {
            elements.createProjectBtn.addEventListener('click', () => {
                const modalEl = document.getElementById('createProjectModal');
                const modal = new bootstrap.Modal(modalEl);
                modal.show();
            });
        }
        if (elements.createProjectForm) {
            elements.createProjectForm.addEventListener('submit', createProject);
        }
        
        // DICOM Listeners
        const dicomForm = document.getElementById('form-upload-dicom');
        if (dicomForm) {
            dicomForm.addEventListener('submit', uploadDicom);
        }
    }

    function escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});
