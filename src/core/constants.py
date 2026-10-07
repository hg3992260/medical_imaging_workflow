"""
Centralized Constants for RSNA Medical Imaging Workflow
"""

# 放射学全域主字典 (基于 RSNA RadLex 标准)
# 用于防止 RAG 检索中的“领域漂移”并实现实体锚定
RADLEX_MASTER_DICTIONARY = {
    "Imaging_Modalities": {
        "CT": ["Computed Tomography", "Computed Tomographic", "CAT scan"],
        "MRI": ["Magnetic Resonance Imaging", "MR Imaging", "fMRI", "dMRI", "7T MRI"],
        "XR": ["X-ray", "Radiography", "Fluoroscopy"],
        "US": ["Ultrasound", "Ultrasonography", "Echocardiography"],
        "NM": ["Nuclear Medicine", "PET", "SPECT", "Scintigraphy"]
    },
    "Anatomic_Regions": {
        "Neuro": ["Brain", "Cerebellum", "Spinal Cord", "Ventricles", "White Matter", "Gray Matter"],
        "Thorax": ["Lung", "Mediastinum", "Heart", "Pleura", "Bronchus"],
        "Abdomen": ["Liver", "Spleen", "Kidney", "Adrenal", "Pancreas", "Aorta", "Gallbladder"],
        "MSK": ["Femur", "Tibia", "Erector Spinae", "Ligament", "Cartilage", "Bone Marrow"]
    },
    "Physical_Metrics": {
        "CT_Specific": ["HU", "Hounsfield Unit", "Attenuation", "Pitch", "CTDIvol", "DLP"],
        "MR_Specific": ["T1", "T2", "T1rho", "ADC", "FA", "TE", "TR", "Inversion Time", "Flip Angle"],
        "General": ["SNR", "CNR", "SD", "Standard Deviation", "Spatial Resolution", "Temporal Resolution"],
        "Diffusion_Dynamics": ["ADC", "Apparent Diffusion Coefficient", "IVIM", "Intravoxel Incoherent Motion", "DKI", "Diffusion Kurtosis Imaging"],
        "Physics_Anchors": ["Quantum Fluctuation", "Diffusion Gradient", "B-value", "Phase Encoding"]
    },
    "Reconstruction_Algorithms": {
        "DL_Based": ["DLIR", "ClearInfinity", "TrueFidelity", "AiCE", "Deep Learning Reconstruction"],
        "Iterative": ["ASiR-V", "iDose", "SAFIRE", "AIDR 3D", "Iterative Reconstruction"],
        "Spectral": ["VMI", "Virtual Monoenergetic", "Effective Atomic Number", "Material Decomposition"]
    },
    "Clinical_Findings": {
        "Morphology": ["Lesion", "Nodule", "Mass", "Cyst", "Hyperdense", "Hypointense"],
        "Enhancement": ["Arterial Phase", "Venous Phase", "Delayed Phase", "Washout", "Ktrans"]
    }
}
