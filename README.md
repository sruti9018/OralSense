# OralSense
Multimodal Oral Cancer Risk Stratification Using Smartphone Images and Areca Nut–Inclusive Behavioral Metadata: A Cross-Population Explainable AI Study
🦷 OralSense — Multimodal Oral Cancer Risk Stratification

Multimodal Oral Cancer Risk Stratification Using Smartphone Images and Areca Nut–Inclusive Behavioral Metadata: A Cross-Population Explainable AI Study

Show Image
Show Image
Show Image
Show Image

📌 Overview
OralSense is a deep learning clinical decision support system for early oral cancer risk stratification using smartphone oral cavity images combined with behavioral metadata (areca nut, tobacco, alcohol, age, sex). The system uses two complementary AI models with Monte Carlo Dropout uncertainty estimation and provides explainable predictions through Grad-CAM heatmaps.
The project addresses a critical healthcare gap in South Asia, where oral cancer rates are among the highest in the world due to widespread areca nut and tobacco use.

🎯 Key Results
ModelValidation AccuracyImage-Only (MobileNetV2)90.3%Multimodal (Image + Metadata)93.1%
Risk classes:

✅ Normal — No signs of malignancy
⚠️ Variations — Minor tissue variations
🔶 OPMD — Oral Potentially Malignant Disorder
🔴 OC — Oral Cancer (immediate referral required)


🗂️ Project Structure
oral_cancer_ai/
├── app.py                    # Flask backend API
├── database.py               # SQLite patient database
├── index.html                # OralSense web app frontend
├── patient_history.html      # Patient history dashboard
├── data/
│   ├── augmented/            # Augmented training dataset
│   │   ├── Normal/           # 2,145 images
│   │   ├── OC/               # 500 images (augmented from 20)
│   │   ├── OPMD/             # 500 images (augmented from 125)
│   │   └── Variations/       # 500 images (augmented from 179)
│   ├── metadata_clean.csv    # Cleaned behavioral metadata
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── models/
│   ├── best_model.h5         # Image-only MobileNetV2 model
│   └── multimodal_model.h5   # Multimodal fusion model
├── notebooks/
│   ├── check_dataset.py
│   ├── explore_dataset.py
│   ├── prepare_dataset.py
│   ├── augment_dataset.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── gradcam.py
│   ├── explore_metadata.py
│   ├── prepare_multimodal.py
│   ├── train_multimodal.py
│   ├── evaluate_multimodal.py
│   ├── retrain_multimodal.py
│   └── generate_report.py
└── results/
    ├── sample_images.png
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── gradcam_results.png
    ├── multimodal_training.png
    ├── multimodal_confusion_matrix.png
    └── project_report.pdf

📦 Dataset
Primary — SMART-OM

2,469 images, 331 subjects, Tamil Nadu, India
4 classes: Normal, Variation, OPMD, Oral Cancer
Metadata: areca nut, tobacco type, alcohol, age, sex
DOI: 10.6084/m9.figshare.31341790

Supplementary — CODE (planned)

~500 images, 110 subjects, Ragas Dental College, Chennai
DOI: 10.6084/m9.figshare.30550889

Validation — Peradeniya (planned)

3,000 images, 714 subjects, Sri Lanka
Cross-population validation


🛠️ Tech Stack
CategoryTechnologyLanguagePython 3.10Deep LearningTensorFlow / KerasPretrained ModelMobileNetV2 (ImageNet)Loss FunctionFocal Loss (γ=2, α=0.25)UncertaintyMonte Carlo Dropout (20 runs)Image ProcessingOpenCV, PillowData HandlingPandas, NumPyVisualizationMatplotlib, SeabornExplainabilityGrad-CAMWeb BackendFlask + Flask-CORSDatabaseSQLiteFrontendHTML5, CSS3, JavaScript

🚀 Setup & Installation
1. Clone the repository
bashgit clone https://github.com/yourusername/oralsense-ai.git
cd oralsense-ai
2. Create conda environment
bashconda create -n oralcancer python=3.10
conda activate oralcancer
3. Install dependencies
bashpip install tensorflow opencv-python pandas numpy matplotlib seaborn scikit-learn flask flask-cors tqdm openpyxl
4. Download the dataset
Go to SMART-OM Figshare and download. Extract to data/ folder.
5. Run notebooks in order
bashpython notebooks/check_dataset.py
python notebooks/augment_dataset.py
python notebooks/train_model.py
python notebooks/train_multimodal.py
6. Start the web app
bashpython app.py
Open Chrome: http://127.0.0.1:5000

🌐 OralSense Web App Features

📷 Upload oral cavity smartphone photo
👤 Patient details — name, ID, age, sex, phone, address, doctor, department
⚠️ Habit history — smoking, chewing, areca nut, alcohol
🤖 Dual AI prediction — image model + multimodal model
📊 Monte Carlo Dropout — 20 forward passes for uncertainty estimation
🔥 Grad-CAM heatmap — shows which area AI focused on
📋 Patient report — auto-generated with clinical recommendation
🖨️ Print/PDF — save patient report
📊 Patient history — track risk progression across visits
🔴 Risk alerts — automatic alert if risk increased since last visit
💾 SQLite database — all scans saved automatically


🧠 Model Architecture
Image-Only Model
Input (224×224×3)
    → MobileNetV2 (pretrained ImageNet, last 30 layers unfrozen)
    → GlobalAveragePooling2D
    → Dense(256, ReLU) → Dropout(0.5)
    → Dense(128, ReLU) → Dropout(0.4)
    → Dense(4, Softmax)
Multimodal Fusion Model
Image Input (224×224×3)       Metadata Input (6 features)
    → MobileNetV2                 → Dense(32, ReLU)
    → GlobalAveragePooling2D      → BatchNormalization
    → Dense(256, ReLU)            → Dense(16, ReLU)
    → Dropout(0.5)                        |
              └────── Concatenate ────────┘
                          → Dense(128, ReLU)
                          → Dropout(0.4)
                          → Dense(4, Softmax)
Metadata Features
FeatureTypeDescriptionAgeContinuous (0–1)Patient age normalizedSexBinary0=Female, 1=MaleSmokingBinarySmoking habitChewingBinaryChewing tobaccoAreca NutBinaryAreca nut usageAlcoholBinaryAlcohol consumption

⚕️ Clinical Risk Levels & Recommendations
RiskClassRecommended Action✅ LowNormalRoutine follow-up in 12 months⚠️ Low-MediumVariationsFollow-up in 3–6 months🔶 MediumOPMDRefer to specialist within 2–4 weeks🔴 HighOCImmediate oncologist referral

📊 Training Details

Optimizer: Adam (lr=1e-4)
Loss: Focal Loss (γ=2.0, α=0.25) — handles class imbalance
Class Weights: Balanced — OC gets 1.82× weight vs Normal 0.42×
Augmentation: Flip, rotation, brightness, zoom, shear
Early Stopping: Patience=10 on val_accuracy
Uncertainty: Monte Carlo Dropout with 20 inference passes
Decision Rule: Always use higher risk result between both models


⚠️ Disclaimer
This system is for research and screening purposes only. It is not a substitute for clinical diagnosis. All predictions must be reviewed by a qualified medical professional.

👩‍💻 Author
S.Sruti
M P Kavi Nisha
Oral Cancer Risk Stratification using Multimodal AI
Tamil Nadu, India · May 2026

📚 References

Devindi et al. — Multimodal Deep CNN Pipeline for AI-Assisted Early Detection of Oral Cancer — IEEE Access, 2024
Frontiers in Oral Health — AI and the Diagnosis of Oral Cavity Cancer from Clinical Photographs — 2025
Ou et al. — Deep Learning Based Multimodal Fusion Model for Skin Lesion Diagnosis — Frontiers in Surgery, 2022
Sharma et al. — Exploring Data Modalities and Advances in AI for Oral Cancer Detection — IET Image Processing, 2025
Pivarathne et al. — A Comprehensive Dataset of Annotated Oral Cavity Images — Oral Oncology, 2024
