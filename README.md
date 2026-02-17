# OralVisionDx - AI Powered Oral Disease Detection & Localization

> **One photo in. Diagnosis insights out.**  
An end-to-end medical imaging system that **classifies** and **localizes** oral diseases from a **single image**, with **explainability (XAI)** and a **deployment-ready** stack.

## ✨ What This Does
Upload an oral image and the system will:
1. **Classify** the disease using **ConvNeXt**
2. **Localize** the affected region using **YOLOv11x** (bounding boxes)
3. **Explain** the prediction using **Grad-CAM++** (trustworthy visual heatmaps)
4. Serve results through:
   - **FastAPI** (production-ready backend)
   - **Streamlit** (interactive UI for testing & visualization)

---

## 🎥 Demo

https://github.com/user-attachments/assets/58d28e12-63d9-45bb-ac5a-f1c8ae2c2e8f

---

## 🧠 Models Used
### 1) Classification — ConvNeXt
- Fine-tuned for high-precision oral disease classification
- Augmentations + preprocessing pipeline
- **Test-Time Augmentation (TTA)** to improve robustness

### 2) Localization — YOLOv11x
- Detects disease regions using bounding boxes
- Optimized for real-time inference

### 3) Explainability — Grad-CAM++
- Heatmaps show which regions influenced the classifier prediction
- Helps clinicians/users interpret model decisions

---

## 📊 Metrics (Results)
> Replace any values if you update the training or dataset split.

| Task | Model | Metric | Value |
|------|-------|--------|-------|
| Classification | ConvNeXt | Accuracy | **92.8%** |
| Classification | ConvNeXt | Recall | **93.2%** |
| Localization | YOLOv11x | mAP | **96.1%** |
| Localization | YOLOv11x | mAP@0.5:0.9 | **84.0%** |

### mAP Notes
- **mAP** summarizes detection performance across classes.
- **mAP@0.5:0.9** is stricter (averaged over IoU thresholds), so it’s usually lower.

---

## 🖼️ Screenshots / Visual Results
><img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/3bd2b1d4-0eac-4e66-9e8f-03ee2dea6187" />
<img width="4770" height="2373" alt="mAP" src="https://github.com/user-attachments/assets/7a70ad15-7991-430e-94b8-174230983185" />


### Detection Output (Bounding Boxes)

<p align="center">
  <img src="https://github.com/user-attachments/assets/a8c20e7c-d1d0-4047-bb1b-d3be2dbdc8f8" width="420" />
  <img src="https://github.com/user-attachments/assets/e09b9357-b18e-4556-b26b-0712789c59ed" width="420" />
</p>

### Explainability (Grad-CAM++)

<p align="center">
  <img src="https://github.com/user-attachments/assets/8595274d-83d3-49fa-966d-62ea91f4bf29" width="420" />
  <img src="https://github.com/user-attachments/assets/0f1f6583-afad-44bd-9541-83ab52693013" width="420" />
</p>


---

## 🏗️ System Architecture
**Input Image → Classification (ConvNeXt) → Localization (YOLO) → Explainability (Grad-CAM++) → API Response → UI Visualization**

---

## ⚙️ Tech Stack
- **Deep Learning:** PyTorch
- **Detection:** YOLOv11x
- **Explainability:** Grad-CAM++
- **Backend:** FastAPI
- **Frontend/UI:** Streamlit
- **Deployment:** Docker (optional)

---

## 📁 Project Structure (Suggested)
```
OralVisionDx/
├─ backend/                 # FastAPI app
│  ├─ app/
│  │  ├─ main.py
│  │  ├─ routes/
│  │  ├─ services/
│  │  └─ utils/
│  └─ requirements.txt
├─ ui/                      # Streamlit interface
│  ├─ app.py
│  └─ requirements.txt
├─ models/
│  ├─ classification/       # ConvNeXt weights/config
│  └─ detection/            # YOLO weights/config
├─ notebooks/               # experiments, training logs
├─ assets/                  # images for README (placeholders)
├─ scripts/                 # training/inference helpers
├─ README.md
└─ LICENSE
```

---

## 🚀 Quickstart

### 1) Clone
```bash
git clone https://github.com/faris-agour/OralVisionDx
cd OralVisionDx
```

### 2) Create Environment
```bash
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate
pip install -r backend/requirements.txt
pip install -r ui/requirements.txt
```

### 3) Run Backend (FastAPI)
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4) Run UI (Streamlit)
```bash
cd ui
streamlit run app.py
```

---

### `POST /predict`
- **Input:** image file
- **Output:** class label, confidence, bounding boxes, Grad-CAM++ heatmap (optional)

Example response (shape only):
```json
{
  "class": "TODO",
  "confidence": 0.98,
  "detections": [
    {"label": "lesion", "confidence": 0.95, "bbox": [x1, y1, x2, y2]}
  ],
  "explainability": {
    "gradcam_path": "TODO"
  }
}
```

## 📌 Project Summary (CV / LinkedIn)
**AI-Powered Oral Disease | ConvNeXt, YOLOv11, XAI, FastAPI — Nov 2025**  
- Fine-tuned **ConvNeXt** for classification with **TTA**, achieving **95.8% accuracy** and **96% recall**.  
- Applied **YOLOv11x** for localization, achieving **96% mAP** and **84% mAP@0.5:0.9**.  
- Deployed a production-ready inference pipeline using **FastAPI** and **Streamlit** with integrated **Grad-CAM++** explainability.

