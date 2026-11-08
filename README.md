# WasteOps

### What if we could understand waste before we collect it?

**An AI-powered waste intelligence platform exploring how computer vision and machine learning can transform waste monitoring and management.**

Waste management remains heavily dependent on manual inspection, fragmented information, and reactive decision-making.

What if ordinary images could help us understand what's inside a waste bin, estimate how full it is, identify contamination, and predict when collection is needed?

That's the idea behind WasteOps.

The goal is to build a software-first intelligence platform that transforms visual waste data into actionable operational insights.

Not just identifying waste. Understanding it.

---

## The Vision

WasteOps aims to bring together computer vision, machine learning, and predictive analytics to make waste monitoring more accessible, measurable, and intelligent.

The platform is being developed around five core capabilities:

| Capability | Objective |
|---|---|
| Bin Fill Estimation | Estimate visible bin occupancy and remaining capacity |
| Material Intelligence | Identify materials present in waste streams |
| Contamination Detection | Identify materials that do not belong in a particular waste stream |
| Collection Forecasting | Predict when a bin may reach its collection threshold |
| Waste Analytics | Convert observations into operational insights |

These represent the broader development vision. The current release establishes the initial machine-learning foundation.

---

## Version 1.0 — The Foundation

The first release focuses on building and evaluating a waste-image classification pipeline.

### Implemented

- Fine-tuned MobileNetV3 image classification model.
- UCI RealWaste dataset integration.
- Training, validation, and testing pipeline.
- Model checkpoint generation.
- Evaluation using accuracy and macro-F1.
- FastAPI backend foundation.
- Automated tests.

### Initial Model Results

| Metric | Result |
|---|---:|
| Dataset | UCI RealWaste |
| Training samples | 3,801 |
| Validation samples | 475 |
| Test samples | 476 |
| Test accuracy | **86.13%** |
| Test macro-F1 | **0.8614** |

The model achieved a validation macro-F1 of approximately 0.8976 during training.

**Evaluation note:** The current evaluation uses an image-level split. It does not establish generalization to unseen waste facilities or different camera environments. Stronger site-held-out and cross-environment evaluations are planned.

---

## Technology Stack

- Python
- PyTorch
- Torchvision
- MobileNetV3
- FastAPI
- Scikit-learn
- Pillow

---

## Development Roadmap

### Phase 1 — Visual Waste Intelligence

- [x] Build the initial classification pipeline.
- [x] Train and evaluate MobileNetV3.
- [x] Establish a reproducible training workflow.
- [ ] Improve robustness across different environments.

### Phase 2 — Intelligent Bin Monitoring

- [ ] Develop image-based bin-fill estimation.
- [ ] Explore calibrated occupancy measurement.
- [ ] Estimate remaining bin capacity.
- [ ] Build threshold-based collection alerts.

### Phase 3 — Mixed-Waste Understanding

- [ ] Explore object detection and instance segmentation.
- [ ] Estimate material composition in mixed-waste scenes.
- [ ] Develop contamination detection.
- [ ] Investigate material recovery recommendations.

### Phase 4 — Predictive Waste Operations

- [ ] Build time-series collection forecasting.
- [ ] Estimate bin accumulation rates.
- [ ] Prioritize collection based on predicted fill levels.
- [ ] Develop a unified waste-monitoring dashboard.

### Phase 5 — Generalization and Deployment

- [ ] Evaluate on unseen locations and environments.
- [ ] Improve uncertainty estimation and confidence calibration.
- [ ] Explore lightweight inference and deployment.
- [ ] Investigate software-first integrations for individuals and waste-management businesses.

Roadmap items are planned research and engineering objectives, not currently validated features.

---

## Dataset

WasteOps v1.0 uses the UCI RealWaste dataset, containing real-world landfill images across nine waste categories.

Dataset: https://archive.ics.uci.edu/dataset/908/realwaste

The dataset is not included in this repository. Download it from the official source and follow its licensing and attribution requirements.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/sreevatsarajesh/Wasteops-AI.git
cd Wasteops-AI
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -e .
```

### 4. Download the dataset

Download RealWaste from the official UCI repository and place it in the directory expected by the dataset preparation scripts.

### 5. Train the model

```bash
PYTHONPATH=. python scripts/train.py --epochs 12 --workers 0
```

### 6. Run the tests

```bash
python -m pytest -q
```

The trained model checkpoint is maintained through Git LFS.

---

## Research Direction

A central research challenge for WasteOps is moving from closed-set image classification toward reliable real-world waste intelligence.

Questions I want to investigate include:

- Can a vision model estimate bin occupancy reliably under changing camera angles and lighting?
- Can mixed-waste scenes be decomposed into meaningful material categories?
- Can contamination be identified using visual material predictions and waste-stream rules?
- Can repeated visual observations support useful collection forecasts?
- Can these capabilities work across different facilities without expensive specialized hardware?

The emphasis is not simply on achieving good benchmark accuracy. It is on building systems that remain useful outside controlled datasets.

---

## Collaboration

I'm interested in connecting with people working in:

- Computer vision and deep learning.
- Waste management and recycling.
- Sustainability and climate technology.
- Predictive analytics and intelligent infrastructure.

Technical feedback, dataset recommendations, research collaboration, and contributions are welcome.

GitHub: https://github.com/sreevatsarajesh/Wasteops-AI

---

## Project Status

**Version:** 1.0.0  
**Stage:** Initial research and development release  
**Primary focus:** Waste-image classification foundation

WasteOps is an evolving project. Its reported performance applies to the evaluation setup described above. Future capabilities will be developed and validated incrementally.

**Building toward a world where waste is understood, not just collected.**