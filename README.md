
# 🏭 Self-Healing Industrial Defect Detection Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=for-the-badge&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-High%20Performance-green?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blueviolet?style=for-the-badge&logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker)

> **"A computer vision system that learns from its own mistakes in real-time."**

---

## 📖 Overview

This project implements an **End-to-End MLOps pipeline** for industrial quality control (casting product inspection). Unlike traditional static models, this system features an **Autonomous Active Learning Loop**.

It detects **"Drift"** (low-confidence predictions due to lighting, blur, or rotation), automatically collects these edge cases, retrains itself using **Incremental Learning** to prevent catastrophic forgetting, and updates the live API without downtime via **Hot Reloading**.

### 🚀 Key Features

* **🔍 Robust Computer Vision:** MobileNetV2 based binary classification (Defect vs. OK).
* **🧠 Self-Healing Mechanism:** Automatically captures uncertain predictions (<80% confidence).
* **🔄 Active Learning Loop:** Retrains the model on "hard" examples using **Data Augmentation** (Rotation, Flip, Brightness) & **Oversampling**.
* **⚡ Hot-Swapping:** Updates the production model in the live API (`/update-model`) without restarting the server.
* **🖥️ Cyberpunk Dashboard:** A Streamlit-based visual interface for real-time inspection.
* **🛡️ Chaos Engineering:** Includes a `Chaos Monkey` script to stress-test the model with Blur, Rotation, Noise, and Low Light.
* **📊 Experiment Tracking:** MLflow integration for logging metrics and model versions.

---

## 🏗️ System Architecture

The system is designed to close the loop between **Inference** and **Training**.

```mermaid
graph TD
    User[Camera / Streamlit UI] -->|Image| API(FastAPI Endpoint)
    API -->|Predict| Decision{Confidence > 80%?}
    
    Decision -- Yes --> Response[Return Result: OK/DEFECT]
    Decision -- No --> Drift[Save to Drift Folder]
    
    Drift --> Accumulate[Wait for Threshold N=5]
    Accumulate --> Trigger[Trigger Auto-Retrain]
    
    Trigger --> Augment[Data Augmentation & Oversampling]
    Augment --> Train[Fine-Tune Model (Incremental)]
    
    Train --> Save[Update production_model.pth]
    Save --> Signal[Send Hot-Reload Signal]
    Signal --> API
```

## 🛠️ Installation

### Prerequisites
* Python 3.9+
* Docker (Optional)

### 1. Clone the Repository
```bash
git clone https://github.com/kayrabacak/Industrial-Defect-Detection.git
cd Industrial-Defect-Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run with Docker (Recommended)
```bash
docker build -t defect-detector:v1 .
docker run -p 8000:80 defect-detector:v1
```

## 🖥️ Usage & Workflow

### 1. Start the API (The Brain)
Start the FastAPI server locally. This handles the model logic and hot-reloading.
```bash
uvicorn api.main:app --reload
```
*API is now running at http://127.0.0.1:8000*

### 2. Launch the Dashboard (The Eyes)
Open the visual interface to interact with the model manually.
```bash
streamlit run dashboard.py
```
*Access via browser at http://localhost:8501*

### 3. Simulate Edge Cases (Chaos Monkey)
Run the stress-test script to send challenging images (blurred, rotated, dark) to the API.
```bash
python chaos_test.py
```
*Observe the terminal: The system will flag low-confidence predictions as ⚠️ RİSK (Drift) and save them.*

### 4. Trigger the Feedback Loop
Run the autonomous retraining agent. It checks the drift folder, augments the data, and retrains the model.
```bash
python src/auto_retrain.py
```
*Observe: The script will fine-tune the model and trigger the API to update itself automatically.*

## 📊 Results (Real World Performance)

The table below demonstrates the system's ability to adapt to new environments (e.g., camera rotation or low light) without manual intervention.

| Scenario | Condition | Initial Confidence | After Auto-Retrain | Status |
| :--- | :--- | :--- | :--- | :--- |
| Normal | 100% Brightness | 99.9% | 99.9% | ✅ Stable |
| Drift | 20% Brightness | 74.3% (Fail) | 88.8% (Pass) | 🚀 Improved |
| Edge Case | 90° Rotation | 55.2% (Fail) | 100.0% (Pass) | 🚀 Solved |
| Edge Case | Motion Blur | 63.7% (Fail) | 99.7% (Pass) | 🚀 Solved |

## 📂 Project Structure

```text
industrial-defect-detection/
│
├── api/
│   └── main.py              # FastAPI app with Hot-Reload logic
├── dashboard.py             # [UI] Streamlit Visual Interface 🎨
├── data/
│   ├── raw/                 # Training dataset
│   └── drifted_samples/     # Storage for low-confidence images
├── models/
│   ├── production_model.pth # The active brain of the system
│   └── backup/              # Original baseline model
├── src/
│   ├── train_tracker.py     # Training script with MLflow & Incremental Learning
│   └── auto_retrain.py      # The "Manager" script (Monitors drift -> Triggers training)
├── chaos_test.py            # Stress testing tool (Blur, Noise, Rotation)
├── Dockerfile               # Container configuration
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

## 🧠 Technical Deep Dive

### Resolving Catastrophic Forgetting
Traditional fine-tuning can make a model forget previous knowledge. We solved this by:
1. **Incremental Learning:** Loading the existing `production_model.pth` weights instead of starting from scratch.
2. **Conservative Training:** Using a very low Learning Rate (0.00005) to gently adjust weights.
3. **Data Augmentation:** Using Pillow to create variations (rotation, flip, brightness) of the drift data to prevent overfitting on specific samples.

### Hot Reloading Strategy
The API exposes an `/update-model` endpoint. When retraining is complete, `auto_retrain.py` sends a POST request. The API reloads the state dictionary into the RAM without killing the process, ensuring **Zero Downtime**.

## 🤝 Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## 📝 License
This project is licensed under the MIT License.
