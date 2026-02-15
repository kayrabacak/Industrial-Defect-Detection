from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
import uuid
from datetime import datetime

# --- KRİTİK AYAR DEĞİŞİKLİĞİ ---
# Artık versiyonlu isim (v1) yerine, hep güncellenen ismi kullanıyoruz.
# Eğer production_model.pth henüz yoksa (ilk çalıştırış), v1'e bakabiliriz.
if os.path.exists("models/production_model.pth"):
    MODEL_PATH = "models/production_model.pth"
else:
    MODEL_PATH = "models/defect_detection_model_v1.pth" # Yedek/Başlangıç

DRIFT_FOLDER = "data/drifted_samples"
CONFIDENCE_THRESHOLD = 80.0 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['def_front', 'ok_front']

app = FastAPI(title="Self-Healing Industrial API")

# --- MODEL YÜKLEME FONKSİYONU ---
def load_model_logic():
    print(f"⏳ Model yükleniyor: {MODEL_PATH}")
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ Model hafızaya alındı!")
    else:
        print("⚠️ HATA: Model dosyası bulunamadı!")
    
    model.to(DEVICE)
    model.eval()
    return model

# Global değişken (İlk açılışta yükle)
model = load_model_logic()

# --- GÖRÜNTÜ İŞLEME (Değişmedi) ---
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0).to(DEVICE), image

# --- YENİ ENDPOINT: HOT RELOAD (Modeli Canlıda Değiştir) ---
@app.post("/update-model")
def update_model():
    global model, MODEL_PATH
    
    # Eğitim scripti yeni modeli 'production_model.pth' olarak kaydetti.
    # Biz de yolumuzu oraya çeviriyoruz.
    MODEL_PATH = "models/production_model.pth"
    
    print("🔄 SİNYAL ALINDI: Model güncelleniyor...")
    try:
        new_model = load_model_logic()
        model = new_model # RAM'deki eski modeli yenisiyle değiştir
        return {"status": "success", "message": "API başarıyla güncellendi. Yeni beyin devrede!"}
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

# --- TAHMİN ENDPOINT (Aynı) ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        tensor, original_image = transform_image(image_bytes)
    except Exception as e:
        return JSONResponse(content={"error": "Görsel işlenemedi"}, status_code=400)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_p, top_class = probabilities.topk(1, dim=1)
        confidence = top_p.item() * 100
        label = CLASS_NAMES[top_class.item()]

    status = "CONFIDENT"
    saved_path = None
    
    if confidence < CONFIDENCE_THRESHOLD:
        status = "UNCERTAIN_DATA_SAVED"
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
        save_path = os.path.join(DRIFT_FOLDER, filename)
        original_image.save(save_path)
        saved_path = save_path
        print(f"📉 Drift Yakalandı: {save_path}")

    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "system_status": status,
        "saved_for_retraining": saved_path is not None
    }