import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- AYARLAR ---
MODEL_PATH = "models/defect_detection_model_v1.pth"  # İndirdiğin modelin yolu
CLASS_NAMES = ['def_front', 'ok_front'] # Eğitimdeki sınıf sırası (Genelde alfabetiktir)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    """
    Model mimarisini tekrar oluşturur.
    ÖNEMLİ: Colab'de ne yaptıysak aynısını burada tanımlamalıyız.
    """
    model = models.mobilenet_v2(pretrained=False) # Ağırlıkları birazdan biz yükleyeceğiz
    model.classifier[1] = nn.Linear(model.last_channel, 2) # 2 Sınıf (Hatalı/Sağlam)
    return model

def load_trained_model(path):
    print(f"🔄 Model yükleniyor: {path}")
    model = get_model()
    # Eğitilmiş ağırlıkları (state_dict) yükle
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # Eğitim modunu kapat (Dropout vs. çalışmasın)
    print("✅ Model başarıyla yüklendi!")
    return model

def predict_image(model, image_path):
    # Görüntü Ön İşleme (Colab'dekiyle AYNI olmalı)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3), # 1 kanalı 3 yap
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Resmi aç
    if not os.path.exists(image_path):
        return "HATA: Dosya bulunamadı!"
        
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0) # Batch boyutu ekle: [1, 3, 224, 224]
    image_tensor = image_tensor.to(DEVICE)
    
    # Tahmin
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_p, top_class = probabilities.topk(1, dim=1)
        
        score = top_p.item()
        label = CLASS_NAMES[top_class.item()]
        
    return label, score

# --- TEST KISMI ---
if __name__ == "__main__":
    # Test edilecek bir resim yolu ver (Bilgisayarından rastgele bir test resmi seç)
    # Örnek: "data/raw/test/def_front/cast_def_0_1059.jpeg"
    test_image_path = "test_resmi.jpeg" 
    
    # Eğer test resmi yoksa uyarı verelim
    if not os.path.exists(test_image_path):
        print(f"⚠️ Lütfen '{test_image_path}' adında bir resmi proje ana dizinine koyun.")
    else:
        # Modeli yükle ve tahmin et
        model = load_trained_model(MODEL_PATH)
        sonuc, guven = predict_image(model, test_image_path)
        
        print("\n" + "="*30)
        print(f"📸 İncelenen Resim: {test_image_path}")
        print(f"🔍 TAHMİN: {sonuc.upper()}")
        print(f"📊 Güven Skoru: %{guven*100:.2f}")
        
        if "def" in sonuc:
            print("🚨 ALARM: Ürün HATALI! Üretim bandını durdurun.")
        else:
            print("✅ Ürün SAĞLAM. Devam edilebilir.")
        print("="*30)