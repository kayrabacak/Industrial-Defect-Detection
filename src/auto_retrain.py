import os
import shutil
import subprocess
import requests
import uuid
import random
from datetime import datetime
from PIL import Image, ImageEnhance, ImageOps # Görüntü işleme kütüphaneleri

# --- AYARLAR ---
DRIFT_FOLDER = "data/drifted_samples"
TRAIN_DEFECT_DIR = "data/raw/casting_data/casting_data/train/def_front"
API_UPDATE_URL = "http://127.0.0.1:8000/update-model"

MIN_SAMPLES_TO_RETRAIN = 5 
OVERSAMPLE_FACTOR = 10 

def augment_image(image):
    """
    Resmi rastgele değiştirir (Mutasyon)
    Böylece model ezberlemez, öğrenir!
    """
    # 1. Rastgele Döndürme (-15 ile +15 derece arası)
    angle = random.randint(-15, 15)
    img = image.rotate(angle)
    
    # 2. Rastgele Parlaklık (%80 ile %120 arası)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # 3. Rastgele Yatay Çevirme (Mirror) - %50 şansla
    if random.random() > 0.5:
        img = ImageOps.mirror(img)
        
    return img

def check_and_retrain():
    files = [f for f in os.listdir(DRIFT_FOLDER) if f.endswith('.jpg')]
    file_count = len(files)
    
    print(f"🔍 Drift Klasörü: {file_count} dosya var.")

    if file_count < MIN_SAMPLES_TO_RETRAIN:
        print(f"⏳ Yetersiz veri. En az {MIN_SAMPLES_TO_RETRAIN} gerekli. Bekleniyor...")
        return

    print(f"🚨 EŞİK AŞILDI! Akıllı Retraining başlıyor...")
    
    count = 0
    for f in files:
        src_path = os.path.join(DRIFT_FOLDER, f)
        
        try:
            # Resmi aç
            original_img = Image.open(src_path)
            
            # OVERSAMPLE_FACTOR kadar TÜREV üret
            for i in range(OVERSAMPLE_FACTOR):
                # Resmi mutasyona uğrat
                aug_img = augment_image(original_img)
                
                # Yeni isimle kaydet
                new_name = f"aug_{uuid.uuid4().hex[:8]}.jpg"
                dst_path = os.path.join(TRAIN_DEFECT_DIR, new_name)
                
                # RGB formatına çevir (JPEG kaydederken hata olmasın)
                aug_img.convert('RGB').save(dst_path)
            
            # İş bitince orijinali de ham haliyle bir kere kaydet (veya silebilirsin)
            original_img.save(os.path.join(TRAIN_DEFECT_DIR, f))
            
            # Orijinali drift klasöründen sil
            original_img.close()
            os.remove(src_path)
            count += 1
            
        except Exception as e:
            print(f"Dosya işlenirken hata: {f} -> {e}")

    print(f"📦 {count} adet zor veri, {OVERSAMPLE_FACTOR}x varyasyonla çoğaltılarak havuza eklendi.")

    # 4. Eğitimi Başlat
    print("🚀 Model eğitimi (Fine-Tuning) başlatılıyor...")
    try:
        subprocess.run(["python", "src/train_tracker.py", "--epochs", "3", "--lr", "0.00005"], check=True)
        print("🎉 YENİDEN EĞİTİM TAMAMLANDI!")
        
        # 5. API Güncelle
        try:
            requests.post(API_UPDATE_URL)
            print("✅ API GÜNCELLENDİ.")
        except:
            print("❌ API KAPALI.")

    except subprocess.CalledProcessError as e:
        print(f"❌ Eğitim hatası: {e}")

if __name__ == "__main__":
    check_and_retrain()