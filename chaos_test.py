import requests
from PIL import Image, ImageEnhance, ImageFilter
import io
import numpy as np
import random

# --- AYARLAR ---
API_URL = "http://127.0.0.1:8000/predict"
IMAGE_PATH = "test_resmi.jpeg"  # Senin test resmin

# --- YARDIMCI FONKSİYONLAR (MODÜLER YAPI) ---

def apply_brightness(image, factor):
    """Parlaklığı değiştirir (0.0 = Siyah, 1.0 = Orijinal)"""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def apply_blur(image, radius):
    """Bulanıklık ekler (radius arttıkça bulanıklaşır)"""
    return image.filter(ImageFilter.GaussianBlur(radius))

def apply_rotation(image, angle):
    """Resmi döndürür (90, 180, 270)"""
    return image.rotate(angle, expand=True)

def apply_noise(image, factor):
    """Resme rastgele gürültü (Noise) ekler"""
    img_array = np.array(image)
    noise = np.random.randint(0, 50, img_array.shape, dtype='uint8')
    
    # Gürültüyü ekle (Ama çok bozmamak için factor ile çarp)
    # Basit bir salt-and-pepper simülasyonu
    noisy_image = Image.fromarray(np.clip(img_array + noise * factor, 0, 255).astype('uint8'))
    return noisy_image

def send_to_api(image, scenario_name):
    """Resmi API'ya gönderir ve sonucu raporlar"""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    files = {'file': (f'{scenario_name}.jpg', img_byte_arr, 'image/jpeg')}
    
    try:
        response = requests.post(API_URL, files=files)
        result = response.json()
        
        conf = result['confidence']
        pred = result['prediction']
        
        # Emoji ile durum göstergesi
        status = "✅" if conf > 80 else "⚠️ RİSK (Drift)"
        print(f"{scenario_name:<20} | {pred:<10} | %{conf:<6} | {status}")
        
    except Exception as e:
        print(f"{scenario_name:<20} | HATA: {str(e)}")

# --- ANA TEST MERKEZİ ---
def run_chaos_monkey():
    original_img = Image.open(IMAGE_PATH).convert('RGB')
    
    print("\n" + "="*60)
    print(f"🐵 CHAOS MONKEY TEST BAŞLIYOR: {IMAGE_PATH}")
    print("="*60)
    print(f"{'Senaryo':<20} | {'Tahmin':<10} | {'Güven':<7} | {'Durum'}")
    print("-" * 60)

    # 1. PARLAKLIK TESTLERİ
    send_to_api(apply_brightness(original_img, 0.5), "Az Işık (%50)")
    send_to_api(apply_brightness(original_img, 0.2), "Karanlık (%20)")

    # 2. BULANIKLIK TESTLERİ
    send_to_api(apply_blur(original_img, 2), "Hafif Bulanık")
    send_to_api(apply_blur(original_img, 5), "Çok Bulanık")

    # 3. DÖNDÜRME TESTLERİ (Senin istediğin kritik kısım)
    send_to_api(apply_rotation(original_img, 90),  "Döndür (90°)")
    send_to_api(apply_rotation(original_img, 180), "Döndür (180°)")
    send_to_api(apply_rotation(original_img, 270), "Döndür (270°)")

    # 4. GÜRÜLTÜ TESTİ
    send_to_api(apply_noise(original_img, 0.5), "Kamera Paraziti")

    print("="*60 + "\n")

if __name__ == "__main__":
    run_chaos_monkey()