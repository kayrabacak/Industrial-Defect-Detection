
# 🏭 Industrial Defect Detection (Endüstriyel Hata Tespit Sistemi)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Yapay zeka destekli, endüstriyel parçaların kalite kontrolünü gerçekleştiren web tabanlı bir analiz aracıdır. Derin öğrenme modelleri (MobileNetV2) kullanarak üretim hattındaki hatalı ürünleri tespit eder ve operatöre görsel geri bildirim sağlar.

## 🌟 Özellikler

- **Anlık Analiz:** Yüklenen görüntüleri saniyeler içinde analiz eder.
- **Yüksek Doğruluk:** Eğitilmiş derin öğrenme modelleri ile hassas tespit.
- **Kullanıcı Dostu Arayüz:** Streamlit tabanlı modern ve karanlık mod arayüzü.
- **Görsel Geri Bildirim:** Hatalı ve sağlam ürünler için renk kodlu uyarılar ve animasyonlar.
- **Kolay Kurulum:** Tek tıkla çalıştırılabilir yapı.

## 🛠️ Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

1. **Projeyi Klonlayın:**
   ```bash
   git clone https://github.com/kullaniciadi/industrial-defect-detection.git
   cd industrial-defect-detection
   ```

2. **Sanal Ortam Oluşturun (Opsiyonel ama Önerilir):**
   ```bash
   python -m venv venv
   # Windows için:
   .\venv\Scripts\activate
   # Linux/Mac için:
   source venv/bin/activate
   ```

3. **Gerekli Kütüphaneleri Yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Kullanım

Uygulamayı başlatmak için iki yöntem bulunmaktadır:

### Yöntem 1: Hızlı Başlatma (Windows)
Klasör içerisindeki `run_dashboard.bat` dosyasına çift tıklayarak uygulamayı otomatik olarak başlatabilirsiniz.

### Yöntem 2: Terminal Üzerinden
```bash
streamlit run dashboard.py
```

Uygulama tarayıcınızda otomatik olarak açılacaktır (genellikle `http://localhost:8501`).
Arayüz üzerinden "Browse files" butonuna tıklayarak veya sürükle-bırak yöntemiyle test etmek istediğiniz parça görselini yükleyin. Sistem otomatik olarak analizi gerçekleştirecektir.

## 📂 Proje Yapısı

```
industrial-defect-detection/
├── api/                # API entegrasyonları (FastAPI vb.)
├── data/               # Veri setleri ve örnek görseller
├── models/             # Eğitilmiş model dosyaları (.pth, .pt)
├── notebooks/          # Deney ve eğitim not defterleri (Jupyter)
├── src/                # Kaynak kodlar ve yardımcı modüller
├── venv/               # Sanal ortam dosyaları
├── dashboard.py        # Streamlit ana uygulama dosyası
├── requirements.txt    # Proje bağımlılıkları
├── run_dashboard.bat   # Windows için hızlı başlatma betiği
└── README.md           # Proje dokümantasyonu
```

## 💻 Teknoloji Yığını

- **Dil:** Python
- **Arayüz:** Streamlit
- **Yapay Zeka:** PyTorch, Torchvision
- **Görüntü İşleme:** PIL (Pillow), Numpy

## 🤝 Katkıda Bulunma

1. Bu projeyi forklayın (https://github.com/kullaniciadi/industrial-defect-detection/fork)
2. Özellik dalınızı oluşturun (`git checkout -b feature/YeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Dalınıza push yapın (`git push origin feature/YeniOzellik`)
5. Bir Pull Request oluşturun

## 📝 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.
