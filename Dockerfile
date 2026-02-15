# 1. Base Image: Python 3.9'un hafif sürümünü kullan (Linux tabanlı)
FROM python:3.9-slim

# 2. Çalışma dizinini ayarla (Container'ın içindeki klasör)
WORKDIR /app

# 3. Gereklilikleri kopyala ve kur (Önce bunu yapıyoruz ki Docker Cache kullansın)
COPY requirements.txt .
# --no-cache-dir ile image boyutunu küçük tutuyoruz
RUN pip install --no-cache-dir -r requirements.txt

# 4. Tüm proje dosyalarını container içine kopyala
COPY . .

# 5. Modeller klasörünün varlığını ve modelin orada olduğunu kontrol et (Opsiyonel ama güvenli)
# (Eğer modelin 'models' klasöründe değilse hata verecektir, burayı kontrol et)

# 6. Portu dışarı aç (FastAPI varsayılan olarak 8000 veya biz 80 yapacağız)
EXPOSE 80

# 7. Uygulamayı başlat
# host 0.0.0.0 demek, dış dünyadan gelen isteklere açığım demektir.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]