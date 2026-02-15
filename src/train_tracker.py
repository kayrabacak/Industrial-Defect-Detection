import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import time
import os
import argparse

# --- AYARLAR ---
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--data_dir', type=str, default='data/raw/casting_data/casting_data') 
args = parser.parse_args()

PRODUCTION_MODEL_PATH = "models/production_model.pth"

# --- MLFLOW ---
mlflow.set_experiment("Industrial_Defect_Detection")

def train():
    with mlflow.start_run():
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("learning_rate", args.lr)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔥 Cihaz: {device}")

        # 1. Veri Hazırlığı
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x])
                          for x in ['train', 'test']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True)
                       for x in ['train', 'test']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

        # --- 2. MODEL YÜKLEME (KRİTİK DÜZELTME BURADA!) ---
        print("🧠 Model hazırlanıyor...")
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.last_channel, 2)
        
        # Eğer daha önce eğitilmiş model varsa ONU YÜKLE (Hafızayı koru)
        if os.path.exists(PRODUCTION_MODEL_PATH):
            print(f"♻️  Önceki 'Production Model' bulundu! Kaldığı yerden devam ediliyor: {PRODUCTION_MODEL_PATH}")
            model.load_state_dict(torch.load(PRODUCTION_MODEL_PATH, map_location=device))
        else:
            print("✨ Yeni model sıfırdan (ImageNet'ten) başlatılıyor.")

        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier[1].parameters(), lr=args.lr)

        # 3. Eğitim Döngüsü
        best_acc = 0.0
        
        print("🚀 Eğitim Başlıyor...")
        for epoch in range(args.epochs):
            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'Epoch {epoch+1}/{args.epochs} | {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc

        print(f"🏆 En İyi Doğruluk: {best_acc}")
        
        # 4. Modeli Kaydet
        mlflow.pytorch.log_model(model, "model")
        
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), PRODUCTION_MODEL_PATH)
        print(f"✅ PRODUCTION MODEL GÜNCELLENDİ: {PRODUCTION_MODEL_PATH}")

if __name__ == "__main__":
    train()