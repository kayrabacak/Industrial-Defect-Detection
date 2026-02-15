import streamlit as st
import torch
from PIL import Image
from src.inference import load_trained_model, predict_image, CLASS_NAMES, MODEL_PATH
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Industrial AI Eye",
    page_icon="🏭",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (PREMIUM DARK/CYBERPUNK LOOK) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* Title Styling */
    h1 {
        font-family: 'Courier New', monospace;
        color: #00ff41; /* Matrix Green */
        text-align: center;
        text-shadow: 0 0 10px #00ff41;
    }
    
    /* Subheaders */
    h2, h3 {
        color: #ffffff;
        font-family: 'Helvetica', sans-serif;
    }

    /* Upload Button */
    .stFileUploader > div > button {
        background-color: #212121;
        color: #00ff41;
        border: 1px solid #00ff41;
    }

    /* Custom Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #1a1a1a;
        border: 1px solid #333;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background-color: rgba(0, 255, 65, 0.1);
        border-left: 5px solid #00ff41;
    }
    .stError {
        background-color: rgba(255, 0, 0, 0.1);
        border-left: 5px solid #ff0000;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("🏭 Fabrika Kontrol")
    st.info("Bu sistem, üretim hattındaki döküm parçalarını yapay zeka ile analiz eder.")
    
    st.markdown("---")
    st.caption("Geliştirici: AI Assistant & User")
    st.caption("Version: 1.0.0 (Beta)")
    
    st.markdown("### 📊 Model Durumu")
    st.success("Model Aktif (MobileNetV2)")

# --- MAIN PAGE ---
st.title("👁️ INDUSTRIAL DEFECT DETECTOR")
st.markdown("<p style='text-align: center; color: #888;'>Yapay Zeka Destekli Kalite Kontrol Sistemi</p>", unsafe_allow_html=True)
st.markdown("---")

# --- MODEL LOADING (CACHED) ---
@st.cache_resource
def load_model():
    # Model path might need adjustment depending on where you run streamlit
    return load_trained_model(MODEL_PATH)

try:
    with st.spinner("🧠 Yapay Zeka Modeli Yükleniyor..."):
        model = load_model()
    # Toast message for success (unobtrusive)
    # st.toast("Model Hazır!", icon="🚀")
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu: {e}")
    st.stop()

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("📸 Analiz Edilecek Parçayı Yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Yüklenen Görüntü")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True, caption="İşleniyor...")
        
        # Save temp file for inference
        with open("temp_image.jpg", "wb") as f:
            uploaded_file.getbuffer()  # Reset buffer just in case
            image.save(f) # Save as image ensures format correctness
            
    # Perform Prediction
    with col2:
        st.subheader("🔍 Analiz Sonucu")
        
        # Simulate processing time for "Visual Show" effect
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            status_text.text(f"Pikseller taranıyor... %{i+1}")
            progress_bar.progress(i + 1)
            time.sleep(0.01) # 1 saniyelik görsel şov
            
        label, score = predict_image(model, "temp_image.jpg")
        
        # Display Result
        if "ok" in label:
            st.success("✅ SAĞLAM (OK)")
            st.metric(label="Güven Skoru", value=f"%{score*100:.2f}", delta="Onaylandı")
            st.balloons() # Şov efekti 1
        else:
            st.error("🚨 HATALI (DEFECT)")
            st.metric(label="Güven Skoru", value=f"%{score*100:.2f}", delta="-Reddedildi", delta_color="inverse")
            # Warning effect
            st.markdown("""
                <div style="padding: 20px; background-color: #ff4b4b; color: white; border-radius: 10px; text-align: center;">
                    <h3>⚠️ ÜRETİM HATASI TESPİT EDİLDİ</h3>
                    <p>Lütfen parçayı ayırın ve operatöre bildirin.</p>
                </div>
            """, unsafe_allow_html=True)
else:
    # Empty state show
    st.info("👆 Başlamak için yukarıdan bir fotoğraf yükleyin.")
    
    # Optional: Gallery of examples (if files exist, otherwise skip)
    st.markdown("---")
    st.markdown("### 🧪 Test Örnekleri İle Deneyin")
    
    example_cols = st.columns(3)
    # You could add buttons here to load sample images if they exist
