
import streamlit as st
import joblib
import pandas as pd
import os

#Config 
MODEL_PATH = "./models/random_forest.joblib"  

IPHONE_MODELS = [
    "iPhone SE (2nd Gen)", "iPhone SE (3rd Gen)",
    "iPhone 11", "iPhone 11 Pro", "iPhone 11 Pro Max",
    "iPhone 12", "iPhone 12 Mini", "iPhone 12 Pro", "iPhone 12 Pro Max",
    "iPhone 13", "iPhone 13 Mini", "iPhone 13 Pro", "iPhone 13 Pro Max",
    "iPhone 14", "iPhone 14 Plus", "iPhone 14 Pro", "iPhone 14 Pro Max",
    "iPhone 15", "iPhone 15 Plus", "iPhone 15 Pro", "iPhone 15 Pro Max",
]
STORAGE_OPTIONS  = ["64GB", "128GB", "256GB", "512GB", "1TB"]
COLOUR_OPTIONS   = ["Black", "White", "Blue", "Red", "Green", "Purple", "Yellow", "Gold", "Silver", "Midnight", "Starlight"]
YES_NO           = ["No", "Yes"]
AVAILABILITY     = ["Available", "Not Available"]
# ─────────────────────────────────────────────────────────────────────────────


st.set_page_config(
    page_title="iPhone Resale Price Predictor",
    page_icon="📱",
    layout="centered",
)

# Custom CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #0a0a0f; }

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #ffffff18;
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    color: #e2e8f0;
    margin: 0 0 0.3rem 0;
    letter-spacing: -1px;
}
.hero p { color: #94a3b8; font-size: 0.95rem; margin: 0; }

.result-box {
    background: linear-gradient(135deg, #0f3460, #1a1a2e);
    border: 1px solid #3b82f6aa;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-box .price {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    color: #60a5fa;
    letter-spacing: -2px;
}
.result-box .label { color: #94a3b8; font-size: 0.9rem; margin-bottom: 0.5rem; }
.result-box .usd   { color: #64748b; font-size: 1rem; margin-top: 0.3rem; }

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #3b82f6;
    margin: 1.5rem 0 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Header 
st.markdown("""
<div class="hero">
    <h1>📱 iPhone Resale Price</h1>
    <p>Dự đoán giá iPhone cũ dựa trên Ensemble Learning 
</div>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def prepare_input(data):
    """Convert storage string to numeric, convert Yes/No to boolean."""
    data = data.copy()
    # Extract numeric value from storage (e.g., "64GB" → 64)
    if "storage" in data.columns:
        data["storage"] = data["storage"].str.extract(r"(\d+)").astype(int)
    # Convert Yes/No to True/False
    for col in ["battery_renew", "screen_replacement", "display_replacement", "screen_damages", "backglass_damages"]:
        if col in data.columns:
            data[col] = (data[col] == "Yes").astype(bool)
    # Map availability
    if "availability" in data.columns:
        data["availability"] = data["availability"].map({"Available": "full-set", "Not Available": "phone-only"})
    return data

model = load_model(MODEL_PATH)
if model is None:
    st.error(f"⚠️ Không tìm thấy model tại `{MODEL_PATH}`. Hãy chạy `train.py` trước.")
    st.stop()

#  Input form 
st.markdown('<div class="section-title">Thông tin iPhone</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    iphone_model      = st.selectbox("📱 Model", IPHONE_MODELS, index=6)
    storage           = st.selectbox("💾 Storage", STORAGE_OPTIONS, index=1)
    colour            = st.selectbox("🎨 Màu sắc", COLOUR_OPTIONS)
    availability      = st.selectbox("🏪 Tình trạng có hàng", AVAILABILITY)

with col2:
    battery_health        = st.slider("🔋 Battery Health (%)", 50, 100, 87)
    months_since_release  = st.slider("📅 Số tháng kể từ khi ra mắt", 1, 60, 24)
    ios_updates           = st.slider("⚙️ Số lần cập nhật iOS", 0, 20, 8)
    price_at              = st.number_input("💵 Giá gốc khi mua (LKR)", min_value=50000, max_value=1000000, value=250000, step=5000)

st.markdown('<div class="section-title">Tình trạng máy</div>', unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)
with col3:
    screen_replacement  = st.selectbox("🖥️ Thay màn hình?",  YES_NO)
    screen_damages      = st.selectbox("💥 Màn hình bị vỡ?", YES_NO)
with col4:
    display_replacement = st.selectbox("📺 Thay display?",    YES_NO)
    backglass_damages   = st.selectbox("🔙 Kính lưng bị vỡ?",YES_NO)
with col5:
    battery_renew       = st.selectbox("🔋 Thay pin mới?",    YES_NO)
    exchange_rate       = st.number_input("💱 Tỷ giá USD→LKR", min_value=200, max_value=400, value=323)

# Predict 
st.markdown("")
predict_btn = st.button("🔍 Dự đoán giá", use_container_width=True, type="primary")

if predict_btn:
    input_data = pd.DataFrame([{
        "Model":                       iphone_model,
        "battery_health":              battery_health,
        "battery_renew":               battery_renew,
        "screen_replacement":          screen_replacement,
        "display_replacement":         display_replacement,
        "storage":                     storage,
        "colour":                      colour,
        "backglass_damages":           backglass_damages,
        "screen_damages":              screen_damages,
        "availability":                availability,
        "ios_updates":                 ios_updates,
        "Months_since_release":        months_since_release,
        "price_at":                    price_at,
        "Exchange_rate_1_USD_to_LKR":  exchange_rate,
    }])
    
    input_data = prepare_input(input_data)

    try:
        predicted_lkr = model.predict(input_data)[0]
        predicted_usd = predicted_lkr / exchange_rate

        st.markdown(f"""
        <div class="result-box">
            <div class="label">Giá dự đoán</div>
            <div class="price">LKR {predicted_lkr:,.0f}</div>
            <div class="usd">≈ USD {predicted_usd:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        # Gợi ý nhanh
        st.markdown("")
        if battery_health < 80:
            st.info("💡 Battery health thấp — cân nhắc thay pin để tăng giá trị máy.")
        if screen_damages == "Yes" or backglass_damages == "Yes":
            st.warning("⚠️ Máy có hư hỏng — giá có thể thấp hơn thị trường.")

    except Exception as e:
        st.error(f"❌ Lỗi khi dự đoán: {e}")
        st.info("Kiểm tra lại tên cột trong input_data có khớp với dataset không.")

# Footer 
st.markdown("---")