import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Giải thuật QHTT", layout="wide", page_icon="📈")

st.title("📈 Chương trình giải Quy Hoạch Tuyến Tính Tổng Quát")
st.markdown("---")

# =========================================================================
# QUẢN LÝ SESSION STATE
# =========================================================================
if "n_vars" not in st.session_state: st.session_state.n_vars = 2
if "n_cons" not in st.session_state: st.session_state.n_cons = 3
if "opt_type" not in st.session_state: st.session_state.opt_type = "MAX"
if "init_obj" not in st.session_state:
    st.session_state.init_obj = pd.DataFrame([[0.0, 0.0]], columns=["x1", "x2"])
if "init_cons" not in st.session_state:
    st.session_state.init_cons = pd.DataFrame([[0.0, 0.0, "<=", 0.0] for _ in range(3)], columns=["x1", "x2", "Dấu", "RHS"])

# ----------------- GIAO DIỆN SIDEBAR -----------------
st.sidebar.header("Cài đặt chung")
method = st.sidebar.radio(
    "CHỌN PHƯƠNG PHÁP GIẢI:",
    ("1. Scipy (Tổng quát, nhanh)", "2. Đồ thị (Chỉ 2 biến)", "3. Từ vựng (Đơn hình Dantzig)", "4. Từ vựng (Đơn hình Bland)", "5. Chạy tất cả (So sánh)")
)
st.sidebar.markdown("---")

# =========================================================================
# VŨ KHÍ 4: NHẬN DIỆN ẢNH BẰNG AI (GOOGLE GEMINI)
# =========================================================================
st.sidebar.subheader("📸 Quét ảnh bằng AI")
uploaded_file = st.sidebar.file_uploader("Tải ảnh bài toán (viết tay/chụp)", type=["jpg", "png", "jpeg"])

# 🔒 TỰ ĐỘNG LẤY API KEY TỪ HỆ THỐNG BÍ MẬT CỦA STREAMLIT
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    api_key = None
    st.sidebar.error("⚠️ App chưa được cấu hình API Key. Hãy khai báo trong mục Settings > Secrets!")

if st.sidebar.button("🧠 Quét & Tự động điền"):
    if uploaded_file is not None and api_key:
        try:
            import google.generativeai as genai
            from PIL import Image
            import json

            genai.configure(api_key=api_key)
            image = Image.open(uploaded_file)
            
            prompt = """
            Bạn là chuyên gia Toán học. Hãy đọc bài toán Quy hoạch tuyến tính trong ảnh.
            Trả về DUY NHẤT một JSON hợp lệ (không có markdown, không có text khác) với cấu trúc:
            {
                "opt_type": "MAX" hoặc "MIN",
                "n_vars": số nguyên (số lượng biến),
                "n_cons": số nguyên (số lượng ràng buộc),
                "obj": [mảng số thực chứa các hệ số hàm mục tiêu],
                "cons": [
                    {"coeffs": [mảng hệ số], "sign": "<=" hoặc ">=" hoặc "=", "rhs": số vế phải},
                    ...
                ]
            }
            """
            
            with st.spinner("🤖 AI đang giải mã chữ viết tay của bạn..."):
                response = None
                last_error = None
                model_names_to_try = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-1.5-flash-latest']
                
                for m_name in model_names_to_try:
                    try:
                        model = genai.GenerativeModel(m_name)
                        response = model.generate_content([prompt, image])
                        break 
                    except Exception as e:
                        last_error = str(e)
                        continue 
                
                if response is None:
                    # 🔴 IN TRỰC TIẾP LỖI GỐC CỦA GOOGLE RA MÀN HÌNH WEB ĐỂ CHẨN ĐOÁN
                    st.error(f"🛑 BÁO CÁO LỖI TỪ GOOGLE: {last_error}")
                    st.info("💡 Bạn hãy copy hoặc chụp màn hình dòng chữ tiếng Anh màu đỏ ở trên gửi lại đây nhé!")
                    st.stop() 
                
                raw_text = response.text.strip()
                if raw_text.startswith("
http://googleusercontent.com/immersive_entry_chip/0
http://googleusercontent.com/immersive_entry_chip/1

### 🎯 Các bước tiếp theo bạn cần làm:
1. Dán đống này vào file `app.py` trên GitHub của bạn rồi bấm nút **Commit**.
2. Sang trang quản lý Streamlit Cloud bấm **Reboot app** để nó ăn mã mới.
3. Ấn nút **Quét ảnh** trên trang web. Lúc này, nó sẽ hiện một dòng chữ tiếng Anh màu đỏ ngay trên màn hình. Bạn hãy chép dòng đó gửi lại đây cho mình nhé!
