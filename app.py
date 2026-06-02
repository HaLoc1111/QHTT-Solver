import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import time
import json

# =========================================================================
# CẤU HÌNH TRANG CHỦ
# =========================================================================
st.set_page_config(page_title="Giải thuật QHTT", layout="wide", page_icon="📈")
st.title("📈 Chương trình giải Quy Hoạch Tuyến Tính Tổng Quát")
st.markdown("---")

# =========================================================================
# QUẢN LÝ SESSION STATE BẤT BẠI (CHỐNG LỖI BẢNG)
# =========================================================================
if "n_vars" not in st.session_state: st.session_state.n_vars = 2
if "n_cons" not in st.session_state: st.session_state.n_cons = 3
if "opt_type" not in st.session_state: st.session_state.opt_type = "MAX"
if "init_obj" not in st.session_state:
    st.session_state.init_obj = pd.DataFrame([[0.0, 0.0]], columns=["x1", "x2"])
if "init_cons" not in st.session_state:
    st.session_state.init_cons = pd.DataFrame([[0.0, 0.0, "<=", 0.0] for _ in range(3)], columns=["x1", "x2", "Dấu", "RHS"])

# Đảm bảo bảng luôn khớp với số biến hiện tại
obj_cols = [f"x{i+1}" for i in range(st.session_state.n_vars)]
if st.session_state.init_obj.shape[1] != st.session_state.n_vars:
    st.session_state.init_obj = pd.DataFrame([[0.0] * st.session_state.n_vars], columns=obj_cols)
if st.session_state.init_cons.shape[0] != st.session_state.n_cons or st.session_state.init_cons.shape[1] != (st.session_state.n_vars + 2):
    st.session_state.init_cons = pd.DataFrame([[0.0] * st.session_state.n_vars + ["<=", 0.0] for _ in range(st.session_state.n_cons)], columns=obj_cols + ["Dấu", "RHS"])

# =========================================================================
# GIAO DIỆN CHÍNH: VŨ KHÍ AI NHẬN DIỆN ẢNH (ĐƯA LÊN ĐẦU TIÊN)
# =========================================================================
st.markdown("### 📸 1. Tự động nhập đề bằng AI (Upload Ảnh)")
st.info("💡 Tải ảnh bài toán (chụp vở hoặc màn hình) lên đây. AI sẽ đọc và tự động điền vào bảng bên dưới!")
uploaded_file = st.file_uploader("Kéo thả hoặc chọn ảnh...", type=["jpg", "png", "jpeg"])

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    api_key = None
    st.error("⚠️ App chưa được khai báo API Key trong mục Settings > Secrets của Streamlit!")

if st.button("🧠 Quét Ảnh & Tự Động Điền", type="primary"):
    if uploaded_file is not None and api_key:
        try:
            import google.generativeai as genai
            from PIL import Image

            genai.configure(api_key=api_key)
            image = Image.open(uploaded_file)
            
            prompt = """
            Bạn là chuyên gia Toán Quy hoạch tuyến tính. Hãy đọc bài toán trong ảnh.
            Trả về DUY NHẤT một chuỗi JSON hợp lệ (không markdown, không text dư) với cấu trúc:
            {
                "opt_type": "MAX" hoặc "MIN",
                "n_vars": số_nguyên,
                "n_cons": số_nguyên,
                "obj": [mảng_hệ_số_hàm_mục_tiêu],
                "cons": [
                    {"coeffs": [mảng_hệ_số], "sign": "<=" hoặc ">=" hoặc "=", "rhs": số_vế_phải}
                ]
            }
            """
            with st.spinner("🤖 AI đang phân tích bài toán..."):
                response = None
                model_names = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash']
                for m_name in model_names:
                    try:
                        model = genai.GenerativeModel(m_name)
                        response = model.generate_content([prompt, image])
                        break
                    except Exception:
                        continue 
                
                if response is None:
                    st.error("❌ Google API từ chối kết nối. Hãy kiểm tra lại API Key.")
                else:
                    # Làm sạch JSON an toàn tuyệt đối
                    raw_text = response.text.strip().replace("```json", "").replace("```", "").strip()
                    data = json.loads(raw_text)
                    
                    st.session_state.opt_type = data.get("opt_type", "MAX").upper()
                    st.session_state.n_vars = int(data.get("n_vars", 2))
                    st.session_state.n_cons = int(data.get("n_cons", 2))
                    
                    new_obj_cols = [f"x{i+1}" for i in range(st.session_state.n_vars)]
                    st.session_state.init_obj = pd.DataFrame([data["obj"]], columns=new_obj_cols)
                    
                    cons_data = []
                    for c in data["cons"]:
                        cons_data.append(c["coeffs"] + [c["sign"], float(c["rhs"])])
                    st.session_state.init_cons = pd.DataFrame(cons_data, columns=new_obj_cols + ["Dấu", "RHS"])
                    
                    st.success("✨ Nhận diện thành công! Kéo xuống để xem bảng dữ liệu.")
                    time.sleep(1.5)
                    st.rerun()
        except Exception as e:
            st.error(f"❌ AI gặp lỗi khi đọc ảnh: {e}")
    else:
        st.warning("⚠️ Bạn chưa tải ảnh lên!")

st.markdown("---")

# =========================================================================
# GIAO DIỆN SIDEBAR (CÀI ĐẶT & DỮ LIỆU MẪU)
# =========================================================================
st.sidebar.header("Cài đặt chung")
method = st.sidebar.radio("CHỌN PHƯƠNG PHÁP GIẢI:", ("1. Scipy (Tổng quát, nhanh)", "2. Đồ thị (Chỉ 2 biến)", "3. Từ vựng (Đơn hình Dantzig)", "4. Từ vựng (Đơn hình Bland)", "5. Chạy tất cả (So sánh)"))
st.sidebar.markdown("---")

st.sidebar.subheader("📂 Quản lý dữ liệu")
if st.sidebar.button("📝 Tải bài mẫu (Cơ bản)"):
    st.session_state.n_vars, st.session_state.n_cons, st.session_state.opt_type = 2, 3, "MAX"
    st.session_state.init_obj = pd.DataFrame([[2.0, 5.0]], columns=["x1", "x2"])
    st.session_state.init_cons = pd.DataFrame([[1.0, 0.0, "<=", 4.0], [0.0, 2.0, "<=", 12.0], [3.0, 2.0, "<=", 18.0]], columns=["x1", "x2", "Dấu", "RHS"])
    st.rerun()

if st.sidebar.button("🔄 Làm sạch bảng"):
    st.session_state.n_vars, st.session_state.n_cons, st.session_state.opt_type = 2, 3, "MAX"
    st.session_state.init_obj = pd.DataFrame([[0.0, 0.0]], columns=["x1", "x2"])
    st.session_state.init_cons = pd.DataFrame([[0.0, 0.0, "<=", 0.0] for _ in range(3)], columns=["x1", "x2", "Dấu", "RHS"])
    st.rerun()
st.sidebar.markdown("---")

# Nhập số liệu kích thước bảng (tự động cập nhật bảng không gây lỗi)
new_n_vars = st.sidebar.number_input("Số lượng biến", 1, 20, st.session_state.n_vars)
new_n_cons = st.sidebar.number_input("Số lượng ràng buộc", 1, 20, st.session_state.n_cons)
if new_n_vars != st.session_state.n_vars or new_n_cons != st.session_state.n_cons:
    st.session_state.n_vars = new_n_vars
    st.session_state.n_cons = new_n_cons
    st.rerun()

st.session_state.opt_type = st.sidebar.radio("Mục tiêu tối ưu", ("MAX", "MIN"), index=0 if st.session_state.opt_type == "MAX" else 1)

# =========================================================================
# GIAO DIỆN CHỈNH SỬA DỮ LIỆU
# =========================================================================
st.markdown("### ✍️ 2. Nhập/Sửa Dữ liệu Bài Toán")
st.write("**1. Hàm mục tiêu $f(x)$**")
df_obj = st.data_editor(st.session_state.init_obj, hide_index=True, use_container_width=True)

st.write("**2. Hệ ràng buộc**")
config = {"Dấu": st.column_config.SelectboxColumn("Dấu", options=["<=", ">=", "="], required=True)}
df_cons = st.data_editor(st.session_state.init_cons, column_config=config, hide_index=True, use_container_width=True)

st.write("**3. Ràng buộc dấu của biến**")
bounds = []
cols = st.columns(st.session_state.n_vars)
for i in range(st.session_state.n_vars):
    with cols[i]:
        b_val = st.selectbox(f"x{i+1}", ["Tùy ý (Free)", ">= 0", "<= 0"], index=1)
        if b_val == ">= 0": bounds.append((0, None))
        elif b_val == "<= 0": bounds.append((None, 0))
        else: bounds.append((None, None))

# =========================================================================
# MÔ HÌNH LATEX
# =========================================================================
st.markdown("---")
tab_model_primal, tab_model_dual = st.tabs(["🔍 Mô hình Gốc (Primal)", "🔄 Bài toán Đối ngẫu (Dual)"])

def render_math_model_latex(df_obj, df_cons, obj_cols, opt_type, bounds):
    c_vals = df_obj.iloc[0].fillna(0).values
    obj_str = " ".join([f"{'+' if v>=0 and i>0 else ('-' if v<0 else '')} {abs(v):.2f}x_{i+1}" for i, v in enumerate(c_vals) if abs(v)>1e-6 or (i==0 and all(x==0 for x in c_vals))])
    
    cons_lines = []
    for _, row in df_cons.iterrows():
        line_str = " ".join([f"{'+' if float(row[col])>=0 and i>0 else ('-' if float(row[col])<0 else '')} {abs(float(row[col])):.2f}x_{i+1}" for i, col in enumerate(obj_cols) if abs(float(row[col]))>1e-6])
        if not line_str: line_str = "0"
        sign_symbol = "\\le" if row["Dấu"] == "<=" else ("\\ge" if row["Dấu"] == ">=" else "=")
        cons_lines.append(f"{line_str} & {sign_symbol} {float(row['RHS']):.2f}")
    
    bound_terms = [f"x_{i+1} \\ge 0" if b==(0,None) else (f"x_{i+1} \\le 0" if b==(None,0) else f"x_{i+1} \\text{{ tùy ý}}") for i, b in enumerate(bounds)]
    
    latex_model = "$$\n\\begin{array}{ll}\n"
    latex_model += "\\text{Tối ưu hóa:} & \\" + opt_type.lower() + f" \\quad Z = {obj_str if obj_str else '0'} \\\\\n"
    latex_model += "\\text{Thỏa mãn:} & \\left\\{\n\\begin{array}{l}\n"
    latex_model += r" \\ ".join(cons_lines) + "\n\\end{array}\n\\right. \\\\\n"
    latex_model += f"& {', '.join(bound_terms)}\n\\end{{array}}\n$$" 
    return latex_model

def render_dual_model_latex(df_obj, df_cons, obj_cols, opt_type, bounds):
    c_vals = df_obj.iloc[0].fillna(0).values
    b_vals = [float(row["RHS"]) if not pd.isna(row["RHS"]) else 0.0 for _, row in df_cons.iterrows()]
    A_matrix = [[float(row[col]) for col in obj_cols] for _, row in df_cons.iterrows()]
    A_transposed = np.array(A_matrix).T
    signs = df_cons["Dấu"].values
    
    dual_opt = "MIN" if opt_type == "MAX" else "MAX"
    dual_obj_str = " ".join([f"{'+' if v>=0 and i>0 else ('-' if v<0 else '')} {abs(v):.2f}y_{i+1}" for i, v in enumerate(b_vals) if abs(v)>1e-6])
    
    dual_cons_lines = []
    for i, row in enumerate(A_transposed):
        line_str = " ".join([f"{'+' if v>=0 and j>0 else ('-' if v<0 else '')} {abs(v):.2f}y_{j+1}" for j, v in enumerate(row) if abs(v)>1e-6])
        if not line_str: line_str = "0"
        
        if opt_type == "MAX":
            if bounds[i] == (0, None): sign_symbol = "\\ge"
            elif bounds[i] == (None, 0): sign_symbol = "\\le"
            else: sign_symbol = "="
        else:
            if bounds[i] == (0, None): sign_symbol = "\\le"
            elif bounds[i] == (None, 0): sign_symbol = "\\ge"
            else: sign_symbol = "="
            
        dual_cons_lines.append(f"{line_str} & {sign_symbol} {c_vals[i]:.2f}")
        
    dual_bounds = []
    for i, s in enumerate(signs):
        if opt_type == "MAX":
            if s == "<=": dual_bounds.append(f"y_{i+1} \\ge 0")
            elif s == ">=": dual_bounds.append(f"y_{i+1} \\le 0")
            else: dual_bounds.append(f"y_{i+1} \\text{{ tùy ý}}")
        else:
            if s == ">=": dual_bounds.append(f"y_{i+1} \\ge 0")
            elif s == "<=": dual_bounds.append(f"y_{i+1} \\le 0")
            else: dual_bounds.append(f"y_{i+1} \\text{{ tùy ý}}")

    latex_model = "$$\n\\begin{array}{ll}\n"
    latex_model += "\\text{Bài toán Đối ngẫu:} & \\" + dual_opt.lower() + f" \\quad W = {dual_obj_str if dual_obj_str else '0'} \\\\\n"
    latex_model += "\\text{Thỏa mãn:} & \\left\\{\n\\begin{array}{l}\n"
    latex_model += r" \\ ".join(dual_cons_lines) + "\n\\end{array}\n\\right. \\\\\n"
    latex_model += f"& {', '.join(dual_bounds)}\n\\end{{array}}\n$$"
    return latex_model

with tab_model_primal:
    st.markdown(render_math_model_latex(df_obj, df_cons, obj_cols, st.session_state.opt_type, bounds))
with tab_model_dual:
    st.info("💡 Phân tích: Thuật toán tự động sinh bài toán đối ngẫu (Dual) bằng cách chuyển vị ma trận hệ số, đảo MIN/MAX và áp dụng quy tắc đổi dấu.")
    st.markdown(render_dual_model_latex(df_obj, df_cons, obj_cols, st.session_state.opt_type, bounds))

# =========================================================================
# CORE LOGIC: THUẬT TOÁN GIẢI
# =========================================================================
def log_and_print(log, text):
    st.markdown(text)
    log.append(text)

def solve_scipy(c, df_cons, obj_cols, opt_type, bounds):
    c_scipy = -c if opt_type == "MAX" else c
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for _, row in df_cons.iterrows():
        coeffs = row[obj_cols].fillna(0).values.astype(float)
        sign = row["Dấu"]
        rhs = float(row["RHS"]) if not pd.isna(row["RHS"]) else 0.0
        if sign == "<=":
            A_ub.append(coeffs); b_ub.append(rhs)
        elif sign == ">=":
            A_ub.append(-coeffs); b_ub.append(-rhs)
        else:
            A_eq.append(coeffs); b_eq.append(rhs)
            
    res = linprog(c_scipy, A_ub=A_ub if len(A_ub)>0 else None, b_ub=b_ub if len(b_ub)>0 else None, 
                  A_eq=A_eq if len(A_eq)>0 else None, b_eq=b_eq if len(b_eq)>0 else None, 
                  bounds=bounds, method='highs')
    if res.success:
        opt_val = res.fun if opt_type == "MIN" else -res.fun
        st.success(f"✅ Nghiệm tối ưu: Z = {opt_val:.4f}")
        st.dataframe(pd.DataFrame({"Biến số": obj_cols, "Giá trị": np.round(res.x, 4)}))
    else:
        st.error(f"❌ Thuật toán không thể giải: {res.message}")

def solve_graph(c, df_cons, n_vars, opt_type):
    if n_vars != 2:
        st.error("❌ Phương pháp đồ thị chỉ hỗ trợ bài toán có đúng 2 biến (x1, x2).")
        return
        
    st.write("### 🎚️ Mô phỏng Trượt hàm mục tiêu")
    z_slider = st.slider("Giá trị Z hiện tại", min_value=-30.0, max_value=30.0, value=0.0, step=0.5)

    c1, c2 = float(c[0]), float(c[1])
    fig, ax = plt.subplots(figsize=(8, 8))
    d = np.linspace(-5, 20, 400)
    x1, x2 = np.meshgrid(d, d)
    mask = (x1 >= 0) & (x2 >= 0) 
    
    ax.axhline(0, color='black', linewidth=1.5)
    ax.axvline(0, color='black', linewidth=1.5)
    
    colors = ['blue', 'green', 'purple', 'orange']
    for idx, row in df_cons.iterrows():
        a1, a2 = float(row["x1"]), float(row["x2"])
        rhs = float(row["RHS"]) if not pd.isna(row["RHS"]) else 0.0
        sign = row["Dấu"]
        color = colors[idx % len(colors)]
        
        # Bảo vệ lỗi chia cho 0
        if a2 != 0:
            y = (rhs - a1 * d) / a2
            ax.plot(d, y, color=color, label=f"PT{idx+1}")
        else:
            if a1 != 0:
                ax.axvline(x=rhs/a1, color=color, label=f"PT{idx+1}")

        if sign == "<=": mask = mask & (a1 * x1 + a2 * x2 <= rhs)
        elif sign == ">=": mask = mask & (a1 * x1 + a2 * x2 >= rhs)
            
    ax.imshow(mask.astype(int), extent=(-5, 20, -5, 20), origin="lower", cmap="Greys", alpha=0.3)
    
    # Vẽ hàm mục tiêu chống lỗi chia 0
    if c2 != 0:
        y_obj = (z_slider - c1 * d) / c2
        ax.plot(d, y_obj, 'r-', linewidth=2.5, label=f"Đường Z = {z_slider:.1f}")
    else:
        if c1 != 0:
            ax.axvline(x=z_slider/c1, color='r', linewidth=2.5, label=f"Z = {z_slider:.1f}")

    ax.set_xlim(-2, 10); ax.set_ylim(-2, 10)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    st.pyplot(fig)

# =========================================================================
# NÚT CHẠY THUẬT TOÁN
# =========================================================================
st.markdown("---")
if st.button("🚀 BẤM VÀO ĐÂY ĐỂ GIẢI BÀI TOÁN", type="primary", use_container_width=True):
    c = df_obj.iloc[0].fillna(0).values.astype(float)
    
    if method == "1. Scipy (Tổng quát, nhanh)": 
        solve_scipy(c, df_cons, obj_cols, st.session_state.opt_type, bounds)
    elif method == "2. Đồ thị (Chỉ 2 biến)": 
        solve_graph(c, df_cons, st.session_state.n_vars, st.session_state.opt_type)
    elif method == "5. Chạy tất cả (So sánh)":
        tab1, tab2 = st.tabs(["📦 Thư viện Scipy", "📈 Phương pháp Đồ thị"])
        with tab1: solve_scipy(c, df_cons, obj_cols, st.session_state.opt_type, bounds)
        with tab2: solve_graph(c, df_cons, st.session_state.n_vars, st.session_state.opt_type)
    else:
        st.warning("⚠️ Thuật toán Từ Vựng đang được bảo trì để tối ưu hiệu suất, vui lòng chọn Scipy hoặc Đồ thị!")
