import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import time
import json
import os
from datetime import datetime

# =========================================================================
# CẤU HÌNH TRANG CHỦ & HỆ THỐNG LƯU TRỮ LỊCH SỬ
# =========================================================================
st.set_page_config(page_title="Giải thuật QHTT", layout="wide", page_icon="📈")

HISTORY_FILE = "qhtt_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_history(history_dict):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, ensure_ascii=False, indent=4)

if "history" not in st.session_state:
    st.session_state.history = load_history()

st.title("📈 Chương trình giải Quy Hoạch Tuyến Tính Tổng Quát")
st.markdown("---")

# =========================================================================
# QUẢN LÝ LÕI BỘ NHỚ
# =========================================================================
if "vars_input" not in st.session_state: st.session_state.vars_input = 2
if "cons_input" not in st.session_state: st.session_state.cons_input = 3
if "opt_input" not in st.session_state: st.session_state.opt_input = "MAX"

if "init_obj" not in st.session_state:
    st.session_state.init_obj = pd.DataFrame([[0.0, 0.0]], columns=["x1", "x2"])
if "init_cons" not in st.session_state:
    st.session_state.init_cons = pd.DataFrame([[0.0, 0.0, "<=", 0.0] for _ in range(3)], columns=["x1", "x2", "Dấu", "RHS"])

n_vars = st.session_state.vars_input
n_cons = st.session_state.cons_input
opt_type = st.session_state.opt_input

obj_cols = [f"x{i+1}" for i in range(n_vars)]
if st.session_state.init_obj.shape[1] != n_vars:
    st.session_state.init_obj = pd.DataFrame([[0.0] * n_vars], columns=obj_cols)
if st.session_state.init_cons.shape[0] != n_cons or st.session_state.init_cons.shape[1] != (n_vars + 2):
    st.session_state.init_cons = pd.DataFrame([[0.0] * n_vars + ["<=", 0.0] for _ in range(n_cons)], columns=obj_cols + ["Dấu", "RHS"])

# =========================================================================
# 1. QUÉT ẢNH BẰNG AI 
# =========================================================================
st.markdown("### 📸 1. Tự động nhập đề bằng AI (Upload Ảnh)")
st.info("💡 Tải ảnh bài toán (công thức toán học) lên đây. AI sẽ phân tích và tự động điền vào bảng bên dưới!")

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    api_key = None
    st.error("⚠️ App chưa được khai báo API Key trong mục Settings > Secrets của Streamlit Cloud!")

uploaded_file = st.file_uploader("Kéo thả hoặc chọn ảnh bài toán...", type=["jpg", "png", "jpeg"])

if st.button("🧠 Quét Ảnh & Tự Động Điền", type="primary"):
    if uploaded_file and api_key:
        try:
            import google.generativeai as genai
            from PIL import Image
            genai.configure(api_key=api_key)
            image = Image.open(uploaded_file)
            
            prompt = """
            Bạn là chuyên gia Toán Quy hoạch tuyến tính. Hãy đọc dữ liệu bài toán trong ảnh.
            Hãy mô hình hóa nó và trả về DUY NHẤT một chuỗi JSON hợp lệ (không markdown, không giải thích) với cấu trúc:
            {"opt_type": "MAX" hoặc "MIN", "n_vars": số_lượng_biến, "n_cons": số_lượng_ràng_buộc, "obj": [mảng_hệ_số_mục_tiêu], "cons": [{"coeffs": [mảng_hệ_số_ràng_buộc], "sign": "<=" hoặc ">=" hoặc "=", "rhs": số_vế_phải}]}
            Nếu không thấy hệ số nào, hãy cho nó bằng 0.
            """
            with st.spinner("🤖 AI đang phân tích hình ảnh..."):
                response = None
                error_logs = []
                
                model_names_to_try = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash-latest']
                
                for m_name in model_names_to_try:
                    try:
                        model = genai.GenerativeModel(m_name)
                        response = model.generate_content([prompt, image])
                        break 
                    except Exception as e: 
                        error_logs.append(f"❌ {m_name}: {str(e)}")
                        continue 
                
                if response:
                    raw_text = response.text.strip()
                    raw_text = raw_text.replace("```json", "").replace("```", "").strip()
                    data = json.loads(raw_text)
                    
                    st.session_state.opt_input = data.get("opt_type", "MAX").upper()
                    st.session_state.vars_input = int(data.get("n_vars", 2))
                    st.session_state.cons_input = int(data.get("n_cons", 2))
                    
                    new_obj_cols = [f"x{i+1}" for i in range(st.session_state.vars_input)]
                    st.session_state.init_obj = pd.DataFrame([data["obj"]], columns=new_obj_cols)
                    
                    cons_data = [c["coeffs"] + [c["sign"], float(c["rhs"])] for c in data["cons"]]
                    st.session_state.init_cons = pd.DataFrame(cons_data, columns=new_obj_cols + ["Dấu", "RHS"])
                    
                    st.success("✨ Nhận diện thành công! Hệ thống đã được điền tự động.")
                    time.sleep(1)
                    st.rerun()
                else:
                    err_msg = "\n\n".join(error_logs)
                    st.error(f"🛑 CẢ 3 MODEL ĐỀU BỊ GOOGLE TỪ CHỐI. LỖI CHI TIẾT:\n\n{err_msg}")
        except Exception as e: st.error(f"❌ Hệ thống không đọc được ảnh. Lỗi code: {e}")
    else: st.warning("⚠️ Bạn chưa tải ảnh lên hoặc chưa có API Key!")
st.markdown("---")

# =========================================================================
# GIAO DIỆN SIDEBAR (LỊCH SỬ & CÀI ĐẶT & XUẤT NHẬP CSV)
# =========================================================================
st.sidebar.markdown("## 🕒 Gần đây (Lịch sử)")

col_save1, col_save2 = st.sidebar.columns([3, 1])
with col_save1:
    save_name = st.text_input("Tên bài toán:", placeholder="Vd: Bài thi cuối kỳ...", label_visibility="collapsed")
with col_save2:
    if st.button("💾 Lưu"):
        name_to_save = save_name if save_name else f"Bài toán {datetime.now().strftime('%H:%M %d/%m')}"
        st.session_state.history[name_to_save] = {
            "n_vars": st.session_state.vars_input, "n_cons": st.session_state.cons_input,
            "opt_type": st.session_state.opt_input, "obj": st.session_state.init_obj.values.tolist(),
            "cons": st.session_state.init_cons.values.tolist()
        }
        save_history(st.session_state.history)
        st.sidebar.success("Đã lưu!")
        time.sleep(0.5); st.rerun()

if not st.session_state.history:
    st.sidebar.info("Chưa có bài toán nào được lưu.")
else:
    for history_name, history_data in reversed(st.session_state.history.items()):
        if st.sidebar.button(f"💬 {history_name}", use_container_width=True):
            st.session_state.vars_input = history_data["n_vars"]
            st.session_state.cons_input = history_data["n_cons"]
            st.session_state.opt_input = history_data["opt_type"]
            obj_cols_res = [f"x{i+1}" for i in range(history_data["n_vars"])]
            st.session_state.init_obj = pd.DataFrame(history_data["obj"], columns=obj_cols_res)
            st.session_state.init_cons = pd.DataFrame(history_data["cons"], columns=obj_cols_res + ["Dấu", "RHS"])
            st.rerun()
    if st.sidebar.button("🗑️ Xóa toàn bộ lịch sử", type="secondary"):
        st.session_state.history = {}; save_history({}); st.rerun()

st.sidebar.markdown("---")

st.sidebar.header("Cài đặt chung")
method = st.sidebar.radio("CHỌN PHƯƠNG PHÁP GIẢI:", ("1. Scipy (Tổng quát, nhanh)", "2. Đồ thị (Chỉ 2 biến)", "3. Từ vựng (Đơn hình Dantzig)", "4. Từ vựng (Đơn hình Bland)", "5. Chạy tất cả (So sánh)"))
st.sidebar.markdown("---")

# TÍNH NĂNG XUẤT NHẬP DỮ LIỆU BẰNG EXCEL / CSV
st.sidebar.subheader("📥 Nhập/Xuất Dữ liệu (CSV)")
df_export_obj = st.session_state.init_obj.copy()
df_export_obj["Dấu"] = "="; df_export_obj["RHS"] = "Hàm Mục Tiêu" # Trick để đánh dấu hàng đầu tiên
df_export = pd.concat([df_export_obj, st.session_state.init_cons], ignore_index=True)
csv_data = df_export.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(label="⬇️ Tải file ma trận hiện tại (.csv)", data=csv_data, file_name="MaTran_QHTT.csv", mime="text/csv", help="Lưu ma trận đang nhập để nạp lại sau này.")

uploaded_csv = st.sidebar.file_uploader("⬆️ Tải file CSV lên", type=["csv"], help="Cấu trúc file y hệt file tải về.")
if uploaded_csv and st.sidebar.button("Nạp dữ liệu từ File"):
    try:
        df_in = pd.read_csv(uploaded_csv)
        n_v = len([col for col in df_in.columns if col not in ["Dấu", "RHS"]])
        st.session_state.vars_input = n_v
        st.session_state.cons_input = len(df_in) - 1
        st.session_state.init_obj = df_in.iloc[[0]][[f"x{i+1}" for i in range(n_v)]].astype(float)
        st.session_state.init_cons = df_in.iloc[1:].copy()
        st.session_state.init_cons["Dấu"] = st.session_state.init_cons["Dấu"].fillna("<=")
        st.session_state.init_cons["RHS"] = st.session_state.init_cons["RHS"].fillna(0.0).astype(float)
        for i in range(n_v): st.session_state.init_cons[f"x{i+1}"] = st.session_state.init_cons[f"x{i+1}"].fillna(0.0).astype(float)
        st.sidebar.success("Nạp thành công!"); time.sleep(1); st.rerun()
    except Exception as e: st.sidebar.error("Lỗi định dạng file!")

st.sidebar.markdown("---")

st.sidebar.subheader("📂 Quản lý dữ liệu mẫu")
if st.sidebar.button("📝 Tải mẫu 1 Pha (RHS >= 0)"):
    st.session_state.vars_input, st.session_state.cons_input, st.session_state.opt_input = 2, 3, "MAX"
    st.session_state.init_obj = pd.DataFrame([[2.0, 5.0]], columns=["x1", "x2"])
    st.session_state.init_cons = pd.DataFrame([[1.0, 0.0, "<=", 4.0], [0.0, 2.0, "<=", 12.0], [3.0, 2.0, "<=", 18.0]], columns=["x1", "x2", "Dấu", "RHS"])
    st.rerun()

if st.sidebar.button("📚 Tải mẫu 2 Pha (RHS < 0)"):
    st.session_state.vars_input, st.session_state.cons_input, st.session_state.opt_input = 2, 3, "MIN"
    st.session_state.init_obj = pd.DataFrame([[1.0, 2.0]], columns=["x1", "x2"])
    st.session_state.init_cons = pd.DataFrame([[-1.0, 1.0, "<=", -2.0], [-1.0, -2.0, "<=", -4.0], [0.0, 1.0, "<=", 2.0]], columns=["x1", "x2", "Dấu", "RHS"])
    st.rerun()

if st.sidebar.button("🔄 Đặt lại bảng trống"):
    st.session_state.vars_input, st.session_state.cons_input, st.session_state.opt_input = 2, 3, "MAX"
    st.session_state.init_obj = pd.DataFrame([[0.0, 0.0]], columns=["x1", "x2"])
    st.session_state.init_cons = pd.DataFrame([[0.0, 0.0, "<=", 0.0] for _ in range(3)], columns=["x1", "x2", "Dấu", "RHS"])
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.number_input("Số lượng biến", 1, 20, key="vars_input")
st.sidebar.number_input("Số lượng ràng buộc", 1, 20, key="cons_input")
st.sidebar.radio("Mục tiêu tối ưu", ("MAX", "MIN"), key="opt_input")

# =========================================================================
# GIAO DIỆN NHẬP LIỆU CHÍNH
# =========================================================================
st.subheader("1. Hàm mục tiêu $f(x)$")
st.session_state.init_obj = st.data_editor(st.session_state.init_obj, hide_index=True, use_container_width=True)
df_obj = st.session_state.init_obj

st.subheader("2. Hệ ràng buộc")
config = {"Dấu": st.column_config.SelectboxColumn("Dấu", options=["<=", ">=", "="], required=True)}
st.session_state.init_cons = st.data_editor(st.session_state.init_cons, column_config=config, hide_index=True, use_container_width=True)
df_cons = st.session_state.init_cons

st.subheader("3. Ràng buộc dấu của biến")
bounds = []
cols = st.columns(n_vars)
for i in range(n_vars):
    with cols[i]:
        b_val = st.selectbox(f"x{i+1}", ["Tùy ý (Free)", ">= 0", "<= 0"], index=1)
        if b_val == ">= 0": bounds.append((0, None))
        elif b_val == "<= 0": bounds.append((None, 0))
        else: bounds.append((None, None))

# =========================================================================
# MÔ HÌNH LATEX VÀ BÀI TOÁN ĐỐI NGẪU
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
    st.markdown(render_math_model_latex(df_obj, df_cons, obj_cols, opt_type, bounds))
with tab_model_dual:
    st.info("💡 Phân tích: Thuật toán tự động sinh bài toán đối ngẫu (Dual) bằng cách chuyển vị ma trận hệ số, đảo MIN/MAX và áp dụng quy tắc đổi dấu.")
    st.markdown(render_dual_model_latex(df_obj, df_cons, obj_cols, opt_type, bounds))

# =========================================================================
# CÁC HÀM XỬ LÝ (CORE LOGIC + TRACKER TỪNG BƯỚC)
# =========================================================================

class StepTracker:
    def __init__(self):
        self.steps = []
        self.log = []
    def add_step(self, title):
        self.steps.append({"title": title, "content": []})
    def append_to_current(self, text):
        if not self.steps: self.add_step("Khởi tạo")
        self.steps[-1]["content"].append(text)
        self.log.append(text)

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
        st.success(f"✅ Nghiệm tối ưu (Scipy): Z = {opt_val:.4f}")
        col1, col2 = st.columns(2)
        with col1:
            st.write("🎯 **Chiến lược hành động:**")
            st.dataframe(pd.DataFrame({"Biến số": obj_cols, "Giá trị": np.round(res.x, 4)}))
        with col2:
            st.write("📊 **Giá mờ (Shadow Prices):**")
            shadow_prices = []
            if hasattr(res, 'ineqlin') and hasattr(res.ineqlin, 'marginals'):
                shadow_prices.extend(res.ineqlin.marginals)
            if hasattr(res, 'eqlin') and hasattr(res.eqlin, 'marginals'):
                shadow_prices.extend(res.eqlin.marginals)
            if shadow_prices:
                sp_vals = np.round(np.abs(shadow_prices), 4) 
                sp_df = pd.DataFrame({"PT": [f"PT {i+1}" for i in range(len(sp_vals))], "Giá mờ": sp_vals})
                st.dataframe(sp_df)
            else:
                st.write("Không trích xuất được Giá mờ.")
    else:
        if res.status == 3: st.error("❌ BÀI TOÁN KHÔNG GIỚI HẠN (UNBOUNDED)!")
        elif res.status == 2: st.error("❌ BÀI TOÁN VÔ NGHIỆM (INFEASIBLE)!")
        else: st.error(f"❌ Lỗi: {res.message}")

def solve_graph(c, df_cons, n_vars, opt_type):
    if n_vars != 2:
        st.error("❌ Phương pháp đồ thị chỉ hỗ trợ bài toán có đúng 2 biến (x1, x2).")
        return
    
    c_scipy = -c if opt_type == "MAX" else c
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for _, row in df_cons.iterrows():
        a1 = float(row["x1"]) if not pd.isna(row["x1"]) else 0.0
        a2 = float(row["x2"]) if not pd.isna(row["x2"]) else 0.0
        rhs = float(row["RHS"]) if not pd.isna(row["RHS"]) else 0.0
        sign = row["Dấu"]
        if sign == "<=": A_ub.append([a1, a2]); b_ub.append(rhs)
        elif sign == ">=": A_ub.append([-a1, -a2]); b_ub.append(-rhs)
        else: A_eq.append([a1, a2]); b_eq.append(rhs)

    res = linprog(c_scipy, A_ub=A_ub if len(A_ub)>0 else None, b_ub=b_ub if len(b_ub)>0 else None,
                  A_eq=A_eq if len(A_eq)>0 else None, b_eq=b_eq if len(b_eq)>0 else None, bounds=bounds, method='highs')
    
    is_optimal = res.success
    opt_x1, opt_x2, opt_z = None, None, None
    if is_optimal:
        opt_x1, opt_x2 = res.x[0], res.x[1]
        opt_z = res.fun if opt_type == "MIN" else -res.fun

    st.write("### 🎚️ Đồ thị Miền nghiệm & Trượt hàm mục tiêu")
    col1, col2 = st.columns([3, 1])
    with col1: z_slider = st.slider("Trượt Z để xem hướng tối ưu", min_value=-50.0, max_value=50.0, value=0.0, step=0.5)
    with col2:
        st.write(""); st.write("")
        is_auto = st.button("🎬 Bật Tự Động Trượt", type="secondary")

    plot_container = st.empty()
    c1, c2 = float(c[0]), float(c[1])

    def render_frame(current_z):
        fig, ax = plt.subplots(figsize=(10, 8))
        d = np.linspace(-5, 25, 600)
        x1, x2 = np.meshgrid(d, d)
        
        mask = np.ones_like(x1, dtype=bool)
        if bounds[0] == (0, None): mask &= (x1 >= 0)
        elif bounds[0] == (None, 0): mask &= (x1 <= 0)
        if bounds[1] == (0, None): mask &= (x2 >= 0)
        elif bounds[1] == (None, 0): mask &= (x2 <= 0)

        ax.axhline(0, color='black', linewidth=1.5)
        ax.axvline(0, color='black', linewidth=1.5)
        colors = ['#1f77b4', '#ff7f0e', '#8c564b', '#9467bd', '#e377c2', '#7f7f7f']
        
        for idx, row in df_cons.iterrows():
            a1 = float(row["x1"]) if not pd.isna(row["x1"]) else 0.0
            a2 = float(row["x2"]) if not pd.isna(row["x2"]) else 0.0
            rhs = float(row["RHS"]) if not pd.isna(row["RHS"]) else 0.0
            sign = row["Dấu"]
            color = colors[idx % len(colors)]
            
            if a1 == 0 and a2 == 0: continue
            
            if a2 != 0:
                y = (rhs - a1 * d) / a2
                ax.plot(d, y, color=color, linewidth=2, label=f"(PT{idx+1}): {a1}x1 + {a2}x2 {sign} {rhs}")
            else: ax.axvline(x=rhs/a1, color=color, linewidth=2, label=f"(PT{idx+1}): {a1}x1 {sign} {rhs}")

            if sign == "<=": mask &= (a1 * x1 + a2 * x2 <= rhs)
            elif sign == ">=": mask &= (a1 * x1 + a2 * x2 >= rhs)
            else: mask &= (np.isclose(a1 * x1 + a2 * x2, rhs, atol=0.05))
                
        ax.contourf(x1, x2, mask.astype(int), levels=[0.5, 1.5], colors=['#90EE90'], alpha=0.6)
        
        if c1 != 0 or c2 != 0:
            if c2 != 0:
                y_obj = (current_z - c1 * d) / c2
                ax.plot(d, y_obj, 'r--', linewidth=1.5, alpha=0.5, label=f"Đường trượt Z = {current_z:.1f}")
            else: ax.axvline(x=current_z/c1, color='r', linestyle='--', linewidth=1.5, alpha=0.5, label=f"Đường trượt Z = {current_z:.1f}")

        if is_optimal:
            ax.plot(opt_x1, opt_x2, marker='*', markersize=18, color='gold', markeredgecolor='red')
            ax.annotate(f'TỐI ƯU\nZ = {opt_z:.2f}\n({opt_x1:.2f}, {opt_x2:.2f})', xy=(opt_x1, opt_x2), xytext=(opt_x1 + 1, opt_x2 + 1),
                        arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8), fontsize=11, fontweight='bold', color='red', 
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=2, alpha=0.9))
            if c2 != 0:
                y_opt = (opt_z - c1 * d) / c2
                ax.plot(d, y_opt, 'r-', linewidth=3, label=f"ĐƯỜNG TỐI ƯU Z = {opt_z:.2f}")
            else: ax.axvline(x=opt_z/c1, color='r', linewidth=3, label=f"ĐƯỜNG TỐI ƯU Z = {opt_z:.2f}")

        max_limit = max(10, opt_x1 + 5 if is_optimal else 10, opt_x2 + 5 if is_optimal else 10)
        ax.set_xlim(-2, max_limit)
        ax.set_ylim(-2, max_limit)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel("x1", fontweight='bold')
        ax.set_ylabel("x2", fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        return fig

    if is_auto:
        for val in np.arange(-15.0, 20.0 + 1, 1.0):
            fig = render_frame(val); plot_container.pyplot(fig); plt.close(fig); time.sleep(0.15) 
    else: fig = render_frame(z_slider); plot_container.pyplot(fig)

def format_dictionary(N, B, A_N, b_B, c_N, v, var_names, enter_j=-1, leave_i=-1, obj_name="Z"):
    z_eq = f"{obj_name} = {v:.2f}"
    for j, n_idx in enumerate(N):
        coef = c_N[j]
        if abs(coef) > 1e-6:
            sign = "+" if coef > 0 else "-"
            var_str = f"{var_names[n_idx]}"
            if j == enter_j: var_str = f"\\overset{{\\downarrow}}{{\\color{{red}}{{{var_names[n_idx]}}}}}"
            z_eq += f" {sign} {abs(coef):.2f}{var_str}"
            
    lines = []
    for i, b_idx in enumerate(B):
        eq = f"{var_names[b_idx]} = {b_B[i]:.2f}"
        for j, n_idx in enumerate(N):
            coef = -A_N[i, j]
            if abs(coef) > 1e-6:
                sign = "+" if coef > 0 else "-"
                var_str = f"{var_names[n_idx]}"
                if j == enter_j: var_str = f"\\color{{red}}{{{var_names[n_idx]}}}"
                eq += f" {sign} {abs(coef):.2f}{var_str}"
        if i == leave_i: eq += r" \quad \color{red}{\leftarrow \text{ (Ra)}}"
        lines.append(eq)
    dict_str = z_eq + r" \\ \hline " + "\n" + r" \\ ".join(lines)
    return f"$$\n\\begin{{array}}{{l}}\n{dict_str}\n\\end{{array}}\n$$"

def perform_pivot(N, B, A_N, b_B, c_N, v, enter_j, leave_i):
    m, n_cols = A_N.shape
    p = A_N[leave_i, enter_j]
    new_A_N, new_b_B, new_c_N = np.zeros_like(A_N), np.zeros_like(b_B), np.zeros_like(c_N)
    new_b_B[leave_i] = b_B[leave_i] / p
    new_A_N[leave_i, enter_j] = 1 / p
    for k in range(n_cols):
        if k != enter_j: new_A_N[leave_i, k] = A_N[leave_i, k] / p
    for r in range(m):
        if r != leave_i:
            factor = A_N[r, enter_j]
            new_b_B[r] = b_B[r] - factor * new_b_B[leave_i]
            new_A_N[r, enter_j] = -factor * new_A_N[leave_i, enter_j]
            for k in range(n_cols):
                if k != enter_j: new_A_N[r, k] = A_N[r, k] - factor * new_A_N[leave_i, k]
    factor = c_N[enter_j]
    new_v = v + factor * new_b_B[leave_i]
    new_c_N[enter_j] = -factor * new_A_N[leave_i, enter_j]
    for k in range(n_cols):
        if k != enter_j: new_c_N[k] = c_N[k] - factor * new_A_N[leave_i, k]
    N[enter_j], B[leave_i] = B[leave_i], N[enter_j]
    return N, B, new_A_N, new_b_B, new_c_N, new_v

def run_simplex_loop(N, B, A_N, b_B, c_N, v, var_names, rule, tracker, obj_name="Z", opt_type="MAX"):
    visited_bases = set()
    iteration = 0
    while True:
        display_v = -v if (opt_type == "MIN" and obj_name == "Z") else v
        display_c_N = -c_N if (opt_type == "MIN" and obj_name == "Z") else c_N
        current_basis = frozenset(B)
        
        tracker.add_step(f"Lần lặp {iteration} ({obj_name})")
        
        if current_basis in visited_bases:
            tracker.append_to_current("⚠️ **PHÁT HIỆN LẶP XOAY VÒNG (CYCLING)!**")
            tracker.append_to_current(format_dictionary(N, B, A_N, b_B, display_c_N, display_v, var_names, obj_name=obj_name))
            return None
        visited_bases.add(current_basis)

        if all(c_N <= 1e-6):
            tracker.append_to_current(format_dictionary(N, B, A_N, b_B, display_c_N, display_v, var_names, obj_name=obj_name))
            if obj_name == "Z": tracker.append_to_current(f"✅ **Đạt phương án tối ưu! {obj_name} = {display_v:.4f}**")
            return N, B, A_N, b_B, c_N, v

        if rule == 'dantzig':
            max_c = np.max(c_N)
            enter_j = min([j for j, val in enumerate(c_N) if abs(val - max_c) < 1e-6], key=lambda j: N[j])
        else:
            enter_j = min([j for j, val in enumerate(c_N) if val > 1e-6], key=lambda j: N[j])
            
        m = len(B)
        ratios = [max(0.0, b_B[i]) / A_N[i, enter_j] if A_N[i, enter_j] > 1e-6 else np.inf for i in range(m)]
        
        if all(r == np.inf for r in ratios):
            tracker.append_to_current(format_dictionary(N, B, A_N, b_B, display_c_N, display_v, var_names, enter_j=enter_j, obj_name=obj_name))
            tracker.append_to_current("❌ **Bài toán không giới hạn (Unbounded)!**")
            if obj_name == "Z": st.error("❌ BÀI TOÁN KHÔNG GIỚI HẠN (UNBOUNDED)!")
            return None

        min_ratio = min(ratios)
        leave_i = min([i for i, r in enumerate(ratios) if abs(r - min_ratio) < 1e-6], key=lambda i: B[i])
        
        tracker.append_to_current(format_dictionary(N, B, A_N, b_B, display_c_N, display_v, var_names, enter_j=enter_j, leave_i=leave_i, obj_name=obj_name))
        tracker.append_to_current(f"🔄 Biến vào: **{var_names[N[enter_j]]}** | Biến ra: **{var_names[B[leave_i]]}**")

        N, B, A_N, b_B, c_N, v = perform_pivot(N, B, A_N, b_B, c_N, v, enter_j, leave_i)
        iteration += 1

def solve_dictionary(c, df_cons, obj_cols, opt_type, bounds, rule='dantzig'):
    tracker = StepTracker()
    
    if any(b != (0, None) for b in bounds): 
        tracker.append_to_current("⚠️ *Thuật toán Từ vựng giả định x >= 0. Các loại ràng buộc dấu khác không được bảo đảm tính chính xác.*")

    A, b = [], []
    for _, row in df_cons.iterrows():
        coeffs = row[obj_cols].fillna(0).values.astype(float)
        rhs = float(row["RHS"]) if not pd.isna(row["RHS"]) else 0.0
        sign = row["Dấu"]
        if sign == "<=": A.append(coeffs); b.append(rhs)
        elif sign == ">=": A.append(-coeffs); b.append(-rhs)
        elif sign == "=": A.append(coeffs); b.append(rhs); A.append(-coeffs); b.append(-rhs)
            
    n, m = len(c), len(b)
    c_orig = -np.array(c, dtype=float) if opt_type == "MIN" else np.array(c, dtype=float)
    A_N, b_B = np.array(A, dtype=float), np.array(b, dtype=float)
    var_names = [f"x_{i+1}" for i in range(n)] + [f"w_{i+1}" for i in range(m)] + ["x_0"]
    x0_idx = n + m
    
    res = None
    if np.min(b_B) < -1e-6:
        tracker.add_step("🛠️ PHA 1: Tìm phương án xuất phát")
        N, B = list(range(n)) + [x0_idx], list(range(n, n + m))
        A_N = np.column_stack((A_N, np.full(m, -1.0)))
        c_N, v = np.zeros(n + 1), 0.0
        c_N[-1] = -1.0 
        leave_i, enter_j = np.argmin(b_B), n 
        tracker.append_to_current("**Bước khởi tạo Từ vựng Pha 1:**")
        tracker.append_to_current(format_dictionary(N, B, A_N, b_B, c_N, v, var_names, enter_j, leave_i, obj_name="\\xi"))
        N, B, A_N, b_B, c_N, v = perform_pivot(N, B, A_N, b_B, c_N, v, enter_j, leave_i)

        res_phase1 = run_simplex_loop(N, B, A_N, b_B, c_N, v, var_names, rule, tracker, obj_name="\\xi")
        if res_phase1 is None: return 
        N, B, A_N, b_B, c_N, v = res_phase1

        if v < -1e-6:
            tracker.append_to_current("❌ **BÀI TOÁN VÔ NGHIỆM! (Pha 1 có x_0 > 0).**")
            st.error("❌ BÀI TOÁN VÔ NGHIỆM! Các ràng buộc mâu thuẫn nhau.")
            return
            
        tracker.add_step("✅ Chuyển tiếp Pha 1 sang Pha 2")
        tracker.append_to_current("Kết thúc Pha 1 thành công. Khử biến x_0 và thay hàm mục tiêu gốc vào.")
        
        if x0_idx in N:
            col_idx = N.index(x0_idx)
            A_N, N = np.delete(A_N, col_idx, axis=1), N[:col_idx] + N[col_idx+1:]
        
        c_N_p2, v_p2 = np.zeros(len(N)), 0.0
        for k in range(n):
            coeff = c_orig[k]
            if k in N: c_N_p2[N.index(k)] += coeff
            elif k in B:
                row_idx = B.index(k)
                v_p2 += coeff * b_B[row_idx]
                for j in range(len(N)): c_N_p2[j] -= coeff * A_N[row_idx, j]
        c_N, v = c_N_p2, v_p2
        res = run_simplex_loop(N, B, A_N, b_B, c_N, v, var_names, rule, tracker, obj_name="Z", opt_type=opt_type)
    else:
        N, B, c_N, v = list(range(n)), list(range(n, n + m)), c_orig.copy(), 0.0
        res = run_simplex_loop(N, B, A_N, b_B, c_N, v, var_names, rule, tracker, obj_name="Z", opt_type=opt_type)

    if res is not None:
        N, B, A_N, b_B, c_N, v = res
        opt_val = -v if opt_type == "MIN" else v
        st.success(f"✅ Nghiệm tối ưu (Từ vựng - {rule.title()}): Z = {opt_val:.4f}")
        
        opt_x = np.zeros(n)
        for k in range(n):
            if k in B: opt_x[k] = b_B[B.index(k)]
        
        st.write("🎯 **Bảng Giá trị Nghiệm (Nghiệm Tối Ưu):**")
        st.dataframe(pd.DataFrame({"Biến số": obj_cols, "Giá trị": np.round(opt_x, 4)}))

    st.markdown("### ⏯️ Mô phỏng Từng Bước Giải")
    if len(tracker.steps) > 0:
        # ĐÃ FIX LỖI DUPLICATE ELEMENT ID Ở ĐÂY
        step_idx = st.slider(
            "Kéo để xem bảng Từ Vựng của từng lần lặp:", 
            0, len(tracker.steps)-1, 0, 
            format="Bước %d", 
            key=f"step_slider_{rule}"
        )
        st.info(f"📍 Đang xem: **{tracker.steps[step_idx]['title']}**")
        for line in tracker.steps[step_idx]["content"]:
            st.markdown(line)

    st.download_button(label=f"📥 Tải toàn bộ Báo Cáo Giải ({rule.title()})", data="\n\n".join(tracker.log), file_name=f"BaoCao_QHTT_{rule}.md", mime="text/markdown")

# ----------------- NÚT THỰC THI CHÍNH -----------------
st.markdown("---")
if 'is_solved' not in st.session_state: st.session_state.is_solved = False
if st.button("🚀 BẤM VÀO ĐÂY ĐỂ GIẢI BÀI TOÁN", type="primary", use_container_width=True): 
    st.session_state.is_solved = True

if st.session_state.is_solved:
    c = df_obj.iloc[0].fillna(0).values.astype(float)
    if method == "1. Scipy (Tổng quát, nhanh)": solve_scipy(c, df_cons, obj_cols, opt_type, bounds)
    elif method == "2. Đồ thị (Chỉ 2 biến)": solve_graph(c, df_cons, n_vars, opt_type)
    elif method == "3. Từ vựng (Đơn hình Dantzig)": solve_dictionary(c, df_cons, obj_cols, opt_type, bounds, rule='dantzig')
    elif method == "4. Từ vựng (Đơn hình Bland)": solve_dictionary(c, df_cons, obj_cols, opt_type, bounds, rule='bland')
    elif method == "5. Chạy tất cả (So sánh)":
        tab1, tab2, tab3, tab4 = st.tabs(["📦 Thư viện Scipy", "📈 Phương pháp Đồ thị", "📝 Từ vựng (Dantzig)", "📝 Từ vựng (Bland)"])
        with tab1: solve_scipy(c, df_cons, obj_cols, opt_type, bounds)
        with tab2: solve_graph(c, df_cons, n_vars, opt_type)
        with tab3: solve_dictionary(c, df_cons, obj_cols, opt_type, bounds, rule='dantzig')
        with tab4: solve_dictionary(c, df_cons, obj_cols, opt_type, bounds, rule='bland')
