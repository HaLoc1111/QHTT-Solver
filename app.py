import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import time
import json

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

            genai.configure(api_key=api_key)
            image = Image.open(uploaded_file)
            
            prompt = "Bạn là chuyên gia Toán học. Hãy đọc bài toán Quy hoạch tuyến tính trong ảnh.\nTrả về DUY NHẤT một JSON hợp lệ (không có markdown, không có text khác) với cấu trúc:\n{\n\"opt_type\": \"MAX\" hoặc \"MIN\",\n\"n_vars\": số nguyên (số lượng biến),\n\"n_cons\": số nguyên (số lượng ràng buộc),\n\"obj\": [mảng số thực chứa các hệ số hàm mục tiêu],\n\"cons\": [\n{\"coeffs\": [mảng hệ số], \"sign\": \"<=\" hoặc \">=\" hoặc \"=\", \"rhs\": số vế phải}\n]\n}"
            
            with st.spinner("🤖 AI đang giải mã chữ viết tay của bạn..."):
                # CHỈ GỌI DUY NHẤT MODEL CHUẨN ĐỂ LẤY LỖI GỐC
                model = genai.GenerativeModel('gemini-2.5-flash')
                
                try:
                    response = model.generate_content([prompt, image])
                except Exception as api_err:
                    st.error(f"🛑 LỖI TỪ GOOGLE: {api_err}")
                    st.stop()
                
                # Làm sạch chuỗi JSON an toàn
                raw_text = response.text.strip()
                raw_text = raw_text.replace("```json", "")
                raw_text = raw_text.replace("```", "")
                raw_text = raw_text.strip()
                    
                data = json.loads(raw_text)
                
                st.session_state.opt_type = data.get("opt_type", "MAX").upper()
                st.session_state.n_vars = int(data.get("n_vars", 2))
                st.session_state.n_cons = int(data.get("n_cons", 2))
                
                obj_cols = [f"x{i+1}" for i in range(st.session_state.n_vars)]
                st.session_state.init_obj = pd.DataFrame([data["obj"]], columns=obj_cols)
                
                cons_data = []
                for c in data["cons"]:
                    row = c["coeffs"] + [c["sign"], float(c["rhs"])]
                    cons_data.append(row)
                st.session_state.init_cons = pd.DataFrame(cons_data, columns=obj_cols + ["Dấu", "RHS"])
                
                st.sidebar.success("✨ Nhận diện thành công! Dữ liệu đã được điền.")
                time.sleep(1.5)
                st.rerun()

        except Exception as e:
            st.sidebar.error(f"❌ Lỗi code xử lý: {e}")
    else:
        st.sidebar.warning("⚠️ Vui lòng tải ảnh lên trước nhé!")

st.sidebar.markdown("---")

st.sidebar.subheader("📂 Quản lý dữ liệu mẫu")
if st.sidebar.button("📝 Tải mẫu 1 Pha (RHS >= 0)", help="Điền bài toán cơ bản (RHS dương). Dùng test Đơn hình 1 pha."):
    st.session_state.n_vars, st.session_state.n_cons, st.session_state.opt_type = 2, 3, "MAX"
    st.session_state.init_obj = pd.DataFrame([[2.0, 5.0]], columns=["x1", "x2"])
    st.session_state.init_cons = pd.DataFrame([[1.0, 0.0, "<=", 4.0], [0.0, 2.0, "<=", 12.0], [3.0, 2.0, "<=", 18.0]], columns=["x1", "x2", "Dấu", "RHS"])
    st.rerun()

if st.sidebar.button("📚 Tải mẫu 2 Pha (Trong vở ghi)", help="Điền bài toán suy biến (RHS âm). Dùng test tính năng Đơn hình 2 pha."):
    st.session_state.n_vars, st.session_state.n_cons, st.session_state.opt_type = 2, 3, "MIN"
    st.session_state.init_obj = pd.DataFrame([[1.0, 2.0]], columns=["x1", "x2"])
    st.session_state.init_cons = pd.DataFrame([[-1.0, 1.0, "<=", -2.0], [-1.0, -2.0, "<=", -4.0], [0.0, 1.0, "<=", 2.0]], columns=["x1", "x2", "Dấu", "RHS"])
    st.rerun()

if st.sidebar.button("🔄 Đặt lại bảng trống", help="Xóa sạch dữ liệu."):
    st.session_state.n_vars, st.session_state.n_cons, st.session_state.opt_type = 2, 3, "MAX"
    st.session_state.init_obj = pd.DataFrame([[0.0, 0.0]], columns=["x1", "x2"])
    st.session_state.init_cons = pd.DataFrame([[0.0, 0.0, "<=", 0.0] for _ in range(3)], columns=["x1", "x2", "Dấu", "RHS"])
    st.rerun()
st.sidebar.markdown("---")

n_vars = st.sidebar.number_input("Số lượng biến", 1, 20, st.session_state.n_vars, key="vars_input")
n_cons = st.sidebar.number_input("Số lượng ràng buộc", 1, 20, st.session_state.n_cons, key="cons_input")
opt_type = st.sidebar.radio("Mục tiêu tối ưu", ("MAX", "MIN"), index=0 if st.session_state.opt_type == "MAX" else 1, key="opt_input")

obj_cols = [f"x{i+1}" for i in range(n_vars)]
if st.session_state.init_obj.shape[1] != n_vars:
    st.session_state.init_obj = pd.DataFrame([[0.0] * n_vars], columns=obj_cols)
if st.session_state.init_cons.shape[0] != n_cons or st.session_state.init_cons.shape[1] != (n_vars + 2):
    st.session_state.init_cons = pd.DataFrame([[0.0] * n_vars + ["<=", 0.0] for _ in range(n_cons)], columns=obj_cols + ["Dấu", "RHS"])

# ----------------- GIAO DIỆN NHẬP LIỆU CHÍNH -----------------
st.subheader("1. Hàm mục tiêu $f(x)$")
df_obj = st.data_editor(st.session_state.init_obj, hide_index=True, use_container_width=True)

st.subheader("2. Hệ ràng buộc")
config = {"Dấu": st.column_config.SelectboxColumn("Dấu", options=["<=", ">=", "="], required=True)}
df_cons = st.data_editor(st.session_state.init_cons, column_config=config, hide_index=True, use_container_width=True)

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
    c_vals = df_obj.iloc[0].values
    obj_str = " ".join([f"{'+' if v>=0 and i>0 else ('-' if v<0 else '')} {abs(v):.2f}x_{i+1}" for i, v in enumerate(c_vals) if abs(v)>1e-6 or (i==0 and all(x==0 for x in c_vals))])
    
    cons_lines = []
    for _, row in df_cons.iterrows():
        line_str = " ".join([f"{'+' if float(row[col])>=0 and i>0 else ('-' if float(row[col])<0 else '')} {abs(float(row[col])):.2f}x_{i+1}" for i, col in enumerate(obj_cols) if abs(float(row[col]))>1e-6])
        if not line_str: line_str = "0"
        sign_symbol = "\\le" if row["Dấu"] == "<=" else ("\\ge" if row["Dấu"] == ">=" else "=")
        cons_lines.append(f"{line_str} & {sign_symbol} {float(row['RHS']):.2f}")
    
    bound_terms = [f"x_{i+1} \\ge 0" if b==(0,None) else (f"x_{i+1} \\le 0" if b==(None,0) else f"x_{i+1} \\text{{ tùy ý}}") for i, b in enumerate(bounds)]
    
    # Đã sửa triệt để lỗi NameError bằng cách tách chuỗi nối tay
    latex_model = "$$\n\\begin{array}{ll}\n"
    latex_model += "\\text{Tối ưu hóa:} & \\" + opt_type.lower() + f" \\quad Z = {obj_str if obj_str else '0'} \\\\\n"
    latex_model += "\\text{Thỏa mãn:} & \\left\\{\n\\begin{array}{l}\n"
    latex_model += r" \\ ".join(cons_lines) + "\n\\end{array}\n\\right. \\\\\n"
    latex_model += "& " + ", ".join(bound_terms) + "\n\\end{array}\n$$" 
    return latex_model

def render_dual_model_latex(df_obj, df_cons, obj_cols, opt_type, bounds):
    c_vals = df_obj.iloc[0].values
    b_vals = [float(row["RHS"]) for _, row in df_cons.iterrows()]
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

    # Đã sửa triệt để lỗi NameError
    latex_model = "$$\n\\begin{array}{ll}\n"
    latex_model += "\\text{Bài toán Đối ngẫu:} & \\" + dual_opt.lower() + f" \\quad W = {dual_obj_str if dual_obj_str else '0'} \\\\\n"
    latex_model += "\\text{Thỏa mãn:} & \\left\\{\n\\begin{array}{l}\n"
    latex_model += r" \\ ".join(dual_cons_lines) + "\n\\end{array}\n\\right. \\\\\n"
    latex_model += "& " + ", ".join(dual_bounds) + "\n\\end{array}\n$$"
    return latex_model

with tab_model_primal:
    st.markdown(render_math_model_latex(df_obj, df_cons, obj_cols, opt_type, bounds))
with tab_model_dual:
    st.info("💡 Phân tích: Thuật toán tự động sinh bài toán đối ngẫu (Dual) bằng cách chuyển vị ma trận hệ số, đảo MIN/MAX và áp dụng quy tắc đổi dấu.")
    st.markdown(render_dual_model_latex(df_obj, df_cons, obj_cols, opt_type, bounds))

# ----------------- CÁC HÀM XỬ LÝ (CORE LOGIC) -----------------

def log_and_print(log, text):
    st.markdown(text)
    log.append(text)

def solve_scipy(c, df_cons, obj_cols, opt_type, bounds):
    c_scipy = -c if opt_type == "MAX" else c
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for _, row in df_cons.iterrows():
        coeffs = row[obj_cols].values.astype(float)
        sign = row["Dấu"]
        rhs = float(row["RHS"])
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
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("🎯 **Chiến lược hành động:**")
            st.dataframe(pd.DataFrame({"Biến số": obj_cols, "Số lượng": np.round(res.x, 4)}))
            st.info("💡 **Giải thích:** Đây là phương án sản xuất tốt nhất của bạn.")
            
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
        if res.status == 3:
            st.error("❌ BÀI TOÁN KHÔNG GIỚI HẠN (UNBOUNDED)!")
            st.write("Miền nghiệm mở toang tới vô cực, hàm mục tiêu không có giới hạn.")
        elif res.status == 2:
            st.error("❌ BÀI TOÁN VÔ NGHIỆM (INFEASIBLE)!")
            st.write("Không có vùng khả thi thỏa mãn các ràng buộc này.")
        else:
            st.error(f"❌ Lỗi: {res.message}")

def solve_graph(c, df_cons, n_vars, opt_type):
    if n_vars != 2:
        st.error("❌ Phương pháp đồ thị chỉ hỗ trợ bài toán có đúng 2 biến (x1, x2).")
        return
        
    st.write("### 🎚️ Mô phỏng Trượt hàm mục tiêu")
    col1, col2 = st.columns([3, 1])
    with col1:
        z_slider = st.slider("Giá trị Z hiện tại", min_value=-30.0, max_value=30.0, value=0.0, step=0.5)
    with col2:
        st.write(""); st.write("")
        is_auto = st.button("🎬 Bật Tự Động Trượt", type="secondary")

    plot_container = st.empty()
    c1, c2 = float(c[0]), float(c[1])

    def render_frame(current_z):
        fig, ax = plt.subplots(figsize=(8, 8))
        d = np.linspace(-5, 20, 400)
        x1, x2 = np.meshgrid(d, d)
        mask = (x1 >= 0) & (x2 >= 0) 
        
        ax.axhline(0, color='black', linewidth=1.5)
        ax.axvline(0, color='black', linewidth=1.5)
        ax.plot(0, 0, 'ko')
        
        colors = ['blue', 'green', 'purple', 'orange']
        for idx, row in df_cons.iterrows():
            a1, a2 = float(row["x1"]), float(row["x2"])
            rhs = float(row["RHS"])
            sign = row["Dấu"]
            color = colors[idx % len(colors)]
            
            if a2 != 0:
                y = (rhs - a1 * d) / a2
                ax.plot(d, y, color=color, label=f"(PT{idx+1}): {a1}x1 + {a2}x2 {sign} {rhs}")
            else:
                ax.axvline(x=rhs/a1, color=color, label=f"(PT{idx+1}): {a1}x1 {sign} {rhs}")

            if sign == "<=": mask = mask & (a1 * x1 + a2 * x2 <= rhs)
            elif sign == ">=": mask = mask & (a1 * x1 + a2 * x2 >= rhs)
                
        ax.imshow(mask.astype(int), extent=(-5, 20, -5, 20), origin="lower", cmap="Greys", alpha=0.3)
        
        if c2 != 0:
            y_obj = (current_z - c1 * d) / c2
            ax.plot(d, y_obj, 'r-', linewidth=2.5, label=f"Đường Z = {current_z:.1f}")

        ax.set_xlim(-2, 10)
        ax.set_ylim(-2, 10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        return fig

    if is_auto:
        for val in np.arange(-15.0, 20.0 + 1, 1.0):
            fig = render_frame(val)
            plot_container.pyplot(fig)
            plt.close(fig) 
            time.sleep(0.15) 
    else:
        fig = render_frame(z_slider)
        plot_container.pyplot(fig)

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

def run_simplex_loop(N, B, A_N, b_B, c_N, v, var_names, rule, log, obj_name="Z", opt_type="MAX"):
    visited_bases = set()
    iteration = 0
    while True:
        display_v = -v if (opt_type == "MIN" and obj_name == "Z") else v
        display_c_N = -c_N if (opt_type == "MIN" and obj_name == "Z") else c_N
        current_basis = frozenset(B)
        if current_basis in visited_bases:
            log_and_print(log, "⚠️ **PHÁT HIỆN LẶP XOAY VÒNG (CYCLING)!**")
            log_and_print(log, format_dictionary(N, B, A_N, b_B, display_c_N, display_v, var_names, obj_name=obj_name))
            return None
        visited_bases.add(current_basis)

        if all(c_N <= 1e-6):
            log_and_print(log, f"**Lần lặp {iteration} (Tối ưu {obj_name}):**")
            log_and_print(log, format_dictionary(N, B, A_N, b_B, display_c_N, display_v, var_names, obj_name=obj_name))
            if obj_name == "Z": log_and_print(log, f"✅ **Đạt phương án tối ưu! {obj_name} = {display_v:.4f}**")
            return N, B, A_N, b_B, c_N, v

        enter_j = min([j for j, val in enumerate(c_N) if val == np.max(c_N)], key=lambda j: N[j]) if rule == 'dantzig' else min([j for j, val in enumerate(c_N) if val > 1e-6], key=lambda j: N[j])
        m = len(B)
        ratios = [b_B[i] / A_N[i, enter_j] if A_N[i, enter_j] > 1e-6 else np.inf for i in range(m)]
        if all(r == np.inf for r in ratios):
            log_and_print(log, f"**Lần lặp {iteration}:**")
            log_and_print(log, format_dictionary(N, B, A_N, b_B, display_c_N, display_v, var_names, enter_j=enter_j, obj_name=obj_name))
            log_and_print(log, "❌ **Bài toán không giới hạn (Unbounded)!**")
            return None

        leave_i = min([i for i, r in enumerate(ratios) if abs(r - min(ratios)) < 1e-6], key=lambda i: B[i])
        log_and_print(log, f"**Lần lặp {iteration}:**")
        log_and_print(log, format_dictionary(N, B, A_N, b_B, display_c_N, display_v, var_names, enter_j=enter_j, leave_i=leave_i, obj_name=obj_name))
        log_and_print(log, f"🔄 Biến vào: **{var_names[N[enter_j]]}** | Biến ra: **{var_names[B[leave_i]]}**")

        N, B, A_N, b_B, c_N, v = perform_pivot(N, B, A_N, b_B, c_N, v, enter_j, leave_i)
        iteration += 1

def solve_dictionary(c, df_cons, obj_cols, opt_type, bounds, rule='dantzig'):
    log = [] 
    
    if any(b != (0, None) for b in bounds): log_and_print(log, "⚠️ *Thuật toán Từ vựng giả định x >= 0.*")
    if any(df_cons["Dấu"] != "<="):
        log_and_print(log, "⚠️ *Hiện tại bộ giải Từ vựng tự viết chỉ tối ưu cho các ràng buộc có dấu <=*")
        return

    A, b = [], []
    for _, row in df_cons.iterrows():
        A.append(row[obj_cols].values.astype(float))
        b.append(float(row["RHS"]))
        
    n, m = len(c), len(b)
    c_orig = -np.array(c, dtype=float) if opt_type == "MIN" else np.array(c, dtype=float)
    A_N, b_B = np.array(A, dtype=float), np.array(b, dtype=float)
    var_names = [f"x_{i+1}" for i in range(n)] + [f"w_{i+1}" for i in range(m)] + ["x_0"]
    x0_idx = n + m
    
    if np.min(b_B) < 0:
        log_and_print(log, "### 🛠️ PHA 1: Bài toán bổ trợ (Tìm phương án xuất phát)")
        N, B = list(range(n)) + [x0_idx], list(range(n, n + m))
        A_N = np.column_stack((A_N, np.full(m, -1.0)))
        c_N, v = np.zeros(n + 1), 0.0
        c_N[-1] = -1.0 
        leave_i, enter_j = np.argmin(b_B), n 
        
        log_and_print(log, "**Bước khởi tạo Từ vựng Pha 1:**")
        log_and_print(log, format_dictionary(N, B, A_N, b_B, c_N, v, var_names, enter_j, leave_i, obj_name="\\xi"))
        N, B, A_N, b_B, c_N, v = perform_pivot(N, B, A_N, b_B, c_N, v, enter_j, leave_i)

        res = run_simplex_loop(N, B, A_N, b_B, c_N, v, var_names, rule, log, obj_name="\\xi")
        if res is None: return 
        N, B, A_N, b_B, c_N, v = res

        if v < -1e-6:
            log_and_print(log, "❌ **BÀI TOÁN VÔ NGHIỆM! (Pha 1 có x_0 > 0).**")
            return
        log_and_print(log, "✅ **Kết thúc Pha 1 thành công. Khử biến x_0 và chuyển sang Pha 2.**")
        
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
        log_and_print(log, "---\n### 🎯 PHA 2: Giải bài toán gốc")
    else:
        N, B, c_N, v = list(range(n)), list(range(n, n + m)), c_orig.copy(), 0.0

    run_simplex_loop(N, B, A_N, b_B, c_N, v, var_names, rule, log, obj_name="Z", opt_type=opt_type)

    report_content = "\n\n".join(log) 
    st.download_button(
        label="📥 Tải file Báo Cáo chi tiết (.md)",
        data=report_content,
        file_name=f"BaoCao_QHTT_{rule}.md",
        mime="text/markdown",
        help="Tải toàn bộ các bước giải thành file văn bản. Bạn có thể mở bằng Notepad, Word hoặc Typora để copy vào đồ án."
    )

# ----------------- NÚT THỰC THI CHÍNH -----------------
if 'is_solved' not in st.session_state: st.session_state.is_solved = False
if st.button("🚀 GIẢI BÀI TOÁN", type="primary"): st.session_state.is_solved = True

if st.session_state.is_solved:
    c = df_obj.iloc[0].values.astype(float)
    if method == "1. Scipy (Tổng quát, nhanh)": solve_scipy(c, df_cons, obj_cols, opt_type, bounds)
    elif method == "2. Đồ thị (Chỉ 2 biến)": solve_graph(c, df_cons, n_vars, opt_type)
    elif method == "3. Từ vựng (Đơn hình Dantzig)": solve_dictionary(c, df_cons, obj_cols, opt_type, bounds, rule='dantzig')
    elif method == "4. Từ vựng (Đơn hình Bland)": solve_dictionary(c, df_cons, obj_cols, opt_type, bounds, rule='bland')
    elif method == "5. Chạy tất cả (So sánh)":
        tab1, tab2, tab3, tab4 = st.tabs(["📦 Thư viện Scipy", "📈 Phương pháp Đồ thị", "📝 Từ vựng (Dantzig)", "📝 Từ vựng (Bland)"])
        with tab1: solve_scipy(c, df_cons, obj_cols, opt_type, bounds)
        with tab2: solve_graph(c, df_cons, n_vars, opt_type)
        with tab3: solve_dictionary(c, df_cons, obj_cols, opt_type, bounds, rule='dantzig')
        with tab4: solve_dictionary(c, df_cons, obj_cols, opt_type, bounds, rule='bland') cứu tôi
