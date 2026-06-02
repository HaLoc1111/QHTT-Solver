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
if "n_vars" not in st.session_state:
    st.session_state.n_vars = 2
if "n_cons" not in st.session_state:
    st.session_state.n_cons = 3
if "opt_type" not in st.session_state:
    st.session_state.opt_type = "MAX"
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
# VŨ KHÍ 1: TỰ ĐỘNG SINH BÀI TOÁN ĐỐI NGẪU VÀ MÔ HÌNH LATEX
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
    
    latex_model = "$$\n\\begin{array}{ll}\n"
    latex_model += f"\\text{{Tối ưu hóa:}} & \\{opt_type.lower()} \\quad Z = {obj_str if obj_str else '0'} \\\\\n"
    latex_model += "\\text{Thỏa mãn:} & \\left\\{\n\\begin{array}{l}\n"
    # Đã sửa lại lỗi join chuỗi và lỗi {array}
    latex_model += r" \\ ".join(cons_lines) + "\n\\end{array}\n\\right. \\\\\n"
    latex_model += f"& {', '.join(bound_terms)}\n\\end{{array}}\n$$" 
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

    latex_model = "$$\n\\begin{array}{ll}\n"
    latex_model += f"\\text{{Bài toán Đối ngẫu:}} & \\{dual_opt.lower()} \\quad W = {dual_obj_str if dual_obj_str else '0'} \\\\\n"
    latex_model += "\\text{Thỏa mãn:} & \\left\\{\n\\begin{array}{l}\n"
    # Đã sửa lại lỗi join chuỗi và lỗi {array}
    latex_model += r" \\ ".join(dual_cons_lines) + "\n\\end{array}\n\\right. \\\\\n"
    latex_model += f"& {', '.join(dual_bounds)}\n\\end{{array}}\n$$"
    return latex_model

with tab_model_primal:
    st.markdown(render_math_model_latex(df_obj, df_cons, obj_cols, opt_type, bounds))
with tab_model_dual:
    st.info("💡 Phân tích: Thuật toán tự động sinh bài toán đối ngẫu (Dual) bằng cách chuyển vị ma trận hệ số, đảo MIN/MAX và áp dụng quy tắc đổi dấu.")
    st.markdown(render_dual_model_latex(df_obj, df_cons, obj_cols, opt_type, bounds))

# ----------------- CÁC HÀM XỬ LÝ (CORE LOGIC) -----------------

def log_and_print(log, text):
    """Hàm phụ trợ: Vừa in ra màn hình, vừa lưu vào log để xuất file Báo cáo"""
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
            st.write("**Nghiệm bài toán (Primal values):**")
            st.dataframe(pd.DataFrame({"Biến": obj_cols, "Giá trị": np.round(res.x, 4)}))
            
        with col2:
            st.write("**Phân tích độ nhạy (Shadow Prices):**")
            shadow_prices = []
            if hasattr(res, 'ineqlin') and hasattr(res.ineqlin, 'marginals'):
                shadow_prices.extend(res.ineqlin.marginals)
            if hasattr(res, 'eqlin') and hasattr(res.eqlin, 'marginals'):
                shadow_prices.extend(res.eqlin.marginals)
            
            if shadow_prices:
                sp_vals = np.round(np.abs(shadow_prices), 4) 
                sp_df = pd.DataFrame({"Ràng buộc (PT)": [f"PT {i+1}" for i in range(len(sp_vals))], "Shadow Price": sp_vals})
                st.dataframe(sp_df)
                st.caption("💡 *Shadow Price: Nếu tăng RHS thêm 1 đơn vị, Z sẽ thay đổi bao nhiêu.*")
            else:
                st.write("Không trích xuất được Shadow Prices.")
    else:
        # BẮT LỖI VÀ DỊCH SANG TIẾNG VIỆT
        if res.status == 3:
            st.error("❌ BÀI TOÁN KHÔNG GIỚI HẠN (UNBOUNDED)!")
            st.write("Hàm mục tiêu có thể tiến tới vô cực do miền nghiệm không bị chặn. Hãy kiểm tra lại các ràng buộc (có thể bạn thiếu ràng buộc chặn trên hoặc chọn sai mục tiêu MIN/MAX).")
        elif res.status == 2:
            st.error("❌ BÀI TOÁN VÔ NGHIỆM (INFEASIBLE)!")
            st.write("Không tồn tại phương án nào thỏa mãn tất cả các ràng buộc cùng một lúc (Tập chấp nhận được là tập rỗng).")
        else:
            st.error(f"❌ Lỗi chưa xác định từ bộ giải: {res.message}")
        st.error(res.message)

def solve_graph(c, df_cons, n_vars, opt_type):
    # (Giữ nguyên hàm vẽ đồ thị cũ, bỏ qua ở đây cho gọn, bạn copy paste mã đồ thị cũ vào đây)
    pass # Thay bằng thân hàm đồ thị đã hoàn thiện ở phiên bản trước
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
        
    # Đã bọc r"" để giữ lại chuẩn xác cấu trúc dấu gạch ngang
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
    log = [] # VŨ KHÍ 3: Mảng lưu trữ toàn bộ lịch sử chạy để xuất Report
    
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

    # VŨ KHÍ 3: HIỂN THỊ NÚT TẢI BÁO CÁO (MARKDOWN)
    report_content = "\n\n".join(log) # Gom tất cả log thành 1 file text
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
    # elif method == "2. Đồ thị (Chỉ 2 biến)": solve_graph(c, df_cons, n_vars, opt_type)
    elif method == "3. Từ vựng (Đơn hình Dantzig)": solve_dictionary(c, df_cons, obj_cols, opt_type, bounds, rule='dantzig')
    elif method == "4. Từ vựng (Đơn hình Bland)": solve_dictionary(c, df_cons, obj_cols, opt_type, bounds, rule='bland')
    elif method == "5. Chạy tất cả (So sánh)":
        tab1, tab3, tab4 = st.tabs(["📦 Thư viện Scipy", "📝 Từ vựng (Dantzig)", "📝 Từ vựng (Bland)"]) # Tạm ẩn tab 2 đồ thị cho code gọn
        with tab1: solve_scipy(c, df_cons, obj_cols, opt_type, bounds)
        with tab3: solve_dictionary(c, df_cons, obj_cols, opt_type, bounds, rule='dantzig')
        with tab4: solve_dictionary(c, df_cons, obj_cols, opt_type, bounds, rule='bland')