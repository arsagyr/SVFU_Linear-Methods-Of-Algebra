"""Lab3: Ручное QR-разложение, обусловленность и время LU/QR."""

import numpy as np
import time
from scipy.linalg import lu, solve_triangular

# ==========================================
# 1. Ручная реализация QR-разложений
# ==========================================

def qr_gram_schmidt(A):
    """Классический процесс ортогонализации Грамма-Шмидта."""
    n, m = A.shape
    Q = np.zeros((n, m))
    R = np.zeros((m, m))
    for j in range(m):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] > 1e-12:
            Q[:, j] = v / R[j, j]
    return Q, R

def qr_givens(A):
    """Метод вращений (Гивенса)."""
    n, m = A.shape
    Q = np.eye(n)
    R = A.copy()
    for j in range(m):
        for i in range(n - 1, j, -1):
            if abs(R[i, j]) > 1e-12:
                r = np.hypot(R[i-1, j], R[i, j])
                c = R[i-1, j] / r
                s = -R[i, j] / r
                G = np.eye(n)
                G[i-1, i-1] = c
                G[i, i-1] = -s
                G[i-1, i] = s
                G[i, i] = c
                R = G.T @ R
                Q = Q @ G
    return Q, R

def qr_householder(A):
    """Метод отражений (Хаусхолдера)."""
    n, m = A.shape
    Q = np.eye(n)
    R = A.copy()
    for k in range(m):
        x = R[k:, k]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        if e[0] > 1e-12:
            v = x + np.sign(x[0]) * e if x[0] != 0 else x + e
            v = v / np.linalg.norm(v)
            H_k = np.eye(n - k) - 2 * np.outer(v, v)
            H = np.eye(n)
            H[k:, k:] = H_k
            R = H @ R
            Q = Q @ H
    return Q, R


# ==========================================
# 2. Вспомогательные функции
# ==========================================

def generate_matrices_A_B(n):
    """Генерирует матрицы A и B из задания 1."""
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    
    # A = (a_ij)
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = 2 * i + 1
            elif i == j - 1:
                A[i, j] = i
            elif i == j + 1:
                A[i, j] = j
            else:
                A[i, j] = 0.0
                
    # B = (b_ij)
    for i in range(n):
        for j in range(n):
            B[i, j] = np.sin((i + 1) * (j + 1) / (2.0 * n))
            
    return A, B

def matrix_cond(A):
    """Безопасное вычисление числа обусловленности (2-норма)."""
    try:
        return np.linalg.cond(A)
    except np.linalg.LinAlgError:
        return float('inf')


# ==========================================
# 3. Главная функция
# ==========================================

def run_lab(n):
    try:
        n = int(n)
    except ValueError:
        return {"ok": False, "error": "n должно быть целым числом"}
    if not (2 <= n <= 200):
        return {"ok": False, "error": "n должно быть от 2 до 200"}

    blocks = []

    # --- ЗАДАНИЕ 1: QR-разложение для A и B ---
    A, B = generate_matrices_A_B(n)
    
    # Тестируем на матрице A
    Q_gs, R_gs = qr_gram_schmidt(A)
    Q_giv, R_giv = qr_givens(A)
    Q_hh, R_hh = qr_householder(A)

    # Проверка ортогональности: ||Q^T * Q - I||
    def orthogonality_error(Q):
        return np.linalg.norm(Q.T @ Q - np.eye(Q.shape[0]))
    
    # Проверка точности разложения ||QR - A||
    def qr_error(Q, R, original):
        return np.linalg.norm(Q @ R - original)

    # Числа обусловленности
    cond_A = matrix_cond(A)
    cond_B = matrix_cond(B)

    table1 = {
        "type": "table",
        "title": f"Задание 1: QR-разложение и ортогональность (n={n})",
        "columns": [
            {"key": "method", "label": "Метод"},
            {"key": "orth_err", "label": "||Q^T Q - I||"},
            {"key": "qr_err", "label": "||QR - A||"},
            {"key": "cond", "label": "Число обусловленности"}
        ],
        "rows": [
            {"method": "Грамм-Шмидт", "orth_err": f"{orthogonality_error(Q_gs):.2e}", "qr_err": f"{qr_error(Q_gs, R_gs, A):.2e}", "cond": f"{cond_A:.2e}"},
            {"method": "Вращения (Гивенс)", "orth_err": f"{orthogonality_error(Q_giv):.2e}", "qr_err": f"{qr_error(Q_giv, R_giv, A):.2e}", "cond": f"{cond_A:.2e}"},
            {"method": "Отражения (Хаусхолдер)", "orth_err": f"{orthogonality_error(Q_hh):.2e}", "qr_err": f"{qr_error(Q_hh, R_hh, A):.2e}", "cond": f"{cond_A:.2e}"}
        ]
    }
    
    table2 = {
        "type": "table",
        "title": f"Матрица B (n={n})",
        "columns": [
            {"key": "param", "label": "Параметр"},
            {"key": "value", "label": "Значение"}
        ],
        "rows": [
            {"param": "Число обусловленности B", "value": f"{cond_B:.2e}"},
            {"param": "Число обусловленности A", "value": f"{cond_A:.2e}"}
        ]
    }
    blocks.extend([table1, table2])

    # --- ЗАДАНИЕ 2: Произведение матриц и время LU/QR ---
    # Возьмем размерность n для исследования
    n_study = n if n >= 10 else 10
    
    # Матрица с малыми диагональными элементами (почти вырожденная)
    L = np.tril(np.random.rand(n_study, n_study)) + np.eye(n_study) * 0.01
    U = np.triu(np.random.rand(n_study, n_study)) + np.eye(n_study) * 0.01
    M = L @ U
    
    cond_M = matrix_cond(M)
    
    # Генерируем случайное решение
    x_true = np.random.rand(n_study)
    b = M @ x_true
    
    # --- Время LU ---
    start = time.perf_counter()
    P, L_lu, U_lu = lu(M)
    y_lu = solve_triangular(L_lu, P @ b, lower=True)
    x_lu = solve_triangular(U_lu, y_lu)
    time_lu = time.perf_counter() - start
    
    # --- Время QR (используем встроенный scipy для честного замера) ---
    from scipy.linalg import qr
    start = time.perf_counter()
    Q_np, R_np = qr(M)
    y_qr = Q_np.T @ b
    x_qr = solve_triangular(R_np, y_qr)
    time_qr = time.perf_counter() - start

    # Ошибки
    r_lu = np.linalg.norm(b - M @ x_lu)
    r_qr = np.linalg.norm(b - M @ x_qr)
    
    rel_lu = r_lu / np.linalg.norm(x_lu) if np.linalg.norm(x_lu) > 0 else 0
    rel_qr = r_qr / np.linalg.norm(x_qr) if np.linalg.norm(x_qr) > 0 else 0

    table3 = {
        "type": "table",
        "title": f"Задание 2: Сравнение LU и QR (n={n_study})",
        "columns": [
            {"key": "method", "label": "Метод"},
            {"key": "time", "label": "Время (сек)"},
            {"key": "residual", "label": "Невязка ||b - Ax||"},
            {"key": "rel_error", "label": "Относит. ошибка r/||x||"}
        ],
        "rows": [
            {"method": "LU-разложение", "time": f"{time_lu:.6f}", "residual": f"{r_lu:.2e}", "rel_error": f"{rel_lu:.2e}"},
            {"method": "QR-разложение", "time": f"{time_qr:.6f}", "residual": f"{r_qr:.2e}", "rel_error": f"{rel_qr:.2e}"}
        ]
    }
    
    table4 = {
        "type": "table",
        "title": "Параметры почти вырожденной матрицы",
        "columns": [
            {"key": "param", "label": "Параметр"},
            {"key": "value", "label": "Значение"}
        ],
        "rows": [
            {"param": "Число обусловленности M", "value": f"{cond_M:.2e}"},
        ]
    }
    blocks.extend([table3, table4])

    # --- График зависимости cond от n ---
    # (До 30, чтобы не грузить сервер)
    max_n_chart = min(n_study, 30)
    ns_chart = list(range(2, max_n_chart + 1))
    conds = []
    for size in ns_chart:
        _, B_chart = generate_matrices_A_B(size)
        conds.append(matrix_cond(B_chart))

    chart = {
        "type": "chart",
        "title": "Число обусловленности матрицы B от n",
        "chart_type": "line",
        "labels": [str(x) for x in ns_chart],
        "datasets": [
            {
                "label": "cond(B)",
                "data": conds,
                "borderColor": "#e74c3c",
                "backgroundColor": "rgba(231, 76, 60, 0.1)",
                "fill": True,
                "tension": 0.2
            }
        ]
    }
    blocks.append(chart)

    # ==========================================
    # ВИЗУАЛИЗАЦИЯ МАТРИЦ
    # ==========================================
    
    # 1. Вывод исходных матриц A и B
    # Ограничим вывод, если n слишком большое (например, > 15)
    # Чтобы таблица не растянулась на весь экран
    max_preview = 10
    n_preview = min(n, max_preview)
    
    A_preview = np.round(A[:n_preview, :n_preview], 2)
    B_preview = np.round(B[:n_preview, :n_preview], 2)
    
    blocks.append({
        "type": "matrix",
        "title": f"Матрицы A и B (фрагмент {n_preview}x{n_preview})",
        "matrices": [
            {"label": "A (трёхдиагональная)", "data": A_preview.tolist()},
            {"label": "B (sin)", "data": B_preview.tolist()}
        ]
    })

    # 2. Матрица ортогональности Q^T Q для метода Хаусхолдера
    # Это докажет, что алгоритм работает корректно
    Q_hh_preview = Q_hh[:n_preview, :n_preview]
    Q_gs_preview = Q_gs[:n_preview, :n_preview]
    
    blocks.append({
        "type": "matrix",
        "title": f"Проверка ортогональности: Q^T Q (фрагмент {n_preview}x{n_preview})",
        "matrices": [
            {"label": "Q^T Q (Хаусхолдер)", "data": np.round(Q_hh_preview.T @ Q_hh_preview, 2).tolist(), "highlight": True},
            {"label": "Q^T Q (Грамм-Шмидт)", "data": np.round(Q_gs_preview.T @ Q_gs_preview, 2).tolist()}
        ]
    })



    return {"ok": True, "blocks": blocks}