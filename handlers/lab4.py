"""Lab4: Итерационные методы (Якоби, Зейдель, SOR)."""

import numpy as np
import time

def build_system_1(n, alpha):
    """Строит систему для Задания 1 (с параметром alpha)."""
    A = np.zeros((n + 1, n + 1))
    f = np.zeros(n + 1)
    
    # Заполняем матрицу A
    for i in range(n + 1):
        A[i, i] = 2.0
        if i < n:
            A[i, i + 1] = -1 - alpha
        if i > 0:
            A[i, i - 1] = -1 + alpha
            
    # Заполняем вектор правой части f
    f[0] = 1 - alpha
    for i in range(1, n):
        f[i] = 0.0
    f[n] = 1 + alpha
    
    return A, f

def build_system_2(n):
    """Строит систему для Задания 2 (для SOR)."""
    A = np.zeros((n + 1, n + 1))
    f = np.zeros(n + 1)
    
    # Заполняем матрицу A
    for i in range(n + 1):
        A[i, i] = 2.0
        if i < n:
            A[i, i + 1] = -1.0
        if i > 0:
            A[i, i - 1] = -1.0
            
    # Заполняем вектор правой части f
    f[0] = 1.0
    for i in range(1, n):
        f[i] = 0.0
    f[n] = -1.0
    
    return A, f

def jacobi(A, f, tol=1e-6, max_iter=5000):
    """Метод Якоби."""
    n = len(f)
    x = np.zeros(n)
    D = np.diag(A)
    R = A - np.diag(D)
    
    history = []
    for k in range(max_iter):
        x_new = (f - R @ x) / D
        res = np.linalg.norm(x_new - x) # Невязка по шагу
        history.append(res)
        if res < tol:
            return x_new, k + 1, history
        x = x_new
        
    return x, max_iter, history

def seidel(A, f, tol=1e-6, max_iter=5000):
    """Метод Зейделя (Гаусса-Зейделя)."""
    n = len(f)
    x = np.zeros(n)
    history = []
    
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x[i] = (f[i] - s) / A[i, i]
            
        res = np.linalg.norm(x - x_old)
        history.append(res)
        if res < tol:
            return x, k + 1, history
            
    return x, max_iter, history

def sor(A, f, omega, tol=1e-6, max_iter=5000):
    """Метод верхней релаксации (SOR)."""
    n = len(f)
    x = np.zeros(n)
    history = []
    
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (f[i] - s)
            
        res = np.linalg.norm(x - x_old)
        history.append(res)
        if res < tol:
            return x, k + 1, history
            
    return x, max_iter, history

def run_lab(n, alpha):
    try:
        n = int(n)
        alpha = float(alpha)
    except ValueError:
        return {"ok": False, "error": "n и alpha должны быть числами"}
        
    if not (3 <= n <= 100):
        return {"ok": False, "error": "n должно быть от 3 до 100"}
        
    blocks = []

    # ==========================================
    # ЗАДАНИЕ 1: Якоби vs Зейдель
    # ==========================================
    A1, f1 = build_system_1(n, alpha)
    
    x_jac, it_jac, hist_jac = jacobi(A1, f1)
    x_sei, it_sei, hist_sei = seidel(A1, f1)
    
    table1 = {
        "type": "table",
        "title": f"Задание 1: Сравнение Якоби и Зейделя (n={n}, alpha={alpha})",
        "columns": [
            {"key": "method", "label": "Метод"},
            {"key": "iterations", "label": "Итераций до сходимости"},
            {"key": "residual", "label": "Конечная невязка"}
        ],
        "rows": [
            {"method": "Якоби", "iterations": it_jac, "residual": f"{hist_jac[-1]:.2e}" if hist_jac else "-"},
            {"method": "Зейделя", "iterations": it_sei, "residual": f"{hist_sei[-1]:.2e}" if hist_sei else "-"}
        ]
    }
    
    # График сходимости (логарифмическая шкала)
    # Чтобы график не был слишком тяжелым, проредим точки, если их много
    def prepare_chart_data(hist):
        step = max(1, len(hist) // 100) # не более 100 точек
        labels = [i for i in range(0, len(hist), step)]
        values = [float(np.log10(h)) for h in hist[::step]]
        return labels, values

    labels_jac, vals_jac = prepare_chart_data(hist_jac)
    labels_sei, vals_sei = prepare_chart_data(hist_sei)

    chart1 = {
        "type": "chart",
        "title": f"Сходимость методов (n={n}, alpha={alpha})",
        "chart_type": "line",
        "labels": [str(i) for i in labels_jac], # Предполагаем, что длины примерно совпадают
        "datasets": [
            {
                "label": "Якоби",
                "data": vals_jac,
                "borderColor": "#e74c3c",
                "tension": 0.1
            },
            {
                "label": "Зейдель",
                "data": vals_sei,
                "borderColor": "#3498db",
                "tension": 0.1
            }
        ]
    }
    
    blocks.extend([table1, chart1])

    # ==========================================
    # ЗАДАНИЕ 2: SOR
    # ==========================================
    A2, f2 = build_system_2(n)
    
    # Подбор коэффициента релаксации omega
    # Теоретически для такой матрицы omega_opt = 2 / (1 + sin(pi / (n+1)))
    # Но мы проверим несколько значений и выведем таблицу
    omegas = [0.5, 1.0, 1.5, 1.8]
    if n <= 50:
        omegas.append(2.0 / (1.0 + np.sin(np.pi / (n + 1))))
    
    table2_rows = []
    best_omega = 1.0
    best_iters = float('inf')
    best_hist = []
    
    for w in omegas:
        x_sor, it_sor, hist_sor = sor(A2, f2, w)
        table2_rows.append({
            "omega": f"{w:.4f}",
            "iterations": it_sor,
            "residual": f"{hist_sor[-1]:.2e}" if hist_sor else "-"
        })
        if it_sor < best_iters:
            best_iters = it_sor
            best_omega = w
            best_hist = hist_sor

    table2 = {
        "type": "table",
        "title": "Задание 2: Метод SOR (подбор omega)",
        "columns": [
            {"key": "omega", "label": "Коэффициент omega"},
            {"key": "iterations", "label": "Итераций"},
            {"key": "residual", "label": "Конечная невязка"}
        ],
        "rows": table2_rows
    }
    
    # График для лучшей омеги
    labels_sor, vals_sor = prepare_chart_data(best_hist)
    chart2 = {
        "type": "chart",
        "title": f"Сходимость SOR (лучшая omega={best_omega:.4f})",
        "chart_type": "line",
        "labels": [str(i) for i in labels_sor],
        "datasets": [
            {
                "label": f"SOR (omega={best_omega:.4f})",
                "data": vals_sor,
                "borderColor": "#2ecc71",
                "tension": 0.1
            }
        ]
    }
    
    blocks.extend([table2, chart2])

    return {"ok": True, "blocks": blocks}