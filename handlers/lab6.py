"""Lab6: Степенной метод (PM) и метод скалярных произведений (SP)."""

import numpy as np
import scipy.linalg as la

def power_method(A, tol=1e-6, max_iter=1000):
    """
    Степенной метод (PM).
    Возвращает: (собственное значение, собственный вектор, количество итераций, история сходимости)
    """
    n = A.shape[0]
    x = np.ones(n) # Начальное приближение
    x = x / np.linalg.norm(x)
    
    history = []
    lam_old = 0.0
    
    for k in range(max_iter):
        # Умножаем матрицу на вектор
        y = A @ x
        # Находим норму (используем бесконечную норму для оценки сдвига)
        lam = np.linalg.norm(y, np.inf)
        # Нормируем
        x_new = y / lam
        
        # Оцениваем разницу
        diff = np.linalg.norm(x_new - x)
        history.append(diff)
        
        if diff < tol:
            return lam, x_new, k + 1, history
            
        x = x_new
        lam_old = lam
        
    return lam, x, max_iter, history


def sp_method(A, tol=1e-6, max_iter=1000):
    """
    Метод скалярных произведений (SP).
    """
    n = A.shape[0]
    x = np.ones(n) # Начальное приближение
    x = x / np.linalg.norm(x)
    
    history = []
    
    for k in range(max_iter):
        y = A @ x
        
        # В методе SP собственное значение оценивается через отношение Рэлея
        # lam = (Ax, x) / (x, x)
        lam = np.dot(y, x) / np.dot(x, x)
        
        # Нормируем вектор
        x_new = y / np.linalg.norm(y)
        
        diff = np.linalg.norm(x_new - x)
        history.append(diff)
        
        if diff < tol:
            return lam, x_new, k + 1, history
            
        x = x_new
        
    return lam, x, max_iter, history


def run_lab(matrix_choice):
    blocks = []
    
    # --- Выбор матрицы ---
    if matrix_choice.startswith("1a"):
        A = np.array([
            [8, 1, 3],
            [2, 10, 0],
            [1, 1, 12]
        ], dtype=float)
        method = "PM"
    elif matrix_choice.startswith("1b"):
        A = np.array([
            [10, 0, 2, 5, 0],
            [2, 1, 0, 4, 0],
            [3, 0, 4, 1, 0],
            [0, 2, 4, 2, 0],
            [3, 0, 0, 1, 1]
        ], dtype=float)
        method = "PM"
    else:
        A = np.array([
            [16, 2, 3, 4],
            [2, 16, 3, 4],
            [3, 3, 16, 2],
            [4, 4, 2, 16]
        ], dtype=float)
        method = "SP"

    n = A.shape[0]
    
    # Выводим матрицу
    blocks.append({
        "type": "matrix",
        "title": f"Исходная матрица A ({n}x{n})",
        "matrices": [
            {"label": "A", "data": A.tolist()}
        ]
    })

    # --- Вызов нашего метода ---
    if method == "PM":
        lam_my, vec_my, iters, history = power_method(A)
    else:
        lam_my, vec_my, iters, history = sp_method(A)

    # --- Точное решение через scipy ---
    eigvals, eigvecs = la.eig(A)
    idx_max = np.argmax(np.abs(eigvals))
    lam_exact = np.real(eigvals[idx_max])
    vec_exact = np.real(eigvecs[:, idx_max])
    
    # Приводим вектора к одному знаку (они могут отличаться на -1)
    if np.dot(vec_my, vec_exact) < 0:
        vec_exact = -vec_exact
        
    # Ошибки
    err_lam = abs(lam_my - lam_exact)
    err_vec = np.linalg.norm(vec_my - vec_exact)

    # --- Таблица с результатами ---
    table_res = {
        "type": "table",
        "title": f"Сравнение результатов ({'Степенной метод' if method == 'PM' else 'Метод скал. произведений'})",
        "columns": [
            {"key": "param", "label": "Параметр"},
            {"key": "my", "label": "Наш метод"},
            {"key": "exact", "label": "scipy.linalg.eig"},
            {"key": "error", "label": "Абсолютная ошибка"}
        ],
        "rows": [
            {"param": "Собственное значение λ", "my": f"{lam_my:.6f}", "exact": f"{lam_exact:.6f}", "error": f"{err_lam:.2e}"},
            {"param": "Итераций", "my": iters, "exact": "-", "error": "-"},
            {"param": "Норма ошибки вектора", "my": "-", "exact": "-", "error": f"{err_vec:.2e}"}
        ]
    }
    blocks.append(table_res)

    # --- Вывод собственного вектора ---
    vec_table = {
        "type": "table",
        "title": "Собственный вектор",
        "columns": [
            {"key": "i", "label": "Индекс"},
            {"key": "my", "label": "Наш метод"},
            {"key": "exact", "label": "scipy.linalg.eig"}
        ],
        "rows": [
            {"i": i, "my": f"{vec_my[i]:.4f}", "exact": f"{vec_exact[i]:.4f}"}
            for i in range(n)
        ]
    }
    blocks.append(vec_table)

    # --- График сходимости ---
    # Прореживаем историю для графика
    step = max(1, len(history) // 50)
    labels = [str(i) for i in range(0, len(history), step)]
    vals = [float(np.log10(h)) for h in history[::step]]
    
    chart = {
        "type": "chart",
        "title": "Сходимость метода (логарифм разности векторов)",
        "chart_type": "line",
        "labels": labels,
        "datasets": [
            {
                "label": "log10(||x_new - x_old||)",
                "data": vals,
                "borderColor": "#3498db",
                "tension": 0.1
            }
        ]
    }
    blocks.append(chart)

    return {"ok": True, "blocks": blocks}