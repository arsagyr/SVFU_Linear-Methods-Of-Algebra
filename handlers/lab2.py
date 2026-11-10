"""Lab2: Разложения матриц (LU, QR, Холецкий) и метод прогонки."""

import numpy as np
from scipy.linalg import lu, qr, cholesky, solve_triangular

def thomas_algorithm(a, b, c, d):
    """
    Метод прогонки (алгоритм Томаса) для трёхдиагональной матрицы.
    a - нижняя диагональ (n-1), b - главная (n), c - верхняя (n-1), d - правая часть (n).
    """
    n = len(d)
    c_star = np.zeros(n - 1)
    d_star = np.zeros(n)
    
    # Прямой ход
    c_star[0] = c[0] / b[0]
    d_star[0] = d[0] / b[0]
    
    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * c_star[i - 1]
        c_star[i] = c[i] / denom
        d_star[i] = (d[i] - a[i - 1] * d_star[i - 1]) / denom
        
    d_star[n - 1] = (d[n - 1] - a[n - 2] * d_star[n - 2]) / (b[n - 1] - a[n - 2] * c_star[n - 2])
    
    # Обратный ход
    x = np.zeros(n)
    x[n - 1] = d_star[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_star[i] - c_star[i] * x[i + 1]
        
    return x


def run_lab(system):
    # --- Определение систем ---
    if system.startswith("а"):
        # 4x4 с плавающей точкой
        A = np.array([
            [6.1, 6.2, -6.3, 6.4],
            [1.1, -1.5, 2.2, -3.8],
            [5.1, -5.0, 4.9, -4.8],
            [1.8, 1.9, 2.3, -2.1]
        ])
        b = np.array([6.5, 4.2, 4.7, 2.2])
        can_cholesky = False
        
    elif system.startswith("б"):
        # 6x6 разреженная
        A = np.array([
            [ 4,  -6,   0,   8,   0,   0],
            [-6,   8, -12,   0,  16,   0],
            [ 0, -12,  16,   0,   0,  10],
            [ 8,   0,   0,  20,  -8,   0],
            [ 0,  16,   0,  -8,  24,  -8],
            [ 0,   0,  10,   0,  -8,  28]
        ], dtype=float)
        b = np.array([24, 54, 84, 48, 72, 158], dtype=float)
        can_cholesky = True # Симметричная и положительно определенная

    elif system.startswith("в"):
        # Трёхдиагональная n=6 (по картинке)
        n = 6
        # Главная диагональ: 2
        # Нижняя и верхняя: -1
        A = np.diag(np.full(n, 2.0)) + np.diag(np.full(n-1, -1.0), 1) + np.diag(np.full(n-1, -1.0), -1)
        # Вектор b по картинке: [3, -4, 4, -4, 4, -3]
        b = np.array([3, -4, 4, -4, 4, -3], dtype=float)
        can_cholesky = True
        
    else:
        # г) 6x6 разреженная (2)
        A = np.array([
            [10, -1,  0,  0,  0,  0],
            [ 1,-10,  3,  0,  0,  0],
            [ 0,  1,-10, -2,  0,  0],
            [ 0,  0,  2,-10, -1,  0],
            [ 0,  0,  0,  1,-10,  6],
            [ 0,  0,  0,  0,  2,-10]
        ], dtype=float)
        b = np.array([10, 4, -10, 1, -10, 2], dtype=float)
        can_cholesky = False

    n = len(b)
    blocks = []

    # --- Вывод исходной системы ---
    blocks.append({
        "type": "matrix",
        "title": "Исходная матрица A и вектор b",
        "matrices": [
            {"label": "A", "data": np.round(A, 2).tolist()},
            {"label": "b (столбец)", "data": np.round(b.reshape(-1, 1), 2).tolist(), "highlight": True}
        ]
    })

    # --- LU разложение ---
    P, L, U = lu(A)
    # Решение через LU: Ax = b -> LUx = Pb -> Ly = Pb, Ux = y
    y = solve_triangular(L, P @ b, lower=True)
    x_lu = solve_triangular(U, y)
    
    blocks.append({
        "type": "table",
        "title": "LU разложение",
        "columns": [{"key": "method", "label": "Метод"}, {"key": "result", "label": "Вектор решения x"}],
        "rows": [
            {"method": "Через LU (scipy.linalg.lu)", "result": str(np.round(x_lu, 4))}
        ]
    })

    # --- QR разложение ---
    Q, R = qr(A)
    x_qr = solve_triangular(R, Q.T @ b)
    
    blocks.append({
        "type": "table",
        "title": "QR разложение",
        "columns": [{"key": "method", "label": "Метод"}, {"key": "result", "label": "Вектор решения x"}],
        "rows": [
            {"method": "Через QR (scipy.linalg.qr)", "result": str(np.round(x_qr, 4))}
        ]
    })

    # --- Разложение Холецкого ---
    if can_cholesky:
        try:
            L_chol = cholesky(A, lower=True)
            y_chol = solve_triangular(L_chol, b, lower=True)
            x_chol = solve_triangular(L_chol.T, y_chol)
            chol_res = str(np.round(x_chol, 4))
        except np.linalg.LinAlgError:
            chol_res = "Матрица не является положительно определённой"
    else:
        chol_res = "Не применимо (матрица не симметрична или не положительно определена)"

    blocks.append({
        "type": "table",
        "title": "Разложение Холецкого",
        "columns": [{"key": "method", "label": "Метод"}, {"key": "result", "label": "Вектор решения x"}],
        "rows": [
            {"method": "Через Холецкого (scipy.linalg.cholesky)", "result": chol_res}
        ]
    })

    # --- Метод прогонки ---
    # Проверяем, является ли матрица трёхдиагональной
    is_tridiagonal = True
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and A[i, j] != 0:
                is_tridiagonal = False
                break
        if not is_tridiagonal:
            break

    if is_tridiagonal:
        a_diag = np.diag(A, -1)
        b_diag = np.diag(A, 0)
        c_diag = np.diag(A, 1)
        x_thomas = thomas_algorithm(a_diag, b_diag, c_diag, b)
        thomas_res = str(np.round(x_thomas, 4))
    else:
        thomas_res = "Не применимо (матрица не трёхдиагональная)"

    blocks.append({
        "type": "table",
        "title": "Метод прогонки",
        "columns": [{"key": "method", "label": "Метод"}, {"key": "result", "label": "Вектор решения x"}],
        "rows": [
            {"method": "Ручной метод прогонки (Томаса)", "result": thomas_res}
        ]
    })

    # --- Проверка (Невязка) ---
    # Используем решение, полученное первым способом (LU), для проверки
    residual = np.linalg.norm(A @ x_lu - b)
    
    blocks.append({
        "type": "list",
        "title": "Проверка точности",
        "items": [
            f"Невязка ||Ax - b|| для LU: {residual:.2e}",
            "Все методы должны давать одинаковый результат (в пределах погрешности)."
        ]
    })

    return {
        "ok": True,
        "blocks": blocks
    }