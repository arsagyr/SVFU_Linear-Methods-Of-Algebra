"""Lab1: Нормы векторов и матриц, число обусловленности."""

import numpy as np
from scipy.linalg import norm as scipy_norm
import numpy.linalg as npla

def run_lab(n, vector, matrix_mode):
    try:
        n = int(n)
    except ValueError:
        return {"ok": False, "error": "n должно быть целым числом"}

    if not (2 <= n <= 50):
        return {"ok": False, "error": "n должно быть в диапазоне 2..50"}

    # --- 1. Работа с вектором ---
    try:
        # Берем первые n элементов, или дополняем нулями, или обрезаем
        vec_list = [float(x.strip()) for x in vector.split(",") if x.strip()]
        if len(vec_list) < n:
            vec_list += [0.0] * (n - len(vec_list))
        x = np.array(vec_list[:n])
    except ValueError:
        return {"ok": False, "error": "Вектор x должен содержать числа, разделенные запятыми"}

    # Вычисление норм вектора вручную
    norm_x_1 = np.sum(np.abs(x))
    norm_x_2 = np.sqrt(np.sum(x ** 2))
    norm_x_inf = np.max(np.abs(x))

    # Сравнение со scipy
    scipy_x_1 = scipy_norm(x, 1)
    scipy_x_2 = scipy_norm(x, 2)
    scipy_x_inf = scipy_norm(x, np.inf)

    vec_table = {
        "type": "table",
        "title": "Нормы вектора x",
        "columns": [
            {"key": "norm", "label": "Норма"},
            {"key": "manual", "label": "Вычислено вручную"},
            {"key": "scipy", "label": "scipy.linalg.norm"},
            {"key": "match", "label": "Совпадает?"}
        ],
        "rows": [
            {"norm": "||x||_1", "manual": round(norm_x_1, 6), "scipy": round(scipy_x_1, 6), "match": np.isclose(norm_x_1, scipy_x_1)},
            {"norm": "||x||_2", "manual": round(norm_x_2, 6), "scipy": round(scipy_x_2, 6), "match": np.isclose(norm_x_2, scipy_x_2)},
            {"norm": "||x||_inf", "manual": round(norm_x_inf, 6), "scipy": round(scipy_x_inf, 6), "match": np.isclose(norm_x_inf, scipy_x_inf)},
        ]
    }

    # --- 2. Работа с матрицей ---
    if matrix_mode == "a_ij = min(i,j)":
        # Создаем матрицу Гильберта-подобного типа
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                A[i, j] = min(i + 1, j + 1) # +1, так как индексы в задаче с 1
    else:
        # Случайная матрица
        np.random.seed(42) # Фиксируем seed для воспроизводимости
        A = np.random.rand(n, n) * 10 - 5 # числа от -5 до 5

    # Нормы матрицы вручную
    norm_A_1 = np.max(np.sum(np.abs(A), axis=0)) # max по столбцам
    norm_A_inf = np.max(np.sum(np.abs(A), axis=1)) # max по строкам
    norm_A_E = np.sqrt(np.sum(A ** 2)) # Норма Фробениуса

    # Сравнение со scipy
    scipy_A_1 = scipy_norm(A, 1)
    scipy_A_inf = scipy_norm(A, np.inf)
    scipy_A_E = scipy_norm(A, 'fro')

    mat_table = {
        "type": "table",
        "title": "Нормы матрицы A",
        "columns": [
            {"key": "norm", "label": "Норма"},
            {"key": "manual", "label": "Вычислено вручную"},
            {"key": "scipy", "label": "scipy.linalg.norm"},
            {"key": "match", "label": "Совпадает?"}
        ],
        "rows": [
            {"norm": "||A||_1", "manual": round(norm_A_1, 6), "scipy": round(scipy_A_1, 6), "match": np.isclose(norm_A_1, scipy_A_1)},
            {"norm": "||A||_inf", "manual": round(norm_A_inf, 6), "scipy": round(scipy_A_inf, 6), "match": np.isclose(norm_A_inf, scipy_A_inf)},
            {"norm": "||A||_E", "manual": round(norm_A_E, 6), "scipy": round(scipy_A_E, 6), "match": np.isclose(norm_A_E, scipy_A_E)},
        ]
    }

    # --- 3. Проверка неравенства ---
    # 1/sqrt(n) * ||A||_alpha <= ||A||_E <= sqrt(n) * ||A||_alpha
    sqrt_n = np.sqrt(n)
    
    ineq_1_left = (1/sqrt_n) * norm_A_1
    ineq_1_right = sqrt_n * norm_A_1
    check_1 = ineq_1_left <= norm_A_E <= ineq_1_right

    ineq_inf_left = (1/sqrt_n) * norm_A_inf
    ineq_inf_right = sqrt_n * norm_A_inf
    check_inf = ineq_inf_left <= norm_A_E <= ineq_inf_right

    ineq_list = {
        "type": "list",
        "title": "Проверка неравенства",
        "items": [
            f"Для alpha=1: {ineq_1_left:.4f} <= {norm_A_E:.4f} <= {ineq_1_right:.4f} -> {'✅ Выполняется' if check_1 else '❌ Не выполняется'}",
            f"Для alpha=inf: {ineq_inf_left:.4f} <= {norm_A_E:.4f} <= {ineq_inf_right:.4f} -> {'✅ Выполняется' if check_inf else '❌ Не выполняется'}"
        ]
    }

    # --- 4. Матрица A (визуализация) ---
    # Округляем для красоты, если матрица большая - выводим только часть
    A_display = A
    if n > 10:
        A_display = A[:10, :10] # первые 10х10
        mat_warning = " (показаны первые 10x10)"
    else:
        mat_warning = ""

    matrix_block = {
        "type": "matrix",
        "title": f"Матрица A (n={n}){mat_warning}",
        "matrices": [
            {
                "label": "A",
                "data": np.round(A_display, 2).tolist(),
                "highlight": matrix_mode == "a_ij = min(i,j)"
            }
        ]
    }

    # --- 5. Число обусловленности ---
    # cond(A) = ||A|| * ||A^-1||
    # Для матрицы a_ij = min(i,j) она плохо обусловлена при больших n
    # Если матрица вырождена, обратной нет. numpy.linalg.cond может выдать inf.
    
    cond_1 = np.linalg.cond(A, 1)
    cond_inf = np.linalg.cond(A, np.inf)
    cond_E = np.linalg.cond(A, 'fro')
    cond_np = npla.cond(A) # По умолчанию 2-норма

    cond_table = {
        "type": "table",
        "title": "Число обусловленности cond(A) = ||A|| * ||A^-1||",
        "columns": [
            {"key": "norm", "label": "Норма"},
            {"key": "cond_val", "label": "Число обусловленности"}
        ],
        "rows": [
            {"norm": "||A||_1 * ||A^-1||_1", "cond_val": f"{cond_1:.4e}"},
            {"norm": "||A||_inf * ||A^-1||_inf", "cond_val": f"{cond_inf:.4e}"},
            {"norm": "||A||_E * ||A^-1||_E", "cond_val": f"{cond_E:.4e}"},
            {"norm": "numpy.linalg.cond (2-норма)", "cond_val": f"{cond_np:.4e}"}
        ]
    }

    # Собираем все блоки в один ответ
    return {
        "ok": True,
        "blocks": [
            vec_table,
            matrix_block,
            mat_table,
            ineq_list,
            cond_table
        ]
    }