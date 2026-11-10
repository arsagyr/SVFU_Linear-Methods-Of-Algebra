"""Lab5: Разреженные матрицы (CSR и CSIR) и их умножение на вектор."""

import numpy as np
import time

# ==========================================
# 1. Классы для CSR и CSIR
# ==========================================

class CSRMatrix:
    """Разреженная несимметричная матрица в формате CSR."""
    def __init__(self, data, indices, indptr, shape):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = shape
        self.nnz = len(data)

    @classmethod
    def from_dense(cls, A):
        n, m = A.shape
        data, indices, indptr = [], [], [0]
        for i in range(n):
            row = A[i]
            nz = np.nonzero(row)[0]
            for j in nz:
                data.append(row[j])
                indices.append(j)
            indptr.append(len(data))
        return cls(np.array(data, dtype=float), np.array(indices, dtype=int), 
                   np.array(indptr, dtype=int), A.shape)

    def dot(self, x):
        """Умножение матрицы на вектор."""
        n = self.shape[0]
        res = np.zeros(n)
        for i in range(n):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for k in range(start, end):
                res[i] += self.data[k] * x[self.indices[k]]
        return res


class CSIRMatrix:
    """Симметричная матрица с портретом (верхний треугольник) в формате CSIR."""
    def __init__(self, data, indices, indptr, shape):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = shape
        # nnz учитывает симметрию (кроме диагонали, которая считается один раз)
        self.nnz = len(data) * 2 - shape[0]

    @classmethod
    def from_dense(cls, A):
        n = A.shape[0]
        data, indices, indptr = [], [], [0]
        for i in range(n):
            row = A[i]
            # Берем только верхний треугольник (включая диагональ)
            nz = np.nonzero(row[i:])[0] + i
            for j in nz:
                data.append(row[j])
                indices.append(j)
            indptr.append(len(data))
        return cls(np.array(data, dtype=float), np.array(indices, dtype=int), 
                   np.array(indptr, dtype=int), A.shape)

    def dot(self, x):
        """Умножение симметричной матрицы на вектор."""
        n = self.shape[0]
        res = np.zeros(n)
        for i in range(n):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for k in range(start, end):
                j = self.indices[k]
                val = self.data[k]
                res[i] += val * x[j]
                if i != j:
                    # Симметричный элемент (нижний треугольник)
                    res[j] += val * x[i]
        return res


# ==========================================
# 2. Генераторы разреженных матриц
# ==========================================

def generate_sparse_unsymmetric(n):
    """Генерирует разреженную несимметричную матрицу (A)."""
    A = np.zeros((n, n))
    # Заполняем примерно 3% элементов
    nnz_per_row = max(1, int(n * 0.03))
    for i in range(n):
        # Главная диагональ
        A[i, i] = np.random.rand() * 10
        # Случайные элементы в строке
        cols = np.random.choice(n, size=nnz_per_row, replace=False)
        for j in cols:
            if i != j:
                A[i, j] = np.random.rand() * 10 - 5
    return A

def generate_sparse_symmetric(n):
    """Генерирует симметричную разреженную матрицу (B)."""
    A = np.zeros((n, n))
    nnz_per_row = max(1, int(n * 0.03))
    for i in range(n):
        A[i, i] = np.random.rand() * 10 + 1.0  # Ненулевая главная диагональ
        cols = np.random.choice(n, size=nnz_per_row, replace=False)
        for j in cols:
            if i != j:
                val = np.random.rand() * 10 - 5
                A[i, j] = val
                A[j, i] = val # Симметрия
    return A


# ==========================================
# 3. Главная функция
# ==========================================

def run_lab(n):
    try:
        n = int(n)
    except ValueError:
        return {"ok": False, "error": "n должно быть целым числом"}
    if not (100 <= n <= 5000):
        return {"ok": False, "error": "n должно быть от 100 до 5000"}

    blocks = []

    # ==========================================
    # ЗАМЕР ДЛЯ ФИКСИРОВАННОГО n
    # ==========================================
    np.random.seed(42)  # Для воспроизводимости
    
    # Генерируем вектор
    x = np.random.rand(n)
    
    # 1. Несимметричная матрица (CSR)
    A_dense = generate_sparse_unsymmetric(n)
    csr = CSRMatrix.from_dense(A_dense)
    
    # Умножение плотной
    t0 = time.perf_counter()
    res_dense = A_dense @ x
    t_dense = time.perf_counter() - t0
    
    # Умножение CSR
    t0 = time.perf_counter()
    res_csr = csr.dot(x)
    t_csr = time.perf_counter() - t0
    
    # 2. Симметричная матрица (CSIR)
    B_dense = generate_sparse_symmetric(n)
    csir = CSIRMatrix.from_dense(B_dense)
    
    # Умножение плотной B
    t0 = time.perf_counter()
    res_dense_b = B_dense @ x
    t_dense_b = time.perf_counter() - t0
    
    # Умножение CSIR
    t0 = time.perf_counter()
    res_csir = csir.dot(x)
    t_csir = time.perf_counter() - t0

    # Проверка правильности (должны совпадать с точностью до 1e-10)
    err_csr = np.linalg.norm(res_dense - res_csr)
    err_csir = np.linalg.norm(res_dense_b - res_csir)

    # Таблица с результатами для текущего n
    table1 = {
        "type": "table",
        "title": f"Время умножения матрицы на вектор (n={n})",
        "columns": [
            {"key": "method", "label": "Формат / Матрица"},
            {"key": "nnz", "label": "Ненулевых элементов (nnz)"},
            {"key": "time", "label": "Время (сек)"},
            {"key": "error", "label": "Ошибка (норма разности)"}
        ],
        "rows": [
            {"method": "Плотная (A)", "nnz": n*n, "time": f"{t_dense:.5f}", "error": "0 (эталон)"},
            {"method": "CSR (A)", "nnz": csr.nnz, "time": f"{t_csr:.5f}", "error": f"{err_csr:.2e}"},
            {"method": "Плотная (B)", "nnz": n*n, "time": f"{t_dense_b:.5f}", "error": "0 (эталон)"},
            {"method": "CSIR (B)", "nnz": csir.nnz, "time": f"{t_csir:.5f}", "error": f"{err_csir:.2e}"}
        ]
    }
    blocks.append(table1)

    # ==========================================
    # ГРАФИК ЗАВИСИМОСТИ ВРЕМЕНИ ОТ n
    # ==========================================
    # Ограничим количество точек для графика (не более 15)
    num_points = 10
    ns = np.linspace(100, n, num_points, dtype=int)
    # Убираем дубликаты, если n маленькое
    ns = np.unique(ns)
    
    times_dense = []
    times_csr = []
    times_csir = []
    
    for size in ns:
        np.random.seed(42)
        x_loop = np.random.rand(size)
        
        # Строим A и B
        A_l = generate_sparse_unsymmetric(size)
        B_l = generate_sparse_symmetric(size)
        
        csr_l = CSRMatrix.from_dense(A_l)
        csir_l = CSIRMatrix.from_dense(B_l)
        
        # Замеряем плотную (A)
        t0 = time.perf_counter()
        A_l @ x_loop
        times_dense.append(time.perf_counter() - t0)
        
        # Замеряем CSR
        t0 = time.perf_counter()
        csr_l.dot(x_loop)
        times_csr.append(time.perf_counter() - t0)
        
        # Замеряем CSIR
        t0 = time.perf_counter()
        csir_l.dot(x_loop)
        times_csir.append(time.perf_counter() - t0)

    chart = {
        "type": "chart",
        "title": "Зависимость времени умножения от размера матрицы",
        "chart_type": "line",
        "labels": [str(size) for size in ns],
        "datasets": [
            {
                "label": "Плотная матрица",
                "data": times_dense,
                "borderColor": "#e74c3c",
                "tension": 0.1
            },
            {
                "label": "CSR (несимметр.)",
                "data": times_csr,
                "borderColor": "#3498db",
                "tension": 0.1
            },
            {
                "label": "CSIR (симметр.)",
                "data": times_csir,
                "borderColor": "#2ecc71",
                "tension": 0.1
            }
        ]
    }
    blocks.append(chart)

    # Добавим примечание
    blocks.append({
        "type": "list",
        "title": "Примечание",
        "items": [
            "Формат CSR хранит только ненулевые элементы (values), их индексы столбцов (indices) и указатели на начало строк (indptr).",
            "Формат CSIR (модификация) хранит только верхний треугольник (включая диагональ) для симметричных матриц, что экономит память в 2 раза.",
            "Плотная матрица тратит время на умножение нулевых элементов, поэтому на больших n разреженные форматы работают в разы быстрее."
        ]
    })

    return {"ok": True, "blocks": blocks}