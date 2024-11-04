import numpy as np
import time
import concurrent.futures


# Generación de matrices de tamaño grande con valores aleatorios de 6 dígitos
#def generate_large_matrix(n):
 #   return np.random.randint(100000, 999999, size=(n, n), dtype=np.int64)

# Cargar matriz desde archivo en función del tamaño
def load_matrix_from_file(prefix, size):
    filename = f"matriz_{prefix}{size}.txt"
    return np.loadtxt(filename, dtype=np.int64)


# 1. NaivLoopUnrollingTwo
def naiv_loop_unrolling_two(A, B):
    n = len(A)
    C = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            for k in range(0, n, 2):
                C[i, j] += A[i, k] * B[k, j]
                if k + 1 < n:
                    C[i, j] += A[i, k + 1] * B[k + 1, j]
    return C

# 2. NaivLoopUnrollingFour
def naiv_loop_unroll_four(A, B):
    n = len(A)
    C = np.zeros((n, n), dtype=int)
    for i in range(0, n, 4):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
                if i + 1 < n:
                    C[i + 1, j] += A[i + 1, k] * B[k, j]
                if i + 2 < n:
                    C[i + 2, j] += A[i + 2, k] * B[k, j]
                if i + 3 < n:
                    C[i + 3, j] += A[i + 3, k] * B[k, j]
    return C

# 3. WinogradOriginal
def winograd_original(A, B):
    n = len(A)
    C = np.zeros((n, n), dtype=int)
    row_factor = [sum(A[i][2 * j] * A[i][2 * j + 1] for j in range(n // 2)) for i in range(n)]
    col_factor = [sum(B[2 * i][j] * B[2 * i + 1][j] for i in range(n // 2)) for j in range(n)]
    for i in range(n):
        for j in range(n):
            C[i][j] = -row_factor[i] - col_factor[j]
            for k in range(n // 2):
                C[i][j] += (A[i][2 * k] + B[2 * k + 1][j]) * (A[i][2 * k + 1] + B[2 * k][j])
    if n % 2 == 1:
        for i in range(n):
            for j in range(n):
                C[i][j] += A[i][n - 1] * B[n - 1][j]
    return C

# 4. Winograd Scaled
def winograd_scaled(A, B):
        # Tamaños de las matrices
        m = len(A)
        n = len(A[0])
        p = len(B[0])

        # Paso 1: Calcular los vectores de multiplicación intermedios
        # Vector de producto de filas de A
        row_factor = [0] * m
        for i in range(m):
            row_factor[i] = sum(A[i][j] * A[i][j + 1] for j in range(0, n - 1, 2))

        # Vector de producto de columnas de B
        col_factor = [0] * p
        for j in range(p):
            col_factor[j] = sum(B[i][j] * B[i + 1][j] for i in range(0, n - 1, 2))

        # Paso 2: Calcular los valores de la matriz resultado C usando los factores
        C = [[0] * p for _ in range(m)]
        for i in range(m):
            for j in range(p):
                # Calcular el producto para la celda C[i][j]
                C[i][j] = -row_factor[i] - col_factor[j]
                for k in range(0, n - 1, 2):
                    C[i][j] += (A[i][k] + B[k + 1][j]) * (A[i][k + 1] + B[k][j])

                # Si n es impar, se necesita un ajuste adicional
                if n % 2 == 1:
                    C[i][j] += A[i][n - 1] * B[n - 1][j]

        return C

# 5. StrassenNaiv
def strassen_naiv(A, B):
    n = len(A)
    if n == 1:
        return A * B
    if n % 2 != 0:
        raise ValueError("El tamaño de la matriz debe ser una potencia de 2 para Strassen")
    mid = n // 2
    A11, A12, A21, A22 = A[:mid, :mid], A[:mid, mid:], A[mid:, :mid], A[mid:, mid:]
    B11, B12, B21, B22 = B[:mid, :mid], B[:mid, mid:], B[mid:, :mid], B[mid:, mid:]
    M1 = strassen_naiv(A11 + A22, B11 + B22)
    M2 = strassen_naiv(A21 + A22, B11)
    M3 = strassen_naiv(A11, B12 - B22)
    M4 = strassen_naiv(A22, B21 - B11)
    M5 = strassen_naiv(A11 + A12, B22)
    M6 = strassen_naiv(A21 - A11, B11 + B12)
    M7 = strassen_naiv(A12 - A22, B21 + B22)
    C = np.zeros((n, n), dtype=int)
    C[:mid, :mid] = M1 + M4 - M5 + M7
    C[:mid, mid:] = M3 + M5
    C[mid:, :mid] = M2 + M4
    C[mid:, mid:] = M1 - M2 + M3 + M6
    return C

# 6. Strassen-Winograd
def strassen_winograd(A, B):
    n = len(A)
    C = np.zeros((n, n), dtype=int)
    if n == 1:
        C[0, 0] = A[0, 0] * B[0, 0]
        return C
    if n % 2 != 0:
        raise ValueError("Strassen-Winograd necesita matrices de tamaño potencia de 2")

    mid = n // 2
    A11, A12, A21, A22 = A[:mid, :mid], A[:mid, mid:], A[mid:, :mid], A[mid:, mid:]
    B11, B12, B21, B22 = B[:mid, :mid], B[:mid, mid:], B[mid:, :mid], B[mid:, mid:]

    M1 = strassen_winograd(A11 + A22, B11 + B22)
    M2 = strassen_winograd(A21 + A22, B11)
    M3 = strassen_winograd(A11, B12 - B22)
    M4 = strassen_winograd(A22, B21 - B11)
    M5 = strassen_winograd(A11 + A12, B22)
    M6 = strassen_winograd(A21 - A11, B11 + B12)
    M7 = strassen_winograd(A12 - A22, B21 + B22)

    C[:mid, :mid] = M1 + M4 - M5 + M7
    C[:mid, mid:] = M3 + M5
    C[mid:, :mid] = M2 + M4
    C[mid:, mid:] = M1 - M2 + M3 + M6
    return C

# 7. Sequential Block III.3
def sequential_block_3(A, B, block_size):
    n = len(A)
    C = np.zeros((n, n), dtype=int)
    for ii in range(0, n, block_size):
        for jj in range(0, n, block_size):
            for kk in range(0, n, block_size):
                for i in range(ii, min(ii + block_size, n)):
                    for j in range(jj, min(jj + block_size, n)):
                        for k in range(kk, min(kk + block_size, n)):
                            C[i, j] += A[i, k] * B[k, j]
    return C

# 8. Sequential Block IV.
def sequential_block(A, B, block_size):
    n = len(A)
    C = np.zeros((n, n), dtype=int)
    for ii in range(0, n, block_size):
        for jj in range(0, n, block_size):
            for kk in range(0, n, block_size):
                for i in range(ii, min(ii + block_size, n)):
                    for j in range(jj, min(jj + block_size, n)):
                        for k in range(kk, min(kk + block_size, n)):
                            C[i][j] += A[i][k] * B[k][j]
    return C

# 9. III.5 Enhanced Parallel Block
def enhanced_parallel_block_v2(A, B, block_size):
    n = len(A)
    C = np.zeros((n, n), dtype=int)

    def compute_block(ii, jj, kk):
        for i in range(ii, min(ii + block_size, n)):
            for j in range(jj, min(jj + block_size, n)):
                for k in range(kk, min(kk + block_size, n)):
                    C[i, j] += A[i, k] * B[k, j]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for ii in range(0, n, block_size):
            for jj in range(0, n, block_size):
                for kk in range(0, n, block_size):
                    futures.append(executor.submit(compute_block, ii, jj, kk))
        concurrent.futures.wait(futures)

    return C


# 10. Enhanced Parallel Block Multiplication
def enhanced_parallel_block(A, B, block_size):
    n = len(A)
    C = np.zeros((n, n), dtype=int)

    def compute_block(ii, jj, kk):
        for i in range(ii, min(ii + block_size, n)):
            for j in range(jj, min(jj + block_size, n)):
                for k in range(kk, min(kk + block_size, n)):
                    C[i][j] += A[i][k] * B[k][j]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for ii in range(0, n, block_size):
            for jj in range(0, n, block_size):
                for kk in range(0, n, block_size):
                    executor.submit(compute_block, ii, jj, kk)
    return C


# Función para medir tiempo de ejecución
def measure_time(func, A, B, block_size=None):
    start = time.perf_counter()  # Comienza la medición con alta precisión
    if block_size:
        result = func(A, B, block_size)
    else:
        result = func(A, B)
    end = time.perf_counter()  # Finaliza la medición
    return (end - start)  * 1000000000  # Tiempo en nanosegundos

# Función para agregar tiempos de ejecución en un archivo .txt
def save_and_display_results(filename, results, matrix_size):
    with open(filename, 'w') as f:
        for name, time_ns in results:
            line = f"Tiempo de ejecucion ({name}) con tamano {matrix_size}x{matrix_size}: {time_ns:.0f} ns\n"
            f.write(line)
            print(line.strip())

# Función para imprimir una matriz
def print_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{val:10d}" for val in row))
    print()

# Prueba de los algoritmos con matrices
matrix_size = 256 # Tamaño de la matriz
block_size = matrix_size//2 # Tamaño de bloque para los algoritmos de bloque
# Cargar matrices A y B desde archivos según el tamaño especificado
A = load_matrix_from_file('A', matrix_size)
B = load_matrix_from_file('B', matrix_size)

# Medición de tiempos
results = [
    ("NaivLoopUnrollingTwo", measure_time(naiv_loop_unrolling_two, A, B)),
    ("NaivLoopUnrollingFour", measure_time(naiv_loop_unroll_four, A, B)),
    ("WinogradOriginal", measure_time(winograd_original, A, B)),
    ("Winograd Scaled", measure_time(winograd_scaled, A, B)),
    ("StrassenNaiv", measure_time(strassen_naiv, A, B)),
    ("Strassen-Winograd", measure_time(strassen_winograd, A, B)),
    ("III.3 Sequential Block", measure_time(sequential_block_3, A, B, block_size)),
    ("IV.3 Sequential Block", measure_time(sequential_block, A, B, block_size)),
    ("III.5 Enhanced Parallel Block", measure_time(enhanced_parallel_block_v2, A, B, block_size)),
    ("IV.5 Enhanced Parallel Block", measure_time(enhanced_parallel_block, A, B, block_size)),    
]

# Guardar los resultados en un archivo .txt con el tamaño de la matriz
save_and_display_results("tiempos_ejecucion_Python.txt", results, matrix_size)