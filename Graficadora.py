import matplotlib.pyplot as plt

# Función para leer datos del archivo .txt y extraer nombres de algoritmos, tamaños de matrices y tiempos de ejecución
def read_execution_times(filename):
    algorithms = []
    times = []
    matrix_size = None

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("Tiempo de ejecucion"):
                parts = line.split(":")
                # Extraer nombre del algoritmo y tamaño de matriz
                name = parts[0].split("(")[1].split(")")[0]
                matrix_size = int(parts[0].split("con tamano")[1].strip().split("x")[0])  # Captura el tamaño de la matriz como entero
                time_ns = int(parts[1].strip().replace(" ns", ""))

                algorithms.append(name)
                times.append(time_ns)

    return algorithms, times, matrix_size

# Función para graficar los tiempos de ejecución y ajustar la unidad según el tamaño de la matriz
def plot_execution_times(filename, title):
    algorithms, times, matrix_size = read_execution_times(filename)
    title_with_size = f"{title} - Tamaño de matriz: {matrix_size}x{matrix_size}"  # Agregar tamaño al título

    # Ajuste de unidad y límites del eje X basado en el tamaño de la matriz
    if matrix_size <= 8:
        # Mostrar en nanosegundos
        times = [time for time in times]
        unit = "ns"
        xlim = max(times) * 1.1
    else:
        # Mostrar en segundos
        times = [time / 1e9 for time in times]  # Convertir a segundos
        unit = "s"
        xlim = max(times) * 1.1

    plt.barh(algorithms, times, color='skyblue')
    plt.xlabel(f"Tiempo de ejecución ({unit})", fontsize=10)
    plt.ylabel("Algoritmo", fontsize=10)
    plt.title(title_with_size, fontsize=12)
    plt.xlim(0, xlim)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

# Crea la figura y las subgráficas con ajustes de tamaño y espaciado
plt.figure(figsize=(16, 8))

# Gráfica para tiempos de ejecución en Python
plt.subplot(1, 2, 1)
plot_execution_times("tiempos_ejecucion_Python.txt", "Tiempos de ejecución Python")

# Gráfica para tiempos de ejecución en C++
plt.subplot(1, 2, 2)
plot_execution_times("tiempos_ejecucion_C.txt", "Tiempos de ejecución C++")

# Ajusta el espaciado entre subgráficas para que no se solapen
plt.subplots_adjust(left=0.157, bottom=0.068, right=0.98, top=0.948, wspace=0.435, hspace=0.205)

plt.show()
