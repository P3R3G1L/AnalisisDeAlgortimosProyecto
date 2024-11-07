import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# Función para leer datos del archivo .txt y extraer nombres de algoritmos y tiempos de ejecución
def read_execution_times(filename):
    algorithms = []
    times = []
    matrix_size = None

    try:
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith("Tiempo de ejecucion"):
                    parts = line.split(":")
                    # Extraer nombre del algoritmo y tamaño de matriz
                    name = parts[0].split("(")[1].split(")")[0]
                    matrix_size = int(parts[0].split("con tamano")[1].strip().split("x")[0])  # Tamaño de la matriz como entero
                    time_ns = int(parts[1].strip().replace(" ns", ""))

                    algorithms.append(name)
                    times.append(time_ns)
    except FileNotFoundError:
        print(f"Archivo {filename} no encontrado.")
    
    return algorithms, times, matrix_size

# Función para determinar la unidad de tiempo y el escalado en función del valor máximo
def determine_time_unit(max_time):
    if max_time >= 1e9:
        return "s", 1e9
    elif max_time >= 1e6:
        return "ms", 1e6
    elif max_time >= 1e3:
        return "μs", 1e3
    else:
        return "ns", 1

# Función para graficar los tiempos de ejecución con la unidad adecuada
def plot_execution_times(ax, filename, title):
    algorithms, times, matrix_size = read_execution_times(filename)
    title_with_size = f"{title} - Tamaño de matriz: {matrix_size}x{matrix_size}" if matrix_size else title

    max_time = max(times) if times else 1
    time_unit, scale_factor = determine_time_unit(max_time)
    scaled_times = [time / scale_factor for time in times] if times else []

    ax.clear()  # Limpia la gráfica antes de dibujarla nuevamente
    ax.barh(algorithms, scaled_times, color='skyblue')
    ax.set_xlabel(f"Tiempo de ejecución ({time_unit})", fontsize=10)
    ax.set_ylabel("Algoritmo", fontsize=10)
    ax.set_title(title_with_size, fontsize=12)
    ax.set_xlim(0, max(scaled_times) * 1.1 if scaled_times else 1)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

# Función para actualizar la gráfica cuando se selecciona un tamaño de matriz
def update_plot(size, ax_python, ax_cpp):
    selected_size = int(size)  # Convierte el tamaño seleccionado a entero
    filename_python = f"tiempos_ejecucion_Python_{selected_size}.txt"
    filename_c = f"tiempos_ejecucion_C_{selected_size}.txt"

    # Actualiza ambas gráficas con el tamaño seleccionado
    plot_execution_times(ax_python, filename_python, "Tiempos de ejecución Python")
    plot_execution_times(ax_cpp, filename_c, "Tiempos de ejecución C++")
    
    plt.draw()  # Redibuja la figura

# Lista de tamaños de matriz disponibles
sizes = [2, 4, 8, 16, 32, 64, 128, 256]
initial_size = sizes[0]

# Configuración inicial de la figura y subgráficas
fig, (ax_python, ax_cpp) = plt.subplots(1, 2, figsize=(16, 8))
fig.subplots_adjust(bottom=0.3)  # Ajusta espacio inferior para el slider

# Slider para seleccionar el tamaño de la matriz
slider_ax = fig.add_axes([0.25, 0.01, 0.5, 0.05], facecolor="lightgoldenrodyellow")
size_slider = Slider(slider_ax, "Tamaño de Matriz", valmin=min(sizes), valmax=max(sizes), valinit=initial_size, valstep=sizes)

# Configurar evento para actualizar la gráfica al cambiar el tamaño
size_slider.on_changed(lambda val: update_plot(val, ax_python, ax_cpp))

# Gráfica inicial con el tamaño de matriz por defecto
update_plot(initial_size, ax_python, ax_cpp)
# Ajusta el espaciado entre subgráficas para que no se solapen
plt.subplots_adjust(left=0.157, bottom=0.124, right=0.98, top=0.948, wspace=0.435, hspace=0.205)

plt.show()
