import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import subprocess
import os
import tkinter as tk
from tkinter import messagebox
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
def determine_global_time_unit(times_python, times_cpp):
    # Calcula el tiempo máximo global
    global_max_time = max(max(times_python, default=1), max(times_cpp, default=1))
    
    # Determina la unidad de tiempo y el factor de escala basado en el tiempo máximo
    if global_max_time >= 1e9:
        return "s", 1e9
    elif global_max_time >= 1e6:
        return "ms", 1e6
    elif global_max_time >= 1e3:
        return "μs", 1e3
    else:
        return "ns", 1

# Función para graficar los tiempos de ejecución con la unidad adecuada
def plot_execution_times(ax, filename, title, time_unit, scale_factor):
    algorithms, times, matrix_size = read_execution_times(filename)
    title_with_size = f"{title} - Tamaño de matriz: {matrix_size}x{matrix_size}" if matrix_size else title

    # Escala los tiempos utilizando el factor global
    scaled_times = [time / scale_factor for time in times] if times else []

    ax.clear()  # Limpia la gráfica antes de dibujarla nuevamente
    ax.barh(algorithms, scaled_times, color='skyblue')
    ax.set_xlabel(f"Tiempo de ejecución ({time_unit})", fontsize=10)
    ax.set_ylabel("Algoritmo", fontsize=10)
    ax.set_title(title_with_size, fontsize=12)
    ax.set_xlim(0, max(scaled_times) * 1.1 if scaled_times else 1)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

# Función para actualizar las gráficas cuando se selecciona un tamaño de matriz
def update_plot(size, ax_python, ax_cpp):
    selected_size = int(size)  # Convierte el tamaño seleccionado a entero
    filename_python = f"tiemposDeEjecucion/tiemposDeEjecucionPython/tiempos_ejecucion_Python_{selected_size}.txt"
    filename_c = f"tiemposDeEjecucion/tiemposDeEjecucionC++/tiempos_ejecucion_C_{selected_size}.txt"

    # Leer datos de ambos archivos
    _, times_python, _ = read_execution_times(filename_python)
    _, times_cpp, _ = read_execution_times(filename_c)

    # Determinar la unidad de tiempo y el factor de escala globales
    time_unit, scale_factor = determine_global_time_unit(times_python, times_cpp)

    # Actualizar ambas gráficas utilizando la escala global
    plot_execution_times(ax_python, filename_python, "Tiempos de ejecución Python", time_unit, scale_factor)
    plot_execution_times(ax_cpp, filename_c, "Tiempos de ejecución C++", time_unit, scale_factor)

    plt.draw()  # Redibuja la figura


# Función para ejecutar los programas externos
def execute_programs(size):
    selected_size = int(size)
    python_script = r"C:\Users\Nicolas\Documents\PFAnalisisDeAlgoritmos\programas\AnalisisAlgoritmosProyectoPython.py"
    cpp_executable = r"C:\Users\Nicolas\Documents\PFAnalisisDeAlgoritmos\programas\AnalisisAlgortimosProyectoC.exe"  # Asegúrate de compilar el .cpp a .exe

    try:
        # Ejecutar el script de Python
        subprocess.run(["python", python_script, str(selected_size)], check=True)
        print(f"Ejecutado {python_script} con tamaño {selected_size}")

        # Ejecutar el programa C++
        subprocess.run([cpp_executable, str(selected_size)], check=True)
        print(f"Ejecutado {cpp_executable} con tamaño {selected_size}")

     # Si ambos programas se ejecutaron correctamente
        show_popup("Ejecutado", f"Ambos programas se ejecutaron con tamaño {size}")
    

    except FileNotFoundError as e:
        print(f"Error: {e}")
        show_popup("Error", f"Hubo un error en la ejecución de los programas. Detalles: {e}")
    except subprocess.CalledProcessError as e:
        print(f"Error en la ejecución: {e}")
        show_popup("Error", f"Hubo un error en la ejecución de los programas. Detalles: {e}")


def show_popup(title, message):
    root = tk.Tk()
    root.withdraw()  # No mostrar la ventana principal de tkinter
    messagebox.showinfo(title, message)  # Mostrar la ventana emergente
    
# Lista de tamaños de matriz disponibles
sizes = [2, 4, 8, 16, 32, 64, 128, 256]
initial_size = sizes[0]

# Configuración inicial de la figura y subgráficas
fig, (ax_python, ax_cpp) = plt.subplots(1, 2, figsize=(16, 8))
fig.subplots_adjust(bottom=0.4)  # Ajusta espacio inferior para el slider y botones

# Slider para seleccionar el tamaño de la matriz
slider_ax = fig.add_axes([0.25, 0.1, 0.5, 0.05], facecolor="lightgoldenrodyellow")
size_slider = Slider(slider_ax, "Tamaño de Matriz", valmin=min(sizes), valmax=max(sizes), valinit=initial_size, valstep=sizes)

# Botón para graficar
button_ax = fig.add_axes([0.25, 0.02, 0.2, 0.05])
button_graficar = Button(button_ax, "Graficar")

# Botón para ejecutar programas
button_exec_ax = fig.add_axes([0.55, 0.02, 0.2, 0.05])
button_ejecutar = Button(button_exec_ax, "Ejecutar Programas")

# Configurar eventos
size_slider.on_changed(lambda val: update_plot(val, ax_python, ax_cpp))
button_graficar.on_clicked(lambda event: update_plot(size_slider.val, ax_python, ax_cpp))
button_ejecutar.on_clicked(lambda event: execute_programs(size_slider.val))
plt.subplots_adjust(left=0.152, bottom=0.383, right=0.983, top=0.88, wspace=0.427, hspace=0.2)

# Gráfica inicial con el tamaño de matriz por defecto
update_plot(initial_size, ax_python, ax_cpp)
plt.show()
