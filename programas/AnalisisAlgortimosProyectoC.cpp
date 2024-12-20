#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <functional>
#include <fstream>
#include <thread>
#include <iostream>
#include <cstdlib> 

using namespace std;
using namespace chrono;

// Función para llenar la matriz con números aleatorios de 6 dígitos
//void fillMatrix(vector<vector<int>>& matrix, int n) {
//    random_device rd;
//    mt19937 gen(rd());
//    uniform_int_distribution<> dis(100000, 999999);

//    for (int i = 0; i < n; ++i)
//        for (int j = 0; j < n; ++j)
//            matrix[i][j] = dis(gen);
//}

//void saveMatrixToFile(const vector<vector<int>>& matrix, const string& filename) {
 //   ofstream file(filename);
 //   if (file.is_open()) {
  //      int n = matrix.size();
  //     for (int i = 0; i < n; ++i) {
    //        for (int j = 0; j < n; ++j) {
    //            file << matrix[i][j] << " ";
     //       }
     //       file << endl;  // Nueva línea después de cada fila
     //   }
     //   file.close();
     //   cout << "Matriz guardada en " << filename << endl;
   // }
   // else {
    //    cerr << "No se pudo abrir el archivo " << filename << endl;
    //}
//}

// Inicializa el archivo para un tamaño específico de matriz
void initializeLogFile(int matrixSize) {
    string filename = "tiemposDeEjecucion/tiemposDeEjecucionC++/tiempos_ejecucion_C_" + to_string(matrixSize) + ".txt";
    ofstream file(filename, ios::trunc); // 'ios::trunc' borra el contenido del archivo si existe
    if (!file.is_open()) {
        cerr << "No se pudo abrir el archivo " << filename << " para inicializarlo" << endl;
    } else {
        cout << "Archivo inicializado: " << filename << endl;
    }
    file.close();
}

// Registra el tiempo de ejecución en el archivo correspondiente al tamaño de la matriz
void logExecutionTime(const string& algorithmName, int matrixSize, long long duration) {
    string filename = "tiemposDeEjecucion/tiemposDeEjecucionC++/tiempos_ejecucion_C_" + to_string(matrixSize) + ".txt";
    ofstream file(filename, ios::app); // 'ios::app' añade datos al final del archivo
    if (file.is_open()) {
        file << "Tiempo de ejecucion (" << algorithmName << ") con tamano " << matrixSize << "x" << matrixSize << ": " << duration << " ns\n";
        file.close();
        cout << "Tiempo registrado en " << filename << endl;
    } else {
        cerr << "No se pudo abrir el archivo " << filename << endl;
    }
}

bool loadMatrixFromFile(vector<vector<int>>& matrix, const string& filename) {
    ifstream file(filename);
    if (file.is_open()) {
        int n = matrix.size();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                file >> matrix[i][j];
            }
        }
        file.close();
        cout << "Matriz cargada desde " << filename << endl;
        return true;
    }
    else {
        cerr << "No se pudo abrir el archivo " << filename << endl;
        return false;
    }
}

// Algoritmo 1: NaivLoopUnrollingTwo
void NaivLoopUnrollingTwo(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = 0; // Inicializar el valor de C[i][j]
            for (int k = 0; k < n; k += 2) {
                // Desenrollado de bucle para procesar dos elementos por iteración
                C[i][j] += A[i][k] * B[k][j];
                if (k + 1 < n) {
                    C[i][j] += A[i][k + 1] * B[k + 1][j];
                }
            }
        }
    }
}

// Algoritmo 2: NaivLoopUnrollingFour
void naivLoopUnrollingFour(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n) {
    for (int i = 0; i < n; i += 4)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
                if (i + 1 < n) C[i + 1][j] += A[i + 1][k] * B[k][j];
                if (i + 2 < n) C[i + 2][j] += A[i + 2][k] * B[k][j];
                if (i + 3 < n) C[i + 3][j] += A[i + 3][k] * B[k][j];
            }
}

// Algoritmo 3: WinogradOriginal
void WinogradOriginal(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n) {
    vector<int> rowFactor(n, 0);
    vector<int> colFactor(n, 0);

    // Calcular factores de las filas de A
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n / 2; ++j) {
            rowFactor[i] += A[i][2 * j] * A[i][2 * j + 1];
        }
    }

    // Calcular factores de las columnas de B
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n / 2; ++i) {
            colFactor[j] += B[2 * i][j] * B[2 * i + 1][j];
        }
    }

    // Calcular los elementos de C usando los factores calculados
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = -rowFactor[i] - colFactor[j];
            for (int k = 0; k < n / 2; ++k) {
                C[i][j] += (A[i][2 * k] + B[2 * k + 1][j]) * (A[i][2 * k + 1] + B[2 * k][j]);
            }
        }
    }
}

void winogradScaled(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n) {
    // Tamaños de las matrices
    int m = A.size();
    int p = B[0].size();

    // Paso 1: Calcular los vectores de multiplicación intermedios
    // Vector de producto de filas de A
    vector<int> row_factor(m, 0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n - 1; j += 2) {
            row_factor[i] += A[i][j] * A[i][j + 1];
        }
    }

    // Vector de producto de columnas de B
    vector<int> col_factor(p, 0);
    for (int j = 0; j < p; ++j) {
        for (int i = 0; i < n - 1; i += 2) {
            col_factor[j] += B[i][j] * B[i + 1][j];
        }
    }

    // Paso 2: Calcular los valores de la matriz resultado C usando los factores
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            // Calcular el producto para la celda C[i][j]
            C[i][j] = -row_factor[i] - col_factor[j];
            for (int k = 0; k < n - 1; k += 2) {
                C[i][j] += (A[i][k] + B[k + 1][j]) * (A[i][k + 1] + B[k][j]);
            }

            // Si n es impar, se necesita un ajuste adicional
            if (n % 2 == 1) {
                C[i][j] += A[i][n - 1] * B[n - 1][j];
            }
        }
    }
}

// Algoritmo 5: StrassenNaiv
// Función para sumar dos matrices
vector<vector<int>> add1(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int n = A.size();
    vector<vector<int>> C(n, vector<int>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}
// Función para restar dos matrices
vector<vector<int>> subtract(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int n = A.size();
    vector<vector<int>> C(n, vector<int>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
}

void StrassenNaiv(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n) {
    if (n == 1) {
        C[0][0] = A[0][0] * B[0][0];
        return;
    }

    // Dividir matrices en submatrices
    int newSize = n / 2;
    vector<vector<int>> A11(newSize, vector<int>(newSize));
    vector<vector<int>> A12(newSize, vector<int>(newSize));
    vector<vector<int>> A21(newSize, vector<int>(newSize));
    vector<vector<int>> A22(newSize, vector<int>(newSize));

    vector<vector<int>> B11(newSize, vector<int>(newSize));
    vector<vector<int>> B12(newSize, vector<int>(newSize));
    vector<vector<int>> B21(newSize, vector<int>(newSize));
    vector<vector<int>> B22(newSize, vector<int>(newSize));

    for (int i = 0; i < newSize; ++i) {
        for (int j = 0; j < newSize; ++j) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + newSize];
            A21[i][j] = A[i + newSize][j];
            A22[i][j] = A[i + newSize][j + newSize];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + newSize];
            B21[i][j] = B[i + newSize][j];
            B22[i][j] = B[i + newSize][j + newSize];
        }
    }

    // Calcular productos usando StrassenNaiv
    vector<vector<int>> M1(newSize, vector<int>(newSize));
    vector<vector<int>> M2(newSize, vector<int>(newSize));
    vector<vector<int>> M3(newSize, vector<int>(newSize));
    vector<vector<int>> M4(newSize, vector<int>(newSize));
    vector<vector<int>> M5(newSize, vector<int>(newSize));
    vector<vector<int>> M6(newSize, vector<int>(newSize));
    vector<vector<int>> M7(newSize, vector<int>(newSize));

    StrassenNaiv(add1(A11, A22), add1(B11, B22), M1, newSize);
    StrassenNaiv(add1(A21, A22), B11, M2, newSize);
    StrassenNaiv(A11, subtract(B12, B22), M3, newSize);
    StrassenNaiv(A22, subtract(B21, B11), M4, newSize);
    StrassenNaiv(add1(A11, A12), B22, M5, newSize);
    StrassenNaiv(subtract(A21, A11), add1(B11, B12), M6, newSize);
    StrassenNaiv(subtract(A12, A22), add1(B21, B22), M7, newSize);

    // Combinar los resultados en la matriz C
    for (int i = 0; i < newSize; ++i) {
        for (int j = 0; j < newSize; ++j) {
            C[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
            C[i][j + newSize] = M3[i][j] + M5[i][j];
            C[i + newSize][j] = M2[i][j] + M4[i][j];
            C[i + newSize][j + newSize] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
        }
    }
}

// Algoritmo 6: Strassen-Winograd (Versión simplificada)
void add(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int size) {
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            C[i][j] = A[i][j] + B[i][j];
}

void subtract(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int size) {
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            C[i][j] = A[i][j] - B[i][j];
}
void strassenWinograd(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n) {
    if (n <= 2) {
        // Condición base: multiplicación convencional para matrices pequeñas
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < n; ++k)
                    C[i][j] += A[i][k] * B[k][j];
        return;
    }

    int newSize = n / 2;
    vector<vector<int>> 
        A11(newSize, vector<int>(newSize)), A12(newSize, vector<int>(newSize)), 
        A21(newSize, vector<int>(newSize)), A22(newSize, vector<int>(newSize)),
        B11(newSize, vector<int>(newSize)), B12(newSize, vector<int>(newSize)), 
        B21(newSize, vector<int>(newSize)), B22(newSize, vector<int>(newSize)),
        C11(newSize, vector<int>(newSize)), C12(newSize, vector<int>(newSize)), 
        C21(newSize, vector<int>(newSize)), C22(newSize, vector<int>(newSize)),
        M1(newSize, vector<int>(newSize)), M2(newSize, vector<int>(newSize)), 
        M3(newSize, vector<int>(newSize)), M4(newSize, vector<int>(newSize)), 
        T1(newSize, vector<int>(newSize)), T2(newSize, vector<int>(newSize));

    // Dividir las matrices en submatrices
    for (int i = 0; i < newSize; ++i) {
        for (int j = 0; j < newSize; ++j) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + newSize];
            A21[i][j] = A[i + newSize][j];
            A22[i][j] = A[i + newSize][j + newSize];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + newSize];
            B21[i][j] = B[i + newSize][j];
            B22[i][j] = B[i + newSize][j + newSize];
        }
    }

    // Calcular matrices M1 a M4 (combinaciones simplificadas de Strassen-Winograd)
    add(A11, A22, T1, newSize); // T1 = A11 + A22
    add(B11, B22, T2, newSize); // T2 = B11 + B22
    strassenWinograd(T1, T2, M1, newSize); // M1 = (A11 + A22) * (B11 + B22)

    add(A21, A22, T1, newSize); // T1 = A21 + A22
    strassenWinograd(T1, B11, M2, newSize); // M2 = (A21 + A22) * B11

    subtract(B12, B22, T2, newSize); // T2 = B12 - B22
    strassenWinograd(A11, T2, M3, newSize); // M3 = A11 * (B12 - B22)

    subtract(B21, B11, T2, newSize); // T2 = B21 - B11
    strassenWinograd(A22, T2, M4, newSize); // M4 = A22 * (B21 - B11)

    // Calcular submatrices C11, C12, C21 y C22 combinando M1, M2, M3 y M4
    add(M1, M4, T1, newSize);
    subtract(T1, M2, T2, newSize);
    add(T2, M3, C11, newSize); // C11 = M1 + M4 - M2 + M3

    add(M1, M3, C12, newSize); // C12 = M1 + M3

    add(M2, M4, C21, newSize); // C21 = M2 + M4

    subtract(M1, M3, T1, newSize);
    add(T1, M2, T2, newSize);
    add(T2, M4, C22, newSize); // C22 = M1 - M3 + M2 + M4

    // Combinar submatrices en C
    for (int i = 0; i < newSize; ++i) {
        for (int j = 0; j < newSize; ++j) {
            C[i][j] = C11[i][j];
            C[i][j + newSize] = C12[i][j];
            C[i + newSize][j] = C21[i][j];
            C[i + newSize][j + newSize] = C22[i][j];
        }
    }
}

// Algoritmo 7: III.3 Sequential Block 
void sequentialBlock3(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n, int blockSize) {
    // Inicializa la matriz C en cero
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = 0;
        }
    }

    // Multiplicación en bloques
    for (int i = 0; i < n; i += blockSize) {
        for (int j = 0; j < n; j += blockSize) {
            // Multiplicación de bloques
            for (int k = 0; k < n; k += blockSize) {
                for (int ii = i; ii < min(i + blockSize, n); ++ii) {
                    for (int jj = j; jj < min(j + blockSize, n); ++jj) {
                        for (int kk = k; kk < min(k + blockSize, n); ++kk) {
                            C[ii][jj] += A[ii][kk] * B[kk][jj];
                        }
                    }
                }
            }
        }
    }
}

// Algoritmo 8: IV.3 Sequential Block
void sequentialBlock(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n, int blockSize) {
    for (int ii = 0; ii < n; ii += blockSize)
        for (int jj = 0; jj < n; jj += blockSize)
            for (int kk = 0; kk < n; kk += blockSize)
                for (int i = ii; i < min(ii + blockSize, n); ++i)
                    for (int j = jj; j < min(jj + blockSize, n); ++j)
                        for (int k = kk; k < min(kk + blockSize, n); ++k)
                            C[i][j] += A[i][k] * B[k][j];
}



// Algoritmo 9: III.5 Enhanced Parallel Block
const int MAX_THREADS = 8;  // Define el número máximo de hilos que se utilizarán

// Función para multiplicación de bloques en paralelo
void blockMultiply(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n, int blockSize, int startRow, int endRow) {
    for (int ii = startRow; ii < endRow; ii += blockSize) {
        for (int jj = 0; jj < n; jj += blockSize) {
            for (int kk = 0; kk < n; kk += blockSize) {
                for (int i = ii; i < min(ii + blockSize, n); ++i) {
                    for (int j = jj; j < min(jj + blockSize, n); ++j) {
                        int sum = 0;
                        for (int k = kk; k < min(kk + blockSize, n); ++k) {
                            sum += A[i][k] * B[k][j];
                        }
                        C[i][j] += sum;
                    }
                }
            }
        }
    }
}

void enhancedParallelBlock3(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n, int blockSize) {
    vector<thread> threads;
    int rowsPerThread = n / MAX_THREADS;

    for (int t = 0; t < MAX_THREADS; ++t) {
        int startRow = t * rowsPerThread;
        int endRow = (t == MAX_THREADS - 1) ? n : (t + 1) * rowsPerThread;
        
        threads.push_back(thread(blockMultiply, cref(A), cref(B), ref(C), n, blockSize, startRow, endRow));
    }

    for (auto& th : threads) {
        th.join();
    }
}
// Algoritmo 9: IV.5 Enhanced Parallel Block
void enhancedParallelBlock(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n, int blockSize) {
    vector<thread> threads;

    for (int ii = 0; ii < n; ii += blockSize) {
        for (int jj = 0; jj < n; jj += blockSize) {
            threads.emplace_back([&, ii, jj]() {
                for (int kk = 0; kk < n; kk += blockSize) {
                    for (int i = ii; i < min(ii + blockSize, n); ++i) {
                        for (int j = jj; j < min(jj + blockSize, n); ++j) {
                            for (int k = kk; k < min(kk + blockSize, n); ++k) {
                                C[i][j] += A[i][k] * B[k][j];
                            }
                        }
                    }
                }
            });
        }
    }

    // Espera a que todos los hilos terminen su ejecución
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
}

// Función para medir el tiempo de ejecución
void measureExecutionTime(void (*algorithm)(const vector<vector<int>>&, const vector<vector<int>>&, vector<vector<int>>&, int),
    const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n, const string& name) {
    auto start = high_resolution_clock::now();
    algorithm(A, B, C, n);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start).count();
    cout << "Tiempo de ejecucion (" << name << "): " << duration << " ns" << endl;
    // Guardar el tiempo de ejecución en el archivo
    logExecutionTime(name, n, duration);
}

// Sobrecarga de measureExecutionTime que acepta un argumento extra
void measureExecutionTime(const function<void(const vector<vector<int>>&, const vector<vector<int>>&, vector<vector<int>>&, int)>& algorithm,
    const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n, const string& name) {
    auto start = high_resolution_clock::now();
    algorithm(A, B, C, n);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start).count();
    cout << "Tiempo de ejecucion (" << name << "): " << duration << " ns" << endl;
    // Guardar el tiempo de ejecución en el archivo
    logExecutionTime(name, n, duration);

}


// Función para imprimir una matriz
void printMatrix(const vector<vector<int>>& matrix) {
    for (const auto& row : matrix) {          // Recorre cada fila de la matriz
        for (int element : row) {             // Recorre cada elemento de la fila
            cout << element << " ";           // Imprime el elemento seguido de un espacio
        }
        cout << endl;                         // Salto de línea al final de cada fila
    }
}

// Función principal
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Error: no se proporcionó el tamaño de la matriz." << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]); // Convierte el argumento a entero
    int blockSize = n/2;  // Tamaño de bloque para los algoritmos de bloques
    vector<vector<int>> A(n, vector<int>(n));
    vector<vector<int>> B(n, vector<int>(n));
    vector<vector<int>> C(n, vector<int>(n, 0));
    initializeLogFile(n);
    // Llenar matrices con valores aleatorios de 6 dígitos
    /*fillMatrix(A, n);
    fillMatrix(B, n);

    // Guardar las matrices A y B en archivos .txt
    saveMatrixToFile(A, "matriz_A2.txt");
    saveMatrixToFile(B, "matriz_B2.txt");*/

    // Construir los nombres de archivo basados en el tamaño de la matriz
    // Especificar las rutas relativas de las carpetas
    string filenameA = "matrices/matricesA/matriz_A" + to_string(n) + ".txt";
    string filenameB = "matrices/matricesB/matriz_B" + to_string(n) + ".txt";
    
    // Cargar las matrices desde los archivos
    if (!loadMatrixFromFile(A, filenameA) || !loadMatrixFromFile(B, filenameB)) {
        cerr << "Error al cargar las matrices." << endl;
        return 1;
    }
    
    // Ejecutar cada algoritmo y medir su tiempo de ejecución
    cout << "Ejecutando algoritmos de multiplicación de matrices para matrices de tamaño " << n << "x" << n << endl;

    measureExecutionTime(NaivLoopUnrollingTwo, A, B, C, n, "NaivLoopUnrollingTwo");
    fill(C.begin(), C.end(), vector<int>(n, 0));  // Reset de la matriz C

    measureExecutionTime(naivLoopUnrollingFour, A, B, C, n, "NaivLoopUnrollingFour");
    fill(C.begin(), C.end(), vector<int>(n, 0));  // Reset de la matriz C

    measureExecutionTime(WinogradOriginal, A, B, C, n, "WinogradOriginal");
    fill(C.begin(), C.end(), vector<int>(n, 0));  // Reset de la matriz C 

    measureExecutionTime(winogradScaled, A, B, C, n, "Winograd Scaled");
    fill(C.begin(), C.end(), vector<int>(n, 0));  // Reset de la matriz C

    measureExecutionTime(StrassenNaiv, A, B, C, n, "StrassenNaiv"); 
    fill(C.begin(), C.end(), vector<int>(n, 0));  // Reset de la matriz C  

    measureExecutionTime(strassenWinograd, A, B, C, n, "Strassen-Winograd");
    fill(C.begin(), C.end(), vector<int>(n, 0));  // Reset de la matriz C

    auto sequentialBlockLambda2 = [&](const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n) {
        sequentialBlock3(A, B, C, n, blockSize);
        };

    measureExecutionTime(sequentialBlockLambda2, A, B, C, n, "III.3 Sequential Block");
    fill(C.begin(), C.end(), vector<int>(n, 0));  // Reset de la matriz C

    auto sequentialBlockLambda = [&](const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n) {
        sequentialBlock(A, B, C, n, blockSize);
        };
    
    measureExecutionTime(sequentialBlockLambda, A, B, C, n, "IV.3 Sequential Block");
    fill(C.begin(), C.end(), vector<int>(n, 0));  // Reset de la matriz C

    auto enhancedParallelBlockLambda2 = [&](const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n) {
        enhancedParallelBlock3(A, B, C, n, blockSize);
        };
    measureExecutionTime(enhancedParallelBlockLambda2, A, B, C, n, "III.5 Enhanced Parallel Block");

    auto enhancedParallelBlockLambda = [&](const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n) {
        enhancedParallelBlock(A, B, C, n, blockSize);
        };

    measureExecutionTime(enhancedParallelBlockLambda, A, B, C, n, "IV.5 Enhanced Parallel Block");
    fill(C.begin(), C.end(), vector<int>(n, 0));  // Reset de la matriz C
    
    

      

    

    
    

    

    
    cout << "---------------------------------------------------------------------------------------" << endl;

    return 0;
}

