#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <functional>
#include <fstream>
#include <thread>

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

void logExecutionTime(const string& algorithmName, int matrixSize, long long duration) {
    ofstream file("tiempos_ejecucion.txt", ios::app);  // ios::app permite añadir datos al final del archivo
    if (file.is_open()) {
        file << "Tiempo de ejecución (" << algorithmName << ") con tamaño " << matrixSize << "x" << matrixSize << ": " << duration << " ns\n";
        file.close();
        cout << "Tiempo registrado en execution_times.txt" << endl;
    }
    else {
        cerr << "No se pudo abrir el archivo execution_times.txt" << endl;
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

// Algoritmo 1: Strassen-Winograd (Versión simplificada)
void strassenWinograd(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n) {
    if (n <= 2) {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < n; ++k)
                    C[i][j] += A[i][k] * B[k][j];
        return;
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

// Algoritmo 3: Winograd Scaled
void winogradScaled(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n) {
    vector<int> rowFactor(n, 0), colFactor(n, 0);

    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n / 2; ++k)
            rowFactor[i] += A[i][2 * k] * A[i][2 * k + 1];

    for (int j = 0; j < n; ++j)
        for (int k = 0; k < n / 2; ++k)
            colFactor[j] += B[2 * k][j] * B[2 * k + 1][j];

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = -rowFactor[i] - colFactor[j];
            for (int k = 0; k < n / 2; ++k) {
                C[i][j] += (A[i][2 * k] + B[2 * k + 1][j]) * (A[i][2 * k + 1] + B[2 * k][j]);
            }
        }
    }

    if (n % 2 == 1)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                C[i][j] += A[i][n - 1] * B[n - 1][j];
}

// Algoritmo 4: IV.3 Sequential Block
void sequentialBlock(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n, int blockSize) {
    for (int ii = 0; ii < n; ii += blockSize)
        for (int jj = 0; jj < n; jj += blockSize)
            for (int kk = 0; kk < n; kk += blockSize)
                for (int i = ii; i < min(ii + blockSize, n); ++i)
                    for (int j = jj; j < min(jj + blockSize, n); ++j)
                        for (int k = kk; k < min(kk + blockSize, n); ++k)
                            C[i][j] += A[i][k] * B[k][j];
}

// Algoritmo 5: IV.5 Enhanced Parallel Block 
void enhancedParallelBlock(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n, int blockSize) {
    sequentialBlock(A, B, C, n, blockSize);  // Implementación básica reutilizando sequential block
}

// Algoritmo 6: NaivLoopUnrollingTwo
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

// Algoritmo 7: WinogradOriginal
void WinogradOriginal(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n) {
    // Calcular los productos parciales de las filas de A y columnas de B
    vector<int> rowFactor(n, 0);
    vector<int> colFactor(n, 0);

    // Calcular los factores de las filas para la matriz A
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n / 2; ++j) {
            rowFactor[i] += A[i][2 * j] * A[i][2 * j + 1];
        }
    }

    // Calcular los factores de las columnas para la matriz B
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n / 2; ++i) {
            colFactor[j] += B[2 * i][j] * B[2 * i + 1][j];
        }
    }

    // Calcular la matriz C usando los factores calculados
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = -rowFactor[i] - colFactor[j];
            for (int k = 0; k < n / 2; ++k) {
                C[i][j] += (A[i][2 * k] + B[2 * k + 1][j]) * (A[i][2 * k + 1] + B[2 * k][j]);
            }
        }
    }

}

// Algoritmo 8: StrassenNaiv
// Función para sumar dos matrices
vector<vector<int>> add(const vector<vector<int>>& A, const vector<vector<int>>& B) {
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

    StrassenNaiv(add(A11, A22), add(B11, B22), M1, newSize);
    StrassenNaiv(add(A21, A22), B11, M2, newSize);
    StrassenNaiv(A11, subtract(B12, B22), M3, newSize);
    StrassenNaiv(A22, subtract(B21, B11), M4, newSize);
    StrassenNaiv(add(A11, A12), B22, M5, newSize);
    StrassenNaiv(subtract(A21, A11), add(B11, B12), M6, newSize);
    StrassenNaiv(subtract(A12, A22), add(B21, B22), M7, newSize);

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


// Algoritmo 9: III.3 Sequential Block 
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


// Algoritmo 10: III.5 Enhanced Parallel Block
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

// Algoritmo 10: III.5 Enhanced Parallel Block
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

// Función para medir el tiempo de ejecución
void measureExecutionTime(void (*algorithm)(const vector<vector<int>>&, const vector<vector<int>>&, vector<vector<int>>&, int),
    const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n, const string& name) {
    auto start = high_resolution_clock::now();
    algorithm(A, B, C, n);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start).count();
    cout << "Tiempo de ejecución (" << name << "): " << duration << " ns" << endl;
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
    cout << "Tiempo de ejecución (" << name << "): " << duration << " ns" << endl;
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
int main() {
    int n = 256;  // Tamaño de la matriz
    int blockSize = n/2;  // Tamaño de bloque para los algoritmos de bloques
    vector<vector<int>> A(n, vector<int>(n));
    vector<vector<int>> B(n, vector<int>(n));
    vector<vector<int>> C(n, vector<int>(n, 0));

    // Llenar matrices con valores aleatorios de 6 dígitos
    /*fillMatrix(A, n);
    fillMatrix(B, n);

    // Guardar las matrices A y B en archivos .txt
    saveMatrixToFile(A, "matriz_A2.txt");
    saveMatrixToFile(B, "matriz_B2.txt");*/

    // Construir los nombres de archivo basados en el tamaño de la matriz
    string filenameA = "matriz_A" + to_string(n) + ".txt";
    string filenameB = "matriz_B" + to_string(n) + ".txt";
    
    if (!loadMatrixFromFile(A, filenameA) || !loadMatrixFromFile(B, filenameB)) {
        cerr << "Error al cargar las matrices." << endl;
        return 1;
    }
    
    // Ejecutar cada algoritmo y medir su tiempo de ejecución
    cout << "Ejecutando algoritmos de multiplicación de matrices para matrices de tamaño " << n << "x" << n << endl;

    measureExecutionTime(strassenWinograd, A, B, C, n, "Strassen-Winograd");
    fill(C.begin(), C.end(), vector<int>(n, 0));  // Reset de la matriz C

    measureExecutionTime(naivLoopUnrollingFour, A, B, C, n, "NaivLoopUnrollingFour");
    fill(C.begin(), C.end(), vector<int>(n, 0));  // Reset de la matriz C

    measureExecutionTime(winogradScaled, A, B, C, n, "Winograd Scaled");
    fill(C.begin(), C.end(), vector<int>(n, 0));  // Reset de la matriz C

    auto sequentialBlockLambda = [&](const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n) {
        sequentialBlock(A, B, C, n, blockSize);
        };
    auto enhancedParallelBlockLambda = [&](const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n) {
        enhancedParallelBlock(A, B, C, n, blockSize);
        };

    measureExecutionTime(sequentialBlockLambda, A, B, C, n, "IV.3 Sequential Block");
    fill(C.begin(), C.end(), vector<int>(n, 0));  // Reset de la matriz C

    measureExecutionTime(enhancedParallelBlockLambda, A, B, C, n, "IV.5 Enhanced Parallel Block");
    fill(C.begin(), C.end(), vector<int>(n, 0));  // Reset de la matriz C
    
    measureExecutionTime(NaivLoopUnrollingTwo, A, B, C, n, "NaivLoopUnrollingTwo");
    fill(C.begin(), C.end(), vector<int>(n, 0));  // Reset de la matriz C

    measureExecutionTime(WinogradOriginal, A, B, C, n, "WinogradOriginal");
    fill(C.begin(), C.end(), vector<int>(n, 0));  // Reset de la matriz C   

    measureExecutionTime(StrassenNaiv, A, B, C, n, "StrassenNaiv"); 
    fill(C.begin(), C.end(), vector<int>(n, 0));  // Reset de la matriz C   

    auto sequentialBlockLambda2 = [&](const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n) {
        sequentialBlock3(A, B, C, n, blockSize);
        };
    auto enhancedParallelBlockLambda2 = [&](const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n) {
        enhancedParallelBlock3(A, B, C, n, blockSize);
        };

    measureExecutionTime(sequentialBlockLambda2, A, B, C, n, "III.3 Sequential Block");
    fill(C.begin(), C.end(), vector<int>(n, 0));  // Reset de la matriz C

    measureExecutionTime(enhancedParallelBlockLambda2, A, B, C, n, "III.5 Enhanced Parallel Block");

    return 0;
}

