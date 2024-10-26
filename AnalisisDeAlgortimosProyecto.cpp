#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <functional>
#include <fstream>


using namespace std;
using namespace chrono;

// Función para llenar la matriz con números aleatorios de 6 dígitos
void fillMatrix(vector<vector<int>>& matrix, int n) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(100000, 999999);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            matrix[i][j] = dis(gen);
}

void saveMatrixToFile(const vector<vector<int>>& matrix, const string& filename) {
    ofstream file(filename);
    if (file.is_open()) {
        int n = matrix.size();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                file << matrix[i][j] << " ";
            }
            file << endl;  // Nueva línea después de cada fila
        }
        file.close();
        cout << "Matriz guardada en " << filename << endl;
    }
    else {
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

// Algoritmo 1: Strassen-Winograd (Versión simplificada)
void strassenWinograd(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n) {
    if (n <= 2) {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < n; ++k)
                    C[i][j] += A[i][k] * B[k][j];
        return;
    }
    // Implementación más avanzada puede requerir recursión y dividir matrices
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

// Función para medir el tiempo de ejecución
void measureExecutionTime(void (*algorithm)(const vector<vector<int>>&, const vector<vector<int>>&, vector<vector<int>>&, int),
    const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n, const string& name) {
    auto start = high_resolution_clock::now();
    algorithm(A, B, C, n);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start);
    cout << "Tiempo de ejecución (" << name << "): " << duration.count() << " ns" << endl;
}

// Sobrecarga de measureExecutionTime que acepta un argumento extra
void measureExecutionTime(const function<void(const vector<vector<int>>&, const vector<vector<int>>&, vector<vector<int>>&, int)>& algorithm,
    const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int n, const string& name) {
    auto start = high_resolution_clock::now();
    algorithm(A, B, C, n);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start);
    cout << "Tiempo de ejecución (" << name << "): " << duration.count() << " ns" << endl;
}


// Función principal
int main() {
    int n = 2;  // Tamaño de la matriz
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

    if (!loadMatrixFromFile(A, "matriz_A2.txt") || !loadMatrixFromFile(B, "matriz_B2.txt")) {
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

    return 0;
}

