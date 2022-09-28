//
// Created by huaifeng on 2022-09-27.
//
#include<iostream>
#include <math.h>

using namespace std;

void matmul(const float *A, const float *B,
            size_t m, size_t n, size_t p,
            float *C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i * p + j] = 0;
            for (int k = 0; k < n; k++) {
                float a_ik = A[i * n + k];
                float b_kj = B[k * p + j];
                C[i * p + j] += a_ik * b_kj;
            }
        }

    }
}

void mat_exp(const float *A, size_t m, size_t n, float *B) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            B[i * n + j] = exp(A[i * n + j]);
        }
    }
}

/*
 * a=np.array([
    [1,2,3],
    [2,2,2],
])
b=np.array([
    [1,1],
    [2,2],
    [3,3]
])
 * */
void mat_print(const float *A, size_t m, size_t n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cout << A[i * n + j] << ", ";
        }
        cout << endl;
    }
    cout << endl;
}

void mat_print(const int *A, size_t m, size_t n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cout << A[i * n + j] << ", ";
        }
        cout << endl;
    }
    cout << endl;
}

void mat_normalize(const float *A, size_t m, size_t n, float *B) {
    for (int i = 0; i < m; i++) {
        float sum = 0;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j];
        }
        for (int j = 0; j < n; j++) {
            B[i * n + j] = A[i * n + j] / sum;
        }
    }
}

void one_hot_labels(const unsigned char *A, size_t m, size_t num_classes, float *B) {
    for (int i = 0; i < m; i++) {
        B[i * num_classes + A[i]] = 1;
    }
}


void mat_transpose(const float *A, size_t m, size_t n, float *B) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            B[j * m + i] = A[i * n + j];
        }
    }
}

void mat_sub(const float *A, const float *B, size_t m, size_t n, float *C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = A[i * n + j] - B[i * n + j];
        }
    }
}

int main() {
//    int len = 6;
//    float *A = new float[len]{1, 2, 3, 2, 2, 2};
//    float B[6] = {1, 1, 2, 2, 3, 3,};
//    float C[4];
//    int m = 2;
//    int n = 3;
//    int p = 2;
//    mat_print(A, 2, 3);
//    mat_print(B, 2, 3);
//    mat_sub(A, B, 2, 3, A);
//    mat_print(A, 2, 3);

//    matmul(A, B, m, n, p, C);
//
//    mat_print(C, 2, 2);
//
//    float E[4] = {1, 1, 2, 2};
//    float F[4];
//
//    mat_exp(E, 2, 2, F);
//    mat_print(F, 2, 2);
//
//    float G[6];
//    mat_normalize(A, m, n, G);
//    mat_print(G, m, n);
//
    unsigned char H[3] = {0, 2, 3,};
    int num_classes = 4;
    float I[12] = {0};
    one_hot_labels(H, 3, num_classes, I);
    mat_print(I, 3, 4);
//
//    float J[6];
//    mat_transpose(A, m, n, J);
//
//    mat_print(A, m, n);
//    mat_print(J, n, m);

//    delete[] A;

}