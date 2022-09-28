#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


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

void one_hot_labels(const unsigned char *A, size_t m,
                    size_t num_classes, float *B) {
    for (int i = 0; i < m; i++) {
        B[i * num_classes + int(A[i])] = 1;
    }
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

void mat_scalar_mul(const float *A, const float s, size_t m, size_t n, float *C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = A[i * n + j] * s;
        }
    }
}

void mat_scalar_div(const float *A, const float s, size_t m, size_t n, float *C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = A[i * n + j] / s;
        }
    }
}

void mat_print(const int *A, size_t m, size_t n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << A[i * n + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void mat_print(const float *A, size_t m, size_t n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << A[i * n + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void mat_print(const unsigned char *A, size_t m, size_t n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << int(A[i * n + j]) << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch) {
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for (int i = 0; i < m; i += batch) {
        if (i + batch > m) {
            break;
        }
        const float *batch_X = X + i * n;
        const unsigned char *batch_y = y + i;

        int num_examples = batch;
        int num_classes = k;

        float *x_mul_theta = new float[num_examples * num_classes]{0};
        matmul(batch_X, theta, num_examples, n, num_classes, x_mul_theta);

        float *exp_theta_x = new float[num_examples * num_classes]{0};
        mat_exp(x_mul_theta, num_examples, num_classes, exp_theta_x);


        float *normalize_z = new float[num_examples * num_classes]{0};
        mat_normalize(exp_theta_x, num_examples, num_classes, normalize_z);


        float *labels = new float[num_examples * num_classes]{0};
        one_hot_labels(batch_y, num_examples, num_classes, labels);

        float *trans_x = new float[n * num_examples]{0};
        mat_transpose(batch_X, num_examples, n, trans_x);

        float *grad = new float[n * num_classes]{0};
        float *tmp = new float[num_examples * num_classes]{0};
        mat_sub(normalize_z, labels, num_examples, num_classes, tmp);
        matmul(trans_x, tmp, n, num_examples, num_classes, grad);
        mat_scalar_div(grad, num_examples, n, num_classes, grad);

//        mat_print(grad,n,num_classes);
//        return;

        // so many memory bloat!
        float *lr_grad = new float[n * num_classes]{0};
        mat_scalar_mul(grad, lr, n, num_classes, lr_grad);


        mat_sub(theta, lr_grad, n, num_classes, theta);

        delete[] lr_grad;
        delete[] tmp;
        delete[] grad;
        delete[] trans_x;
        delete[] labels;
        delete[] normalize_z;
        delete[] exp_theta_x;
        delete[] x_mul_theta;

    }

    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m
) {
m.def("softmax_regression_epoch_cpp",
[](
py::array_t<float, py::array::c_style> X,
        py::array_t<unsigned char, py::array::c_style>
y,
py::array_t<float, py::array::c_style> theta,
float lr,
int batch
) {
softmax_regression_epoch_cpp(
static_cast
<const float *>(X
.

request()

.ptr),
static_cast
<const unsigned char *>(y
.

request()

.ptr),
static_cast
<float *>(theta
.

request()

.ptr),
X.

request()

.shape[0],
X.

request()

.shape[1],
theta.

request()

.shape[1],
lr,
batch
);
},
py::arg("X"), py::arg("y"), py::arg("theta"),
py::arg("lr"), py::arg("batch"));
}
