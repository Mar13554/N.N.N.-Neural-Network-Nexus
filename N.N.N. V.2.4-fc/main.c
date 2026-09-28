#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <float.h>
#include <string.h>
#include <stdbool.h>

typedef struct{
    int rows;
    int columns;
    float* arr;
} Matrix;

//  Information-1
float* get_arr(Matrix* mat1){return mat1->arr;}
int get_rows(Matrix* mat1){return mat1->rows;}
int get_columns(Matrix* mat1){return mat1->columns;}

typedef struct{
    float* shift;
    float* rescale;
} Norm_Vals;

//  Generate
Matrix* generate_zero_matrix(int rows, int columns){
    if (rows <= 0 || columns <= 0) return NULL;
    Matrix* mat1 = (Matrix*)malloc(sizeof(Matrix));
    if (mat1 == NULL){return NULL;}
    float* arr = (float*)calloc(rows * columns, sizeof(float));
    if (arr == NULL) {free(mat1); return NULL;}
    mat1->rows = rows; mat1->columns = columns; mat1->arr = arr;
    return mat1;
}
Matrix* generate_one_matrix(int rows, int columns){
    if (rows <= 0 || columns <= 0) return NULL;
    // Matrix
    Matrix* mat1 = (Matrix*)malloc(sizeof(Matrix));
    if (mat1 == NULL){return NULL;}
    float* arr = (float*)malloc(rows * columns * sizeof(float));
    if (arr == NULL){free(mat1); return NULL;}
    for (int i = 0; i < rows*columns; i++){arr[i] = 1;}
    mat1->rows = rows; mat1->columns = columns; mat1->arr = arr;
    return mat1;
}
Matrix* linspace(float a, float b, int amount){
    if (!(a < b) || (amount < 1)){return NULL;}
    Matrix* mat1 = (Matrix*)malloc(sizeof(Matrix));
    if (mat1 == NULL){return NULL;}
    float* arr = (float*)malloc(amount * sizeof(float));
    if (arr == NULL){free(mat1); return NULL;}
    float val = a; float l = (b-a)/(amount-1);
    for (int i = 0; i < amount; i++){arr[i] = val; val += l;}
    mat1->rows = 1; mat1->columns = amount; mat1->arr = arr;
    return mat1;
}
Matrix* generate_matrix(int rows, int columns){
    if (rows <= 0 || columns <= 0) return NULL;
    //Seed
    static int seeded = 0;
    if (!seeded) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        srand(ts.tv_nsec);
        seeded = 1;
    }
    // Matrix
    Matrix* mat1 = (Matrix*)malloc(sizeof(Matrix));
    if (mat1 == NULL){return NULL;}
    float* arr = (float*)malloc(rows * columns * sizeof(float));
    if (arr == NULL){free(mat1); return NULL;}
    for (int i = 0; i < rows*columns; i++){
            arr[i] = ((float)rand() / RAND_MAX)*2 - 1;
    }
    mat1->rows = rows; mat1->columns = columns; mat1->arr = arr;
    return mat1;
}
// Functions
    // Define
float D_exp(float x){return exp(x);}
float D_NoneP(float x){return 1;}
float D_Relu(float x){if (x >= 0){return x;} else {return 0;} }
float D_ReluP(float x){if (x >= 0){return 1;} else {return 0;} }
float D_sign(float x){if (x>0){return 1;} else if (x==0){return 0;} else {return -1;} }
float D_log(float x){return logf(x);}
float D_add(float x, float a){return x+a;}
float D_mul(float x, float a){return x*a;}
float D_pow(float x, float a){return powf(x, a);}
float D_LRelu(float x, float a){if (x >= 0){return x;} else {return x*a;} }
float D_LReluP(float x, float a){if (x >= 0){return 1;} else {return a;} }
float D_Clip(float x, float a1, float a2){if (x > a2){return a2;} else if (x < a1){return a1;} else {return x;}}
    // Pointers
float (*P_exp)(float) = D_exp;
float (*P_NoneP)(float) = D_NoneP;
float (*P_Relu)(float) = D_Relu;
float (*P_ReluP)(float) = D_ReluP;
float (*P_sign)(float) = D_sign;
float (*P_log)(float) = D_log;
float (*P_add)(float, float) = D_add;
float (*P_mul)(float, float) = D_mul;
float (*P_pow)(float, float) = D_pow;
float (*P_LRelu)(float, float) = D_LRelu;
float (*P_LReluP)(float, float) = D_LReluP;
float (*P_Clip)(float, float, float) = D_Clip;

// Operations
    // From py
Matrix* create_from(float* arr1, int R, int C){
    Matrix* mat1 = (Matrix*)malloc(sizeof(Matrix));
    if (mat1 == NULL){return NULL;}
    float* arr2 = (float*)malloc(R*C*sizeof(float));
    if (arr2 == NULL){free(mat1); return NULL;}
    memcpy(arr2, arr1, R*C*sizeof(float));
    mat1->rows = R; mat1->columns = C; mat1->arr = arr2;
    return mat1;
}
    // Self
int num_elements(Matrix* mat1){return (mat1->rows)*(mat1->columns);}
float sum_matrix(Matrix* mat1){
    if (mat1 == NULL){return 0;}
    float sum = 0;
    for (int i = 0; i < (mat1->rows)*(mat1->columns); i++){sum += (mat1->arr)[i];}
    return sum;
}
Matrix* sum_matrix_a(Matrix* mat1, int axis){
    Matrix* mat2 = (Matrix*)malloc(sizeof(Matrix));
    if (mat2 == NULL){return NULL;}
    int R = mat1->rows; int C = mat1->columns;
    if (axis == 0 && R > 0){
        float* arr = (float*)calloc(C, sizeof(float));
        if (arr == NULL) { free(mat2); return NULL; }
        for (int i = 0; i < C; i++){
            for (int j = 0; j < R; j++){
                arr[i] += (mat1->arr)[j*C + i];
            }
        }
        mat2->rows = 1; mat2->columns = C; mat2->arr = arr;
        return mat2;
    }
    else if (axis == 1 && C > 0){
        float* arr = (float*)calloc(R, sizeof(float));
        if (arr == NULL) { free(mat2); return NULL; }
        for (int i = 0; i < R; i++){
            for (int j = 0; j < C; j++){
                arr[i] += (mat1->arr)[i*C + j];
            }
        }
        mat2->rows = R; mat2->columns = 1; mat2->arr = arr;
        return mat2;
    }
    else {free(mat2); return NULL;}
}
Matrix* mean_matrix_a(Matrix* mat1, int axis){
    Matrix* mat2 = (Matrix*)malloc(sizeof(Matrix));
    if (mat2 == NULL) { return NULL;}
    int R = mat1->rows; int C = mat1->columns;
    if (axis == 0 && R > 0){
        float* arr = (float*)calloc(C, sizeof(float));
        if (arr == NULL) { free(mat2); return NULL;}
        for (int i = 0; i < C; i++){
            for (int j = 0; j < R; j++){
                arr[i] += (mat1->arr)[j * C + i];
            }
            arr[i] /= R;
        }
        mat2->rows = 1; mat2->columns = C; mat2->arr = arr;
        return mat2;
    }
    else if (axis == 1 && C > 0){
        float* arr = (float*)calloc(R, sizeof(float));
        if (arr == NULL) { free(mat2); return NULL;}
        for (int i = 0; i < R; i++){
            for (int j = 0; j < C; j++){
                arr[i] += (mat1->arr)[i * C + j];
            }
            arr[i] /= C;
        }
        mat2->rows = R; mat2->columns = 1; mat2->arr = arr;
        return mat2;
    }
    else {free(mat2); return NULL;}
}
Matrix* variance_matrix_a(Matrix* mat1, int axis) {
    Matrix* mean_mat = mean_matrix_a(mat1, axis);
    if (mean_mat == NULL) return NULL;
    Matrix* var_mat = (Matrix*)malloc(sizeof(Matrix));
    if (var_mat == NULL) { free(mean_mat->arr); free(mean_mat); return NULL; }
    int R = mat1->rows; int C = mat1->columns;
    if (axis == 0 && R > 0){
        float* arr = (float*)calloc(C, sizeof(float));
        if (arr == NULL) { free(var_mat); free(mean_mat->arr); free(mean_mat); return NULL; }
        for (int i = 0; i < C; i++){
            float mu = mean_mat->arr[i];
            for (int j = 0; j < R; j++){
                float diff = (mat1->arr)[j * C + i] - mu;
                arr[i] += diff * diff;
            }
            arr[i] /= R;
        }
        var_mat->rows = 1; var_mat->columns = C; var_mat->arr = arr;
    }
    else if (axis == 1 && C > 0) {
        float* arr = (float*)calloc(R, sizeof(float));
        if (arr == NULL) { free(var_mat); free(mean_mat->arr); free(mean_mat); return NULL; }
        for (int i = 0; i < R; i++){
            float mu = mean_mat->arr[i];
            for (int j = 0; j < C; j++){
                float diff = (mat1->arr)[i * C + j] - mu;
                arr[i] += diff * diff;
            }
            arr[i] /= C;
        }
        var_mat->rows = R; var_mat->columns = 1; var_mat->arr = arr;
    }
    else { free(var_mat); free(mean_mat->arr); free(mean_mat); return NULL; }
    free(mean_mat->arr); free(mean_mat);
    return var_mat;
}
    // Operation
Matrix* operation_0(Matrix* mat1, float (*func)(float)){
    Matrix* mat2 = (Matrix*)malloc(sizeof(Matrix));
    if (mat2 == NULL){return NULL;}
    int R = mat1->rows; int C = mat1->columns;
    float* arr = (float*)malloc(R*C*sizeof(float));
    if (arr == NULL){free(mat2); return NULL;}
    for (int i = 0; i < R*C; i++){arr[i] = func((mat1->arr)[i]);}
    mat2->rows = mat1->rows; mat2->columns = mat1->columns; mat2->arr = arr;
    return mat2;
}
Matrix* operation_1(Matrix* mat1, float (*func)(float, float), float x1){
    Matrix* mat2 = (Matrix*)malloc(sizeof(Matrix));
    if (mat2 == NULL){return NULL;}
    int R = mat1->rows; int C = mat1->columns;
    float* arr = (float*)malloc(R*C*sizeof(float));
    if (arr == NULL){free(mat2); return NULL;}
    for (int i = 0; i < R*C; i++){arr[i] = func((mat1->arr)[i], x1);}
    mat2->rows = mat1->rows; mat2->columns = mat1->columns; mat2->arr = arr;
    return mat2;
}
Matrix* operation_2(Matrix* mat1, float (*func)(float, float, float), float x1, float x2){
    Matrix* mat2 = (Matrix*)malloc(sizeof(Matrix));
    if (mat2 == NULL){return NULL;}
    int R = mat1->rows; int C = mat1->columns;
    float* arr = (float*)malloc(R*C*sizeof(float));
    if (arr == NULL){free(mat2); return NULL;}
    for (int i = 0; i < R*C; i++){arr[i] = func((mat1->arr)[i], x1, x2);}
    mat2->rows = mat1->rows; mat2->columns = mat1->columns; mat2->arr = arr;
    return mat2;
}
// Fundamental
Matrix* transpose(Matrix* mat1){
    Matrix* mat2 = (Matrix*)malloc(sizeof(Matrix));
    if (mat2 == NULL){return NULL;}
    int R = mat1->rows; int C = mat1->columns;
    float* arr = (float*)malloc(R * C * sizeof(float));
    if (arr == NULL){free(mat2); return NULL;}
    for (int j = 0; j < C; j++){
        for (int i = 0; i < R; i++){
            arr[j*R + i] = (mat1->arr)[i*C + j];
        }
    }
    mat2->rows = C; mat2->columns = R; mat2->arr = arr;
    return mat2;
}

Matrix* matrix_addition(Matrix* mat1, Matrix* mat2) {
    int R1 = mat1->rows; int C1 = mat1->columns;
    int R2 = mat2->rows; int C2 = mat2->columns;

    // Output Dimensions & Compatibility
    int outR, outC;

    // (a, b) + (a, b); (a, b) + (a, 1); (a, b) + (1, b)
    if (R1 == R2 && C1 == C2){outR = R1; outC = C1;}
    else if (R1 == R2 && (C1 == 1 || C2 == 1)){outR = R1; outC = (C1 > C2) ? C1 : C2;}
    else if (C1 == C2 && (R1 == 1 || R2 == 1)){outR = (R1 > R2) ? R1 : R2; outC = C1;}
    else {return NULL;}

    // Memory
    Matrix* mat3 = (Matrix*)malloc(sizeof(Matrix));
    if (mat3 == NULL){return NULL;};
    float* arr = (float*)malloc(outR * outC * sizeof(float));
    if (arr == NULL) {free(mat3); return NULL; }

    // Addition
    for (int i = 0; i < outR; i++) {
        for (int j = 0; j < outC; j++) {
            int idx1 = (R1 == 1 ? 0 : i) * C1 + (C1 == 1 ? 0 : j);
            int idx2 = (R2 == 1 ? 0 : i) * C2 + (C2 == 1 ? 0 : j);
            arr[i * outC + j] = mat1->arr[idx1] + mat2->arr[idx2];
        }
    }

    mat3->rows = outR; mat3->columns = outC; mat3->arr = arr;
    return mat3;
}

// Element-wise mul and div
Matrix* matrix_mul(Matrix* mat1, Matrix* mat2){
    int R = mat1->rows; int C = mat1->columns;
    if (!(R == mat2->rows && C == mat2->columns)){return NULL;}
    // Allocate
    Matrix* mat3 = (Matrix*)malloc(sizeof(Matrix));
    if (mat3 == NULL){return NULL;}
    float* arr = (float*)malloc(R * C * sizeof(float));
    if (arr == NULL){free(mat3); return NULL;}
    // Multiplication
    for (int i = 0; i < R*C; i++){
        arr[i] = (mat1->arr)[i] * (mat2->arr)[i];
    }
    // Assign
    mat3->rows = R; mat3->columns = C; mat3->arr = arr;
    return mat3;
}

Matrix* matrix_div(Matrix* mat1, Matrix* mat2){
    int R = mat1->rows; int C = mat1->columns;
    if (!(R == mat2->rows && C == mat2->columns)){return NULL;}
    // Allocate
    Matrix* mat3 = (Matrix*)malloc(sizeof(Matrix));
    if (mat3 == NULL){return NULL;}
    float* arr = (float*)malloc(R * C * sizeof(float));
    if (arr == NULL){free(mat3); return NULL;}
    // Division
    for (int i = 0; i < R*C; i++){
        if((mat2->arr)[i] == 0){arr[i] = 0;}
        else{arr[i] = (mat1->arr)[i] / (mat2->arr)[i];}
    }
    // Assign
    mat3->rows = R; mat3->columns = C; mat3->arr = arr;
    return mat3;
}

Matrix* dot_product(Matrix* mat1, Matrix* mat2){
    if (mat1->columns != mat2->rows){return NULL;}
    int iterations = mat1->columns;
    int R = mat1->rows; int C = mat2->columns;
    Matrix* mat3 = (Matrix*)malloc(sizeof(Matrix));
    if (mat3 == NULL){return NULL;}
    float* arr = (float*)malloc(R * C * sizeof(float));
    if (arr == NULL){free(mat3); return NULL;}
    for (int row = 0; row < R; row++){
        for (int column = 0; column < C; column++){
            //Each element in mat3
            float sum = 0;
            for (int i = 0; i < iterations; i++){
                sum += (mat1->arr)[row*(mat1->columns) + i] * (mat2->arr)[i*(mat2->columns) + column];
            }
            arr[row*C + column] = sum;
        }
    }
    mat3->rows = R; mat3->columns = C; mat3->arr = arr;
    return mat3;
}

Matrix* mat_mul(Matrix* mat1, Matrix* mat2) {
    // Safety
    if (mat1 == NULL || mat2 == NULL || mat1->arr == NULL || mat2->arr == NULL) {return NULL;}
    if (mat1->rows <= 0 || mat1->columns <= 0 || mat2->rows <= 0 || mat2->columns <= 0) {return NULL;}
    if (mat1->columns != mat2->rows) {return NULL;}

    int R = mat1->rows; int C = mat2->columns; int K = mat1->columns;

    // Allocation
    size_t total_elements = (size_t)R * (size_t)C;
    if (total_elements > SIZE_MAX / sizeof(float)) {return NULL;}
    Matrix* result = (Matrix*)malloc(sizeof(Matrix));
    if (result == NULL) {return NULL;}

    float* arr = (float*)calloc(total_elements, sizeof(float));
    if (arr == NULL) {
        free(result);
        return NULL;
    }

    // Loop (i-k-j)
    for (int row = 0; row < R; row++) {
        int r_offset = row * K; int res_offset = row * C;
        for (int k = 0; k < K; k++) {
            float val1 = mat1->arr[r_offset + k]; int k_offset = k * C;
            for (int col = 0; col < C; col++) {
                arr[res_offset + col] += val1 * mat2->arr[k_offset + col];
            }
        }
    }

    result->rows = R; result->columns = C; result->arr = arr;
    return result;
}

Matrix* concatenate(Matrix* mat1, Matrix* mat2){
    Matrix* mat3 = (Matrix*)malloc(sizeof(Matrix));
    if (mat3 == NULL){return NULL;}
    int S1 = mat1->rows * mat1->columns;
    int S2 = mat2->rows * mat2->columns;
    float* arr = (float*)malloc((S1 + S2) * sizeof(float));
    if (arr == NULL) {free(mat3); return NULL;}
    memcpy(arr, mat1->arr, S1*sizeof(float));
    memcpy(arr+S1, mat2->arr, S2*sizeof(float));
    mat3->rows = 1; mat3->columns = S1+S2; mat3->arr = arr;
    return mat3;
}

Matrix* stack(Matrix* mat1, Matrix* mat2){
    if (mat1->columns != mat2->columns){return NULL;}
    Matrix* mat3 = concatenate(mat1, mat2);
    if (mat3 == NULL){return NULL;}
    mat3->rows = mat1->rows + mat2->rows; mat3->columns = mat1->columns;
    return mat3;
}

float linalg_norm(Matrix* mat1){
    float squared_sums = 0; int size = mat1->rows * mat1->columns;
    for (int i = 0; i < size; i++){squared_sums += powf((mat1->arr)[i], 2);}
    return sqrt(squared_sums);
}

Norm_Vals* Normalize_minmax(Matrix* mat1){
    int R = mat1->rows; int C = mat1->columns;
    float* shift = (float*)malloc(sizeof(float) * mat1->columns);
    if (shift == NULL){return NULL;}
    float* rescale = (float*)malloc(sizeof(float)* mat1->columns);
    if (rescale == NULL){free(shift); return NULL;}
    Norm_Vals* NV = (Norm_Vals*)malloc(sizeof(Norm_Vals));
    if (NV == NULL){free(shift); free(rescale); return NULL;}

    for (int i = 0; i < C; i++){
        float minimum = FLT_MAX; float maximum = FLT_MIN; float val;
        for (int j = 0; j < R; j++){
            val = mat1->arr[j*C + i];
            if (val < minimum){minimum = val;}
            if (val > maximum){maximum = val;}
        }
        shift[i] = minimum; rescale[i] = maximum - minimum;
    }
    NV->shift = shift; NV->rescale = rescale;
    return NV;
}

Norm_Vals* Normalize_zscore(Matrix* mat1){
    float* shift = (float*)malloc(sizeof(float) * mat1->columns);
    if (shift == NULL){return NULL;}
    float* rescale = (float*)malloc(sizeof(float)* mat1->columns);
    if (rescale == NULL){free(shift); return NULL;}
    Norm_Vals* NV = (Norm_Vals*)malloc(sizeof(Norm_Vals));
    if (NV == NULL){free(shift); free(rescale); return NULL;}

    int C = mat1->columns; int R = mat1->rows;
    for (int i = 0; i < mat1->columns; i++){
        float sum = 0.0f; float sq_sum = 0.0f;
        for (int j = 0; j < R; j++){sum += mat1->arr[j * C + i];}
        float mean = sum / R;

        for (int j = 0; j < R; j++) {
            float diff = mat1->arr[j * C + i] - mean;
            sq_sum += diff * diff;
        }
        float std_dev = sqrtf(sq_sum / R);

        shift[i] = mean; rescale[i] = std_dev;
    }
    NV->shift = shift; NV->rescale = rescale;
    return NV;
}

Matrix* Normalize(Matrix* mat1, Norm_Vals* NV){
    int R = mat1->rows; int C = mat1->columns;
    Matrix* mat2 = (Matrix*)malloc(sizeof(Matrix));
    if (mat2 == NULL){return NULL;}
    float* arr = (float*)malloc(sizeof(float)*R*C);
    if (arr == NULL){free(mat2); return NULL;}

    mat2->arr = arr; mat2->rows = R; mat2->columns = C;
    for (int i = 0; i < C; i++){
        float d = (NV->rescale[i] != 0) ? NV->rescale[i] : 1;
        float func(float x){return (x - NV->shift[i]) / d;}
        for (int j = 0; j < R; j++){
            mat2->arr[j*C + i] = func(mat1->arr[j*C + i]);
        }
    }
    return mat2;
}

Matrix* Unnormalize(Matrix* mat1, Norm_Vals* NV){
    int R = mat1->rows; int C = mat1->columns;
    Matrix* mat2 = (Matrix*)malloc(sizeof(Matrix));
    if (mat2 == NULL){return NULL;}
    float* arr = (float*)malloc(sizeof(float)*R*C);
    if (arr == NULL){free(mat2); return NULL;}

    mat2->arr = arr; mat2->rows = R; mat2->columns = C;
    for (int i = 0; i < C; i++){
        float func(float x){return x * NV->rescale[i] + NV->shift[i];}
        for (int j = 0; j < R; j++){
            mat2->arr[j*C + i] = func(mat1->arr[j*C + i]);
        }
    }
    return mat2;
}

bool any(Matrix* mat1, float num){
    for (int i = 0; i < (mat1->rows * mat1->columns); i++){
            if ((mat1->arr)[i] == num){return true;}
    }
    return false;
}

void clear(Matrix* mat1){
    free(mat1->arr);
    free(mat1);
}

void clear_NV(Norm_Vals* NV){
    free(NV->shift); free(NV->rescale);
    free(NV);
}
