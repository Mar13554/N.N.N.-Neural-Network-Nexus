import ctypes, os, math
lib_path = os.path.abspath("libNNN_math.so")
lib = ctypes.CDLL(lib_path)

# Information-1
lib.get_arr.argtype = ctypes.c_void_p; lib.get_arr.restype = ctypes.POINTER(ctypes.c_float)
lib.get_rows.argtype = ctypes.c_void_p; lib.get_rows.restype = ctypes.c_int
lib.get_columns.argtype = ctypes.c_void_p; lib.get_columns.restype = ctypes.c_int

# Generate
lib.generate_zero_matrix.argtypes = (ctypes.c_int, ctypes.c_int); lib.generate_zero_matrix.restype = ctypes.c_void_p
lib.generate_one_matrix.argtypes = (ctypes.c_int, ctypes.c_int); lib.generate_one_matrix.restype = ctypes.c_void_p
lib.linspace.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_int); lib.linspace.restype = ctypes.c_void_p
lib.generate_matrix.argtypes = (ctypes.c_int, ctypes.c_int); lib.generate_matrix.restype = ctypes.c_void_p

# Functions
FUNC_PTR = ctypes.c_void_p
P_exp = FUNC_PTR.in_dll(lib, "P_exp")
P_NoneP = FUNC_PTR.in_dll(lib, "P_NoneP")
P_Relu = FUNC_PTR.in_dll(lib, "P_Relu")
P_ReluP = FUNC_PTR.in_dll(lib, "P_ReluP")
P_sign = FUNC_PTR.in_dll(lib, "P_sign")
P_log = FUNC_PTR.in_dll(lib, "P_log")
P_add = FUNC_PTR.in_dll(lib, "P_add")
P_mul = FUNC_PTR.in_dll(lib, "P_mul")
P_pow = FUNC_PTR.in_dll(lib, "P_pow")
P_LRelu = FUNC_PTR.in_dll(lib, "P_LRelu")
P_LReluP = FUNC_PTR.in_dll(lib, "P_LReluP")
P_Clip = FUNC_PTR.in_dll(lib, "P_Clip")

# Operations
    # py
lib.create_from.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int); lib.create_from.restype = ctypes.c_void_p
    # Self
lib.num_elements.argtype = ctypes.c_void_p; lib.num_elements.restype = ctypes.c_int
lib.sum_matrix.argtype = ctypes.c_void_p; lib.sum_matrix.restype = ctypes.c_float
lib.sum_matrix_a.argtypes = (ctypes.c_void_p, ctypes.c_int); lib.sum_matrix_a.restype = ctypes.c_void_p
lib.mean_matrix_a.argtypes = (ctypes.c_void_p, ctypes.c_int); lib.mean_matrix_a.restype = ctypes.c_void_p
lib.variance_matrix_a.argtypes = (ctypes.c_void_p, ctypes.c_int); lib.variance_matrix_a.restype = ctypes.c_void_p
    # Custom
lib.operation_0.argtypes = (ctypes.c_void_p, FUNC_PTR); lib.operation_0.restype = ctypes.c_void_p
lib.operation_1.argtypes = (ctypes.c_void_p, FUNC_PTR, ctypes.c_float); lib.operation_1.restype = ctypes.c_void_p
lib.operation_2.argtypes = (ctypes.c_void_p, FUNC_PTR, ctypes.c_float, ctypes.c_float); lib.operation_2.restype = ctypes.c_void_p
    # Fundamental
lib.transpose.argtype = ctypes.c_void_p; lib.transpose.restype = ctypes.c_void_p
        # Element-Wise
lib.matrix_addition.argtypes = (ctypes.c_void_p, ctypes.c_void_p); lib.matrix_addition.restype = ctypes.c_void_p
lib.matrix_mul.argtypes = (ctypes.c_void_p, ctypes.c_void_p); lib.matrix_mul.restype = ctypes.c_void_p
lib.matrix_div.argtypes = (ctypes.c_void_p, ctypes.c_void_p); lib.matrix_div.restype = ctypes.c_void_p
        # Functions
lib.dot_product.argtypes = (ctypes.c_void_p, ctypes.c_void_p); lib.dot_product.restype = ctypes.c_void_p
lib.mat_mul.argtypes = (ctypes.c_void_p, ctypes.c_void_p); lib.mat_mul.restype = ctypes.c_void_p
lib.concatenate.argtypes = (ctypes.c_void_p, ctypes.c_void_p); lib.concatenate.restype = ctypes.c_void_p
lib.stack.argtypes = (ctypes.c_void_p, ctypes.c_void_p); lib.stack.restype = ctypes.c_void_p
lib.linalg_norm.argtype = ctypes.c_void_p; lib.linalg_norm.restype = ctypes.c_float
lib.Normalize_minmax.argtype = ctypes.c_void_p; lib.Normalize_minmax.restype = ctypes.c_void_p
lib.Normalize_zscore.argtype = ctypes.c_void_p; lib.Normalize_zscore.restype = ctypes.c_void_p
lib.Normalize.argtypes = (ctypes.c_void_p, ctypes.c_void_p); lib.Normalize.restype = ctypes.c_void_p
lib.Unnormalize.argtypes = (ctypes.c_void_p, ctypes.c_void_p); lib.Unnormalize.restype = ctypes.c_void_p
lib.any.argtypes = (ctypes.c_void_p, ctypes.c_float); lib.any.restype = ctypes.c_bool

# C free
lib.clear.argtype = ctypes.c_void_p; lib.clear.restype = None
lib.clear_NV.argtype = ctypes.c_void_p; lib.clear_NV.restype = None

# Class Wrapper
class Matrix:
    def __init__(self, obj, size: tuple[int, int], set=False):
        # Size -> (rows, columns)
        self.size = size; self.elements = size[0] * size[1]
        self.ndim = 1 if (size[0] == 1 or size[1] == 1) else 2
        if not set:
            if obj not in [0, 1, 2]: raise Exception("Invalid type")
            match obj:
                case 0: self.val = lib.generate_zero_matrix(size[0], size[1])
                case 1: self.val = lib.generate_one_matrix(size[0], size[1])
                case 2: self.val = lib.generate_matrix(size[0], size[1])
        else: self.val = obj
    def __add__(self, other):
        if isinstance(other, Matrix): new_val = lib.matrix_addition(self.val, other.val)
        else: new_val = lib.operation_1(self.val, P_add, other)
        return Matrix(new_val, self.size, set=True)
    def __mul__(self, other):
        if isinstance(other, Matrix):
            raise Exception("Warning! Use the function 'elementwise_mul' in libN or the method 'matmul' instead")
        else: new_val = lib.operation_1(self.val, P_mul, other)
        return Matrix(new_val, self.size, set=True)
    def __matmul__(self, other):
        if isinstance(other, Matrix):
            new_val = lib.mat_mul(self.val, other.val)
        else:
            print(f"Warning! __matmul__ received type {type(other)} instead of Matrix.")
            return None
        return Matrix(new_val, (self.size[0], other.size[1]), set=True)
    def __sub__(self, other):
        if isinstance(other, Matrix):
            new_val = lib.matrix_addition(self.val, lib.operation_1(other.val, P_mul, -1))
        else: new_val = lib.operation_1(self.val, P_add, -other)
        return Matrix(new_val, self.size, set=True)
    def __truediv__(self, other):
        if isinstance(other, Matrix):
            print("Warning! Use 'elementwise_div' in the class below instead")
            return None
        else: new_val = lib.operation_1(self.val, P_mul, 1/other)
        return Matrix(new_val, self.size, set=True)
    def __pow__(self, other):
        if isinstance(other, Matrix):
            print("__pow__ method does not exist for matrices")
            return None
        else: new_val = lib.operation_1(self.val, P_pow, other)
        return Matrix(new_val, self.size, set=True)
    def __del__(self): lib.clear(self.val)
    def len(self): return lib.num_elements(self.val)
    def confirm_size(self, val): return lib.get_rows(val), lib.get_columns(val)
    def sum(self, axis=None):
        if axis is None: return lib.sum_matrix(self.val)
        try: float(axis)
        except Exception: raise Exception("Invalid axis")
        if axis in [0.0, 1.0]:
            new_val = lib.sum_matrix_a(self.val, axis)
            R, C = self.confirm_size(new_val)
            return Matrix(new_val, (R, C), set=True)
        else: raise Exception("Invalid axis")
    def mean(self, axis=None):
        if axis is None: return lib.sum_matrix(self.val)/lib.num_elements(self.val)
        try: float(axis)
        except Exception: raise Exception("Invalid axis")
        if axis in [0.0, 1.0]:
            new_val = lib.mean_matrix_a(self.val, axis)
            R, C = self.confirm_size(new_val)
            return Matrix(new_val, (R, C), set=True)
        else: raise Exception("Invalid axis")
    def var(self, axis=None):
        if axis is None: return None #WIP
        try: float(axis)
        except Exception: raise Exception("Invalid axis")
        if axis in [0.0, 1.0]:
            new_val = lib.variance_matrix_a(self.val, axis)
            R, C = self.confirm_size(new_val)
            return Matrix(new_val, (R, C), set=True)
        else: raise Exception("Invalid axis")
    def T(self):
        return Matrix(lib.transpose(self.val), (self.size[1], self.size[0]), set=True)
    def L2(self):
        return lib.linalg_norm(self.val)
    def format(self):
        Array = lib.get_arr(self.val)
        mat1 = [[Array[row * self.size[1] + column] for column in range(self.size[1])] for row in range(self.size[0])]
        return mat1

class Norm_Vals:
    def __init__(self, NV_ptr: ctypes.c_void_p):
        self.val = NV_ptr
    def __del__(self):
        lib.clear_NV(self.val)

# Function wrapper
    # 1-D Matrices
def concatenate(mat1: Matrix, mat2: Matrix):
    new_val = lib.concatenate(mat1.val, mat2.val)
    return Matrix(new_val, (1, (mat1.size[0]*mat1.size[1])+(mat2.size[0]*mat2.size[1])), set=True)
def concatenate_all(mats: list[Matrix]):
    main_mat = mats[0]
    for mat in mats[1:]: main_mat = concatenate(main_mat, mat)
    return main_mat

def stack(mat1: Matrix, mat2: Matrix):
    new_val = lib.stack(mat1.val, mat2.val)
    return Matrix(new_val, (mat1.size[0]+mat2.size[0], mat1.size[1]), set=True)
def stack_all(mats: list[Matrix]):
    main_mat = mats[0]
    for mat in mats[1:]: main_mat = stack(main_mat, mat)
    return main_mat

def to_C(list1: list):
    if isinstance(list1[0], list):
        clean_list = [float(item) for row in list1 for item in row]
        R = len(list1); C = len(list1[0])
    else:
        clean_list = [float(i) for i in list1]
        R = 1; C = len(list1)
    if R*C > len(clean_list): clean_list += [0.0 for i in range(R*C - len(clean_list))]
    FloatArray = ctypes.c_float * len(clean_list)
    c_array = FloatArray(*clean_list)
    val = lib.create_from(c_array, R, C)
    return Matrix(val, (R, C), set=True)

def linspace(a: float, b: float, amount: int):
    new_val = lib.linspace(a, b, amount)
    return Matrix(new_val, (1, amount), set=True)

def exp(mat1):
    if isinstance(mat1, Matrix): return Matrix(lib.operation_0(mat1.val, P_exp), mat1.size, set=True)
    else: return math.exp(mat1)
def log(mat1):
    if isinstance(mat1, Matrix): return Matrix(lib.operation_0(mat1.val, P_log), mat1.size, set=True)
    else: return None
def sign(mat1):
    if isinstance(mat1, Matrix): return Matrix(lib.operation_0(mat1.val, P_sign), mat1.size, set=True)
    else: return None

def elementwise_mul(mat1: Matrix, mat2: Matrix):
    if mat1.size != mat2.size:
        print(f"Matrices do not have the compatible size {mat1.size} != {mat2.size}")
        return None
    new_val = lib.matrix_mul(mat1.val, mat2.val)
    return Matrix(new_val, (mat1.size[0], mat2.size[1]), set=True)

def elementwise_div(mat1: Matrix, mat2: Matrix):
    if mat1.size != mat2.size:
        print(f"Matrices do not have the compatible size {mat1.size} != {mat2.size}")
        return None
    new_val = lib.matrix_div(mat1.val, mat2.val)
    return Matrix(new_val, (mat1.size[0], mat2.size[1]), set=True)

def dot(mat1: Matrix, mat2: Matrix):
    if mat1.size[1] != mat2.size[0]:
        print(f"Matrices do not have the compatible size {mat1.size[1]} != {mat2.size[0]}")
        return None
    new_val = lib.dot_product(mat1.val, mat2.val)
    return Matrix(new_val, (mat1.size[0], mat2.size[1]), set=True)

def clip(mat1: Matrix, a, b):
    new_val = lib.operation_2(mat1.val, P_Clip, a, b)
    return Matrix(new_val, mat1.size, set=True)

def normalize_minmax(mat1: Matrix):
    NV_val = lib.Normalize_minmax(mat1.val)
    return Norm_Vals(NV_val)
def normalize_zscore(mat1: Matrix):
    NV_val = lib.Normalize_zscore(mat1.val)
    return Norm_Vals(NV_val)

def normalize(mat1: Matrix, NV: Norm_Vals):
    mat2_val = lib.Normalize(mat1.val, NV.val)
    return Matrix(mat2_val, mat1.size, set=True)
def unnormalize(mat1: Matrix, NV: Norm_Vals):
    mat2_val = lib.Unnormalize(mat1.val, NV.val)
    return Matrix(mat2_val, mat1.size, set=True)

def any(mat1: Matrix, num: float):
    return lib.any(mat1.val, num)

class class_libN:
    def __init__(self):
        self.concatenate = concatenate; self.concatenate_all = concatenate_all
        self.stack = stack; self.stack_all = stack_all
        self.to_C = to_C; self.linspace = linspace
        self.exp = exp; self.log = log; self.sign = sign
        self.elementwise_mul = elementwise_mul; self.elementwise_div = elementwise_div
        self.dot = dot
        self.min_max = normalize_minmax; self.zscore = normalize_zscore
        self.normalize = normalize; self.unnormalize = unnormalize
        self.clip = clip; self.any = any
    def help(self):
        print()
libN = class_libN()

# Activation Functions
def No_Activation(mat1: Matrix): return mat1
def No_Activation_prime(mat1: Matrix):
    new_val = lib.operation_0(mat1.val, P_NoneP)
    return Matrix(new_val, mat1.size, set=True)
#[lambda x: 1/(np.exp(-np.clip(x, -700, 700))+1), lambda x: x*(1-x)]
def Sigmoid(mat1: Matrix):
    new_mat = exp(clip(mat1, -700, 700) + 1) ** -1
    return new_mat
def Sigmoid_prime(mat1: Matrix):
    new_val = elementwise_mul(mat1, (mat1*-1 + 1))
    return Matrix(new_val, mat1.size, set=True)
def Relu(mat1: Matrix):
    new_val = lib.operation_0(mat1.val, P_Relu)
    return Matrix(new_val, mat1.size, set=True)
def Relu_prime(mat1: Matrix):
    new_val = lib.operation_0(mat1.val, P_ReluP)
    return Matrix(new_val, mat1.size, set=True)
coefficient = 0.2
def LeakyRelu(mat1: Matrix):
    new_val = lib.operation_1(mat1.val, P_LRelu, coefficient)
    return Matrix(new_val, mat1.size, set=True)
def LeakyRelu_prime(mat1: Matrix):
    new_val = lib.operation_1(mat1.val, P_LReluP, coefficient)
    return Matrix(new_val, mat1.size, set=True)

if "__main__" == __name__:
    Set_A = [[14.0], [33.0], [8.0], [13.0], [0.0], [45.0], [20.0], [16.0], [27.0], [15.0], [42.0], [39.0], [32.0], [4.0], [47.0], [5.0], [36.0], [28.0], [48.0], [46.0], [30.0], [44.0], [9.0], [1.0], [7.0], [34.0], [35.0], [11.0], [26.0], [18.0], [23.0], [10.0], [6.0], [19.0], [17.0], [25.0], [40.0], [2.0], [43.0], [38.0], [41.0], [49.0], [50.0], [22.0], [21.0], [24.0]]
    A = libN.stack_all([libN.to_C(i) for i in Set_A])
    B = libN.to_C([-0.6821833848953247])
    C = libN.dot(A, B)
    print(C.format())
    print(C.size)
    D = libN.to_C([1])
    E = C - D
    print(E.format())