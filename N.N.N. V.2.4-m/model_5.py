# N.N.N. V.2.4-m

import numpy as np, math, random, copy
from numpy.typing import NDArray
global_seed = 314; numpy_rng = np.random.default_rng(seed=global_seed); global_random = random.Random(global_seed)
def set_global_seed(seed_num: int):
    global global_seed, numpy_rng, global_random
    global_seed = seed_num
    numpy_rng = np.random.default_rng(seed=seed_num)
    global_random = random.Random(seed_num)

#Activation Functions
def softmax_1(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(z)/np.sum(np.exp(z), axis=-1, keepdims=True)
def softmax_2(x):
    s = softmax_1(x)
    I = np.eye(s.shape[-1])
    return I * s[..., None, :] - np.einsum('...i,...j->...ij', s, s)
class Activation_Functions_Class:
    def __init__(self):
        #No activation
        self.no_activation = [lambda x: x, lambda x: 1]
        #Sigmoid [0-1]
        self.sigmoid = [lambda x: 1/(np.exp(-np.clip(x, -700, 700))+1), lambda x: x*(1-x)] #x has already been passed through sigmoid
        #Relu
        self.relu = [lambda x: np.where(x >= 0, x, 0), lambda x: np.where(x > 0, 1, 0)]
        #LeakyRelu
        coefficient = 0.2
        self.leaky_relu = [lambda x: np.where(x >= 0, x, x*coefficient), lambda x: np.where(x > 0, 1, coefficient)]
        #Softmax
        self.softmax = [softmax_1, softmax_2]
Activation_Functions = Activation_Functions_Class()

epsilon = 1e-7 # 0.0000001
#Loss Functions
class Loss_Functions_Class:
    def __init__(self):
        #Both x and y are NDArray[np.float64]
        #Format is the same as above, [loss_function, loss_function_derivative]
        self.MSE = [lambda x, y: np.divide(np.sum(np.pow(y-x, 2)), y.size), lambda x, y: np.multiply(np.divide(2, y.size), x-y)]
        self.RMSE = [lambda x, y: np.sqrt(self.MSE[0](x, y)), None]
        self.MAE = [lambda x, y: np.sum(abs(y-x))/y.size, lambda x, y: (-1/y.size)*np.sign(y-x)]
        #Automatically used for the last layer in classification
        self.Binary_Cross_Entropy_Loss = [lambda x, y: -(y[0]*np.log(x[0])+(1-y[0])*np.log(1-x[0])), lambda x, y: (y/np.clip(x, epsilon, 1 - epsilon))-((1-y)/(1-np.clip(x, epsilon, 1 - epsilon)))]
        self.Categorical_Cross_Entropy_Loss = [lambda x, y: np.sum(y*np.log(x))*(-1), lambda x, y: -(y/x)]
Loss_Functions = Loss_Functions_Class()

#Regularization Functions
    #Min-Max Regularization
constant_func = np.vectorize(lambda x: 0)
def minmax_help(transposed_array):
    shift = []; rescale = []
    for i in range(0, len(transposed_array)):
        input = transposed_array[i]
        min_num = min(input); max_num = max(input)
        if min_num == max_num:
            func = constant_func
            shift.append(min_num); rescale.append(0)
        else:
            func = np.vectorize(lambda x: (x - min_num) / (max_num - min_num))
            shift.append(min_num); rescale.append(max_num - min_num)
        transposed_array[i] = func(input)
    regularized = list(transposed_array.transpose())
    return regularized, shift, rescale
def convert_minmax(Data_set: list[list[NDArray[np.float64]]]):
    #Node (Column) -> Node (Row) -> Magic -> Node (Column)
    transposed_I = np.array(Data_set[0]).transpose() #2D matrix
    regularized_I, shift_I, rescale_I = minmax_help(transposed_I)

    transposed_L = np.array(Data_set[1]).transpose()
    regularized_L, shift_L, rescale_L = minmax_help(transposed_L)

    new_data_set = [regularized_I, regularized_L]
    return new_data_set, (shift_I, rescale_I), (shift_L, rescale_L)

    #Z-Score Regularization
constant_func_Z = np.vectorize(lambda x: 0)
threshold = 10**(-6)
def zscore_help(transposed_array):
    shift = []; rescale = []
    for i in range(0, len(transposed_array)):
        input = transposed_array[i]
        mean = np.mean(input); deviation = np.std(input)
        if deviation < threshold:
            func = constant_func_Z
            shift.append(mean); rescale.append(0)
        else:
            func = np.vectorize(lambda x: (x - mean) / deviation)
            shift.append(mean); rescale.append(deviation)
        transposed_array[i] = func(input)
    regularized = list(transposed_array.transpose())
    return regularized, shift, rescale
def convert_zscore(Data_set: list[list[NDArray[np.float64]]]):
    #Node (Column) -> Node (Row) -> Magic -> Node (Column)
    transposed_I = np.array(Data_set[0]).transpose() #2D matrix
    regularized_I, shift_I, rescale_I = zscore_help(transposed_I)

    transposed_L = np.array(Data_set[1]).transpose()
    regularized_L, shift_L, rescale_L =zscore_help(transposed_L)

    new_data_set = [regularized_I, regularized_L]
    return new_data_set, (shift_I, rescale_I), (shift_L, rescale_L)

    #Set new regularization function
def create_regularization_func(feature_scale_I: tuple[list, list], feature_scale_L: tuple[list, list]):
    # Feature Scale (shift, rescale)
    # From Scaled=(x-shift)/(rescale)
    constant_func = np.vectorize(lambda x: 0)
    def convert_func_help(transposed_array, feature_scale_info: tuple[list, list]):
        shift = feature_scale_info[0]; rescale = feature_scale_info[1]
        for i in range(0, len(transposed_array)):
            input = transposed_array[i]
            if rescale[i] == 0: func = constant_func
            else: func = np.vectorize(lambda x: (x - shift[i]) / rescale[i])
            transposed_array[i] = func(input)
        regularized = list(transposed_array.transpose())
        return regularized
    def convert_func(Data_set: list[list[NDArray[np.float64]]]):
        # Node (Column) -> Node (Row) -> Magic -> Node (Column)
        transposed_I = np.array(Data_set[0]).transpose()  # 2D matrix
        regularized_I = convert_func_help(transposed_I, feature_scale_I)
        transposed_L = np.array(Data_set[1]).transpose()
        regularized_L = convert_func_help(transposed_L, feature_scale_L)
        new_data_set = [regularized_I, regularized_L]
        return new_data_set
    return convert_func

clip_threshold = 4
def gradient_clipping(weight_gradients, bias_gradients, gamma_gradients=None, beta_gradients=None):
    if gamma_gradients is None:
        gradient_list = [weight_gradients, bias_gradients]
        used_clip_threshold = clip_threshold/2
    else:
        gradient_list = [weight_gradients, bias_gradients, gamma_gradients, beta_gradients]
        used_clip_threshold = clip_threshold
    all_gradients = np.concatenate([grad.flatten() for grad in gradient_list])
    gradient_L2_norm = np.linalg.norm(all_gradients)
    if gradient_L2_norm > used_clip_threshold:
        return np.divide(used_clip_threshold, gradient_L2_norm)
    else: return 1

#Loss Regularization
standard_lambda = 0.1
L2_lambda = 2*standard_lambda
L1_func = lambda temp_lambda, matrix: temp_lambda*np.sign(matrix)
L2_func = lambda temp_lambda, matrix: temp_lambda*matrix
loss_regularization_func = L2_func

def normalise(X: NDArray[np.float64]):
    mean_vals = X.mean(axis=0); var_vals = X.var(axis=0)
    std_vals = np.sqrt(var_vals+epsilon)
    return (X-mean_vals)/std_vals, mean_vals, std_vals

class layer:
    def __init__(self, weights: int, nodes: int, activation_functions = None, use_BN = False):
        if activation_functions in [Activation_Functions.relu, Activation_Functions.leaky_relu]:
            standard_deviation = np.sqrt(2/weights)
        elif activation_functions in [Activation_Functions.sigmoid]:
            standard_deviation = np.sqrt(2/(weights+nodes))
        else: standard_deviation = 1
        self.weight_matrix = numpy_rng.standard_normal((weights, nodes)) * standard_deviation
        self.bias_matrix = np.zeros(nodes)
        self.activation_functions = activation_functions
        self.weight_gradient = None; self.bias_gradient = None
        self.w_moving_avg_gradients = np.zeros((weights, nodes)); self.w_moving_avg_squared_gradients = np.zeros((weights, nodes))
        self.b_moving_avg_gradients = np.zeros(nodes); self.b_moving_avg_squared_gradients = np.zeros(nodes)
        self.input_result = None; self.output_result = None
        self.Gradient_wrt_Input = None
        self.use_BN = use_BN
        if use_BN:
            self.gamma = np.ones(nodes); self.beta = np.zeros(nodes)
            self.g_moving_avg_gradients = np.zeros(nodes); self.g_moving_avg_squared_gradients = np.zeros(nodes)
            self.B_moving_avg_gradients = np.zeros(nodes); self.B_moving_avg_squared_gradients = np.zeros(nodes)
            self.normalised_result = None; self.AT_result = None; self.batch_std = None
            self.running_mean = None; self.running_std = None; self.momentum = 0.7

    #2-D inputs
    def calc(self, inputs: NDArray[np.float64], is_training=True):
        # 2-D (Row: Input, Column: Nodes)
        # Path*: Input_Result -> Normalised_Result -> Affine_Transformation AT_Result -> Output_Result
        self.input_result = inputs @ self.weight_matrix + self.bias_matrix
        if self.use_BN:
            if is_training:
                if len(inputs[0]) == 1: self.normalised_result = self.input_result; self.batch_std = 1
                else:
                    self.normalised_result, batch_mean, self.batch_std = normalise(self.input_result)
                    if self.running_mean is None: self.running_mean = batch_mean; self.running_std = self.batch_std
                    else:
                        self.running_mean = (self.momentum * self.running_mean) + (1 - self.momentum) * batch_mean
                        self.running_std = (self.momentum * self.running_std) + (1 - self.momentum) * self.batch_std
            else:
                if self.running_mean is None: self.normalised_result = self.input_result
                else: self.normalised_result = (self.input_result - self.running_mean)/self.running_std
            self.AT_result = self.normalised_result * self.gamma + self.beta
            self.output_result = self.activation_functions[0](self.AT_result)
        else: self.output_result = self.activation_functions[0](self.input_result)
        #Warning Mechanism
        if type(self.Gradient_wrt_Input) is not None and np.any(self.Gradient_wrt_Input == 0):
            print("Warning 'Delta' equals to 0")

    #dy is the Output Error Signal
    def backprop_gradients(self, dy, batch_size: int):
        if self.use_BN:
            self.d_gamma = np.sum(dy * self.normalised_result, axis=0) / batch_size
            self.d_beta = np.sum(dy, axis=0) / batch_size
            Local_Gradient = dy * self.gamma
            term2 = Local_Gradient.mean(axis=0)
            term3 = self.normalised_result * (Local_Gradient * self.normalised_result).mean(axis=0)
            self.Gradient_wrt_Input = (1 / self.batch_std) * (Local_Gradient - term2 - term3)
        else: self.Gradient_wrt_Input = dy

    def set_lambda(self, Model_lambda, L2: bool):
        self.temp_lambda = Model_lambda
        if L2: self.sign_func = lambda x: x
        else: self.sign_func = lambda x: np.sign(x)

    def set_ADAM_settings(self, initial_learning_rate, beta_1, beta_2, epsilon):
        self.initial_learning_rate = initial_learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def update_rule(self): #Call after weight and bias gradient are set
        #Loss regularization
        self.weight_gradient += self.temp_lambda*self.sign_func(self.weight_matrix)
        self.bias_gradient += self.temp_lambda*self.sign_func(self.bias_matrix)
        #ADAM
        self.w_moving_avg_gradients = self.beta_1*self.w_moving_avg_gradients + (1-self.beta_1)*self.weight_gradient
        self.w_moving_avg_squared_gradients = self.beta_2*self.w_moving_avg_squared_gradients + (1-self.beta_2)*(self.weight_gradient**2)
        self.b_moving_avg_gradients = self.beta_1*self.b_moving_avg_gradients + (1-self.beta_1)*self.bias_gradient
        self.b_moving_avg_squared_gradients = self.beta_2*self.b_moving_avg_squared_gradients + (1-self.beta_2)*(self.bias_gradient**2)
        m_hat_w = self.w_moving_avg_gradients/(1-self.beta_1); v_hat_w = self.w_moving_avg_squared_gradients/(1-self.beta_2)
        m_hat_b = self.b_moving_avg_gradients/(1-self.beta_1); v_hat_b = self.b_moving_avg_squared_gradients/(1-self.beta_2)
        self.weight_matrix -= (self.initial_learning_rate*m_hat_w)/np.sqrt(v_hat_w+self.epsilon)
        self.bias_matrix -= (self.initial_learning_rate*m_hat_b)/np.sqrt(v_hat_b+self.epsilon)
        if self.use_BN:
            self.g_moving_avg_gradients = self.beta_1 * self.g_moving_avg_gradients + (1 - self.beta_1) * self.d_gamma
            self.g_moving_avg_squared_gradients = self.beta_2 * self.g_moving_avg_squared_gradients + (1 - self.beta_2) * (self.d_gamma ** 2)
            self.B_moving_avg_gradients = self.beta_1 * self.B_moving_avg_gradients + (1 - self.beta_1) * self.d_beta
            self.B_moving_avg_squared_gradients = self.beta_2 * self.B_moving_avg_squared_gradients + (1 - self.beta_2) * (self.d_beta ** 2)
            m_hat_g = self.g_moving_avg_gradients / (1 - self.beta_1); v_hat_g = self.g_moving_avg_squared_gradients / (1 - self.beta_2)
            m_hat_B = self.B_moving_avg_gradients / (1 - self.beta_1); v_hat_B = self.B_moving_avg_squared_gradients / (1 - self.beta_2)
            self.gamma -= (self.initial_learning_rate * m_hat_g) / np.sqrt(v_hat_g + self.epsilon)
            self.beta -= (self.initial_learning_rate * m_hat_B) / np.sqrt(v_hat_B + self.epsilon)

class Model_Initialize_Information:
    def __init__(self, layer_structure: list[int] = None, activation_functions = Activation_Functions.leaky_relu, loss_functions = Loss_Functions.MSE, L_type: int = 0, R_type: int = 0, use_BN = False, is_regression: bool = True):
        self.layer_structure = layer_structure
        #Activation function (and its derivative)
        self.activation_functions = activation_functions
        #Loss function (and its derivative)
        self.loss_functions = loss_functions
        # L_type as in, Loss regularization type (0: None, 1: L1, 2: L2)
        self.L_type = L_type
        # R_type as in, Input Normalization type (0: None, 1: Min-Max, 2: Z-Score)
        self.R_type = R_type
        # Batch Norm
        self.use_BN = use_BN
        # Regression or Classification
        self.is_regression = is_regression
    def check_validity(self):
        #layer_structure
        if 0 in self.layer_structure: raise Exception("'0' included in self.layer_structure; it should be removed.")
        #Activation functions
        if (self.activation_functions is None and self.is_regression is True): raise Exception("self.activation_functions should be grabbed from the 'Activation_Functions' object.")
        #Loss functions
        if (self.loss_functions is None): raise Exception("self.loss_functions should be grabbed from the 'Loss_Functions' object.")
        #L_type
        if self.L_type not in [0, 1, 2]: raise Exception(f"self.L_type value {self.L_type} is not recognised.\nLoss regularization types 0: None, 1: L1, 2: L2.")
        #R_type
        if self.R_type not in [0, 1, 2]: raise Exception(f"self.R_type value {self.R_type} is not recognised.\nInput regularization type 0: None, 1: Min-Max, 2: Z-Score.")
        #use_BN
        if not isinstance(self.use_BN, bool): raise Exception(f"self.use_BN must be a boolean value (True or False).")
        #is_regression
        if type(self.is_regression) is not bool: raise Exception("self.is_regression must be a boolean value (True or False).")
        return True

class Training_Set:
    def __init__(self, Data_Set: list[list[NDArray[np.float64]]], validation_set_percentage = 0):
        validation_amount_goal = int(len(Data_Set[0])*(validation_set_percentage/100))
        self.data_set = Data_Set
        self.validation_set = [[], []]
        for i in range(0, validation_amount_goal):
            random_idx = random.randint(0, len(self.data_set[0])-1)
            self.validation_set[0].append(Data_Set[0][random_idx]); self.validation_set[1].append(Data_Set[1][random_idx])
            self.data_set[0].pop(random_idx); self.data_set[1].pop(random_idx)

class Checkpoint_Model:
    def __init__(self, layers: list[layer]):
        self.layers = copy.deepcopy(layers)
        self.validation_error = np.inf
        self.epoch = None
    def reassign_layers(self, layers: list[layer]):
        self.layers = copy.deepcopy(layers)
    def clear(self):
        self.validation_error = np.inf
        self.epoch = None

class Model5:
    def __init__(self, model_initialize_information: Model_Initialize_Information):
        self.MII = model_initialize_information
        self.use_BN = self.MII.use_BN
        #Format of layers_amount: [input, ..., ..., output]
        self.Layers = [] #Excluding input
        self.input_length = self.MII.layer_structure[0]; self.output_length = self.MII.layer_structure[-1]
        for i in range(1, len(self.MII.layer_structure)):
            temp_layer = layer(self.MII.layer_structure[i-1], self.MII.layer_structure[i], self.MII.activation_functions, self.use_BN)
            self.Layers.append(temp_layer)
        #Checkpoint
        self.checkpoint = Checkpoint_Model(self.Layers)
        if not self.MII.is_regression: self.Layers[-1].activation_functions = Activation_Functions.softmax
        self.length_l = len(self.Layers)
        self.gradient_accumulated = 0
        #Get amount of parameters
        amount_parameters = 0
        for Layer in self.Layers: amount_parameters += Layer.weight_matrix.size + Layer.bias_matrix.size
        match self.MII.L_type:
            case 0: self.temp_lambda = 0
            case 1: self.temp_lambda = standard_lambda/amount_parameters
            case 2: self.temp_lambda = L2_lambda/amount_parameters
        for Layer in self.Layers: Layer.set_lambda(self.temp_lambda, True if self.MII.L_type != 1 else False)
        self.new_input_regularization = None
        match self.MII.R_type:
            case 0: self.new_input_regularization = lambda x: x
            case 1: self.input_regularization = convert_minmax
            case 2: self.input_regularization = convert_zscore
            case _: raise Exception(f"Input Regularization option: '{self.MII.R_type}' is not valid.")
        self.set_ADAM_settings()
        self.feature_scale_I = None; self.feature_scale_L = None

    def set_ADAM_settings(self, initial_learning_rate = 1, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
        self.initial_learning_rate = initial_learning_rate
        self.beta_1 = beta_1; self.beta_2 = beta_2; self.epsilon = epsilon
        for Layer in self.Layers: Layer.set_ADAM_settings(self.initial_learning_rate, self.beta_1, self.beta_2, self.epsilon)

    def forward_pass(self, inputs: NDArray[np.float64]):
        if len(inputs[0]) != self.input_length: raise Exception(f"Input size doesn't match input nodes {len(inputs[0])} != {self.input_length}")
        self.Layers[0].calc(inputs)
        for i in range(1, self.length_l):
            self.Layers[i].calc(self.Layers[i-1].output_result)
        return self.Layers[-1].output_result

    def error_calc(self, expected_output: NDArray[np.float64]):
        if expected_output.ndim == 1:
            if len(expected_output) != self.Layers[-1].bias_matrix.size:
                raise Exception(f"Expected output size doesn't match output nodes: {expected_output.size} != {self.Layers[-1].bias_matrix.size}")
        elif expected_output.ndim == 2:
            if len(expected_output[0]) != self.Layers[-1].bias_matrix.size:
                raise Exception(f"Expected output size doesn't match output nodes: {expected_output.size} != {self.Layers[-1].bias_matrix.size}")
        else: raise Exception(f"Unexpected {expected_output.ndim}-Dimension")
        error_rate = Loss_Functions.RMSE[0](self.Layers[-1].output_result, expected_output)
        return error_rate

    def error_derivative(self, expected_output: NDArray[np.float64]):
        output_layer_activations = self.Layers[-1].output_result
        if self.MII.is_regression: error_derivative = self.MII.loss_functions[1](output_layer_activations, expected_output)
        else: return None
        return error_derivative #2-D array

    def backward_pass(self, inputs: NDArray[np.float64], error_rate, expected_output: NDArray[np.float64], batch_size: int = 1):
        #Find 2-D "deltas" for outer layer
        if self.MII.is_regression: dy = error_rate * self.Layers[-1].activation_functions[1](self.Layers[-1].output_result)
        else: dy = (self.Layers[-1].output_result - expected_output)
        self.Layers[-1].backprop_gradients(dy, batch_size)
        #Compute "deltas" for the rest of the layers
        for i in range(self.length_l-2, -1, -1):
            dy = (self.Layers[i+1].Gradient_wrt_Input @ self.Layers[i+1].weight_matrix.T) * self.Layers[i].activation_functions[1](self.Layers[i].output_result)
            self.Layers[i].backprop_gradients(dy, batch_size)
        #Update parameters
        for i in range(self.length_l-1, 0, -1):
            weight_gradients = self.Layers[i-1].output_result.T.dot(self.Layers[i].Gradient_wrt_Input) / batch_size
            bias_gradients = np.sum(self.Layers[i].Gradient_wrt_Input, axis=0) / batch_size
            if self.use_BN:
                gamma_gradients = self.Layers[i].d_gamma; beta_gradients = self.Layers[i].d_beta
                scaling_factor = gradient_clipping(weight_gradients, bias_gradients, gamma_gradients, beta_gradients)
                self.Layers[i].d_gamma *= scaling_factor; self.Layers[i].d_beta *= scaling_factor
            else: scaling_factor = gradient_clipping(weight_gradients, bias_gradients)
            weight_gradients *= scaling_factor; bias_gradients *= scaling_factor
            self.Layers[i].weight_gradient = weight_gradients; self.Layers[i].bias_gradient = bias_gradients
            self.Layers[i].update_rule()
        weight_gradients = inputs.T.dot(self.Layers[0].Gradient_wrt_Input) / batch_size
        bias_gradients = np.sum(self.Layers[0].Gradient_wrt_Input, axis=0) / batch_size
        if self.use_BN:
            gamma_gradients = self.Layers[0].d_gamma; beta_gradients = self.Layers[0].d_beta
            scaling_factor = gradient_clipping(weight_gradients, bias_gradients, gamma_gradients, beta_gradients)
            self.Layers[0].d_gamma *= scaling_factor; self.Layers[0].d_beta *= scaling_factor
        else: scaling_factor = gradient_clipping(weight_gradients, bias_gradients)
        weight_gradients *= scaling_factor; bias_gradients *= scaling_factor
        self.Layers[0].weight_gradient = weight_gradients; self.Layers[0].bias_gradient = bias_gradients
        self.Layers[0].update_rule()

    #Given a Dataset list[list[np.array]] return list[list[list[np.array]]] where [[batch],...,[batch]] and batch = [inputs, labels]
    def batch_set(self, Dataset: list[list[NDArray[np.float64]]], batch_type, batch_size):
        if batch_type == 0: return [[[Dataset[0][i]], [Dataset[1][i]]] for i in range(len(Dataset[0]))]
        if batch_type == 2: return [Dataset]
        split_inputs = []; split_labels = []; amount = len(Dataset[1])
        perfect_batch_amount = math.floor(amount/batch_size)
        remainder = amount - (perfect_batch_amount * batch_size)
        for i in range(0, perfect_batch_amount):
            split_inputs.append(Dataset[0][i*batch_size:(i+1) * batch_size])
            split_labels.append(Dataset[1][i*batch_size:(i+1) * batch_size])
        if remainder != 0:
            split_inputs.append(Dataset[0][amount- 1 - remainder:amount - 1])
            split_labels.append(Dataset[1][amount- 1 - remainder:amount - 1])
        batched_set = []
        while len(split_labels) > 0:
            batched_set.append([split_inputs[0], split_labels[0]])
            split_inputs.pop(0); split_labels.pop(0)
        return batched_set

    #Validation Function
    def validation_error(self, validation_set: list[list[NDArray[np.float64]]]):
        if len(validation_set[0]) == 0: return None
        validation_error_sum = 0
        for i in range(0, len(validation_set[0])):
            self.run(validation_set[0][i])
            validation_error_sum += self.error_calc(validation_set[1][i])
        return validation_error_sum/len(validation_set[0])

    def shuffle_dataset(self, Z: list[list]):
        C = list(zip(Z[0], Z[1]))
        shuffled_C = global_random.sample(C, len(C))
        X, Y = map(list, zip(*shuffled_C))
        return [X, Y]

    #Auto train
    #Batch types: 0, Stochastic Gradient Descent (SGD); 1, Mini-Batch Gradient Descent; 2, Batch Gradient Descent
    def train(self, epoch: int, training_obj: Training_Set, batch_type: int = 2, batch_size: int = 0, checkpointing = False, tolerance: float = 1e-6, patience: int = 15, silent: bool = True):
        DataSet = training_obj.data_set
        #Input Regularization
        if self.new_input_regularization == None:
            DataSet, feature_scale_I, feature_scale_L = self.input_regularization(DataSet)
            self.feature_scale_I = feature_scale_I; self.feature_scale_L = feature_scale_L
            self.new_input_regularization = create_regularization_func(feature_scale_I, feature_scale_L)
        else: DataSet = self.new_input_regularization(DataSet)
        ValidationSet = self.new_input_regularization(training_obj.validation_set)

        #Set up
        Error_history_batch = []; Error_history_global = []
        ErrorValidation_epoch = []
        Amount = len(DataSet[0]); Steps = epoch * Amount
        best_loss = float('inf'); patience_counter = 0

        print(f"Steps needed until completion: {Steps}, Epochs: {epoch}, Amount: {Amount}")
        #Epoch loops
        for i in range(epoch):
            #Iterate through batches
            Batches = self.batch_set(self.shuffle_dataset(DataSet), batch_type, batch_size)
            for Batch in Batches:
                Inputs = np.array(Batch[0]); Labels = np.array(Batch[1])
                #Send batch in
                self.forward_pass(Inputs)
                error = self.error_calc(Labels) #Float
                Error_history_batch.append(error)
                self.backward_pass(Inputs, self.error_derivative(Labels), Labels, len(Inputs))

                if not silent: print(f"MSE errors: {Error_history_batch}")

            print(f"Steps left: {Steps - (i + 1) * Amount}")
            Error_history_global.append(sum(Error_history_batch)/len(Error_history_batch))
            Error_history_batch.clear()
            #Validation Error
            ErrorValidation_epoch.append(self.validation_error(ValidationSet))
            if ErrorValidation_epoch[-1] is not None:
                #Checkpointing
                if checkpointing and ErrorValidation_epoch[-1] < self.checkpoint.validation_error:
                    self.checkpoint.reassign_layers(self.Layers)
                    self.checkpoint.validation_error = ErrorValidation_epoch[-1]
                    self.checkpoint.epoch = i
                # Early Stopping
                if ErrorValidation_epoch[-1] < best_loss - tolerance:
                    best_loss = ErrorValidation_epoch[-1]
                    patience_counter = 0
                else: patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early Exit at epoch: {i + 1}")
                    break
            #Information
            if not silent: self.show()
        if checkpointing:
            self.Layers = self.checkpoint.layers
            print("Loaded from epoch:", self.checkpoint.epoch); self.checkpoint.clear()
        return Error_history_global, ErrorValidation_epoch

    # One at a time (dtype MUST be np.float64)
    def run(self, inputs: NDArray[np.float64]):
        if inputs.size != self.input_length: raise Exception(f"Input size doesn't match input nodes {inputs.size} != {self.input_length}")
        #Input regularization
        if self.feature_scale_I != None:
            inputs = self.new_input_regularization([[inputs], []])[0][0]
        #Forward Pass
        self.Layers[0].calc(inputs, is_training=False)
        for i in range(1, self.length_l):
            self.Layers[i].calc(self.Layers[i-1].output_result, is_training=False)
        #Output regularization
        output = self.Layers[-1].output_result.tolist()
        if self.feature_scale_L != None:
            output = self.Layers[-1].output_result.tolist()
            for i in range(0, len(output)):
                output[i] = output[i]*self.feature_scale_L[1][i]+self.feature_scale_L[0][i]
        return output

    def clear_out(self):
        for layer_select in self.Layers:
            layer_select.input_result = None; layer_select.normalised_result = None
            layer_select.AT_result = None; layer_select.output_result = None
            layer_select.weight_gradient = None; layer_select.bias_gradient = None
            layer_select.d_beta = None; layer_select.d_gamma = None
            layer_select.batch_std = None; layer_select.Gradient_wrt_Input = None

    def show(self):
        print("\nModel Information:")
        if max(self.MII.layer_structure) <= 10:
            for i in self.Layers:
                print(f"Weight Matrix: \n {i.weight_matrix} \n Bias Matrix: \n {i.bias_matrix}")
                if self.use_BN:
                    print(f"Gamma Matrix: \n {i.gamma} \n Beta Matrix: \n {i.beta}")
                print(f"\n 'Delta' Matrix: \n {i.Gradient_wrt_Input} \n",
                      f"Input Result: \n {i.input_result} \n Output Result: \n {i.output_result} \n")
        else:
            print(f"Layer 0 (input layer): Nodes: {self.input_length}")
            for idx in range(1, self.length_l+1):
                print(f"Layer: {idx}, Nodes: {self.Layers[idx-1].bias_matrix.size}, Weights amount: {self.Layers[idx-1].weight_matrix.size}")
        if self.MII.R_type != 0:
            print("Format: ")
            print("Feature Scale Input:", self.feature_scale_I, "Feature Scale Label:", self.feature_scale_L)
