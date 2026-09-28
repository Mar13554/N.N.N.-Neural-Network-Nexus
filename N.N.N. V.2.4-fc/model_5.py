# N.N.N. V.2.4-fc
from types import NoneType
from library import Matrix, libN
from library import No_Activation, No_Activation_prime, Relu, Relu_prime, LeakyRelu, LeakyRelu_prime, Sigmoid
import math, random, copy
from debug_tool import mat_print

global_seed = 314; global_random = random.Random(global_seed)
def set_global_seed(seed_num: int):
    global global_seed, global_random
    global_seed = seed_num
    global_random = random.Random(seed_num)

#Activation Functions
"""
def softmax_1(x: Matrix):
    z = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(z)/np.sum(np.exp(z), axis=-1, keepdims=True)
def softmax_2(x):
    s = softmax_1(x)
    I = np.eye(s.shape[-1])
    return I * s[..., None, :] - np.einsum('...i,...j->...ij', s, s)
"""
class Activation_Functions_Class:
    def __init__(self):
        #No activation
        self.no_activation = [No_Activation, No_Activation_prime]
        #Sigmoid [0-1]
        self.sigmoid = [Sigmoid, lambda x: x*(1-x)] #x has already been passed through sigmoid
        #Relu
        self.relu = [Relu, Relu_prime]
        #LeakyRelu
        self.leaky_relu = [LeakyRelu, LeakyRelu_prime]
        #Softmax
        #self.softmax = [softmax_1, softmax_2]
Activation_Functions = Activation_Functions_Class()

epsilon = 1e-7 # 0.0000001
#Loss Functions
class Loss_Functions_Class:
    def __init__(self):
        #Both x and y are of class Matrix
        #Format is the same as above, [loss_function, loss_function_derivative]
        self.MSE = [lambda x, y: ((y-x)**2).sum() / y.elements, lambda x, y: (x-y)* (2/y.elements)]
        self.RMSE = [lambda x, y: self.MSE[0](x, y) ** (1/2), None]
        self.MAE = [lambda x, y: (abs(y-x)).sum() / y.elements, lambda x, y: libN.sign(y-x)*(-1/y.elements)]
        #Automatically used for the last layer in classification
            # One output node, Matrix.sum() is just to convert to float
        self.Binary_Cross_Entropy_Loss = [lambda x, y: -(y*libN.log(x)+(1-y)*libN.log(1-x)).sum(), lambda x, y: ((y/libN.clip(x, epsilon, 1 - epsilon))-((1-y)/(1-libN.clip(x, epsilon, 1 - epsilon)))).sum()]
        self.Categorical_Cross_Entropy_Loss = [lambda x, y: (y*libN.log(x)).sum()*(-1), lambda x, y: -(y/x)]
Loss_Functions = Loss_Functions_Class()

#Regularization Functions
    #Min-Max Regularization
def convert_minmax(Data_set: list[list[Matrix]]):
    transposed_I = libN.stack_all(Data_set[0]).T() #2D matrix
    NV_I = libN.min_max(transposed_I)

    transposed_L = libN.stack_all(Data_set[1]).T()
    NV_L = libN.min_max(transposed_L)
    return NV_I, NV_L

    #Z-Score Regularization
def convert_zscore(Data_set: list[list[Matrix]]):
    transposed_I = libN.stack_all(Data_set[0]) #2D matrix
    NV_I = libN.zscore(transposed_I)

    transposed_L = libN.stack_all(Data_set[1])
    NV_L = libN.zscore(transposed_L)

    return NV_I, NV_L

clip_threshold = 4
def gradient_clipping(weight_gradients, bias_gradients, gamma_gradients=None, beta_gradients=None):
    if gamma_gradients is None:
        gradient_list = [weight_gradients, bias_gradients]
        used_clip_threshold = clip_threshold/2
    else:
        gradient_list = [weight_gradients, bias_gradients, gamma_gradients, beta_gradients]
        used_clip_threshold = clip_threshold
    all_gradients = libN.concatenate_all(gradient_list)
    gradient_L2_norm = all_gradients.L2()
    if gradient_L2_norm > used_clip_threshold:
        return used_clip_threshold / gradient_L2_norm
    else: return 1

#Loss Regularization
standard_lambda = 0.1
L2_lambda = 2*standard_lambda
L1_func = lambda temp_lambda, matrix: temp_lambda*libN.sign(matrix)
L2_func = lambda temp_lambda, matrix: temp_lambda*matrix
loss_regularization_func = L2_func

def normalise(X: Matrix):
    mean_vals = X.mean(axis=0); var_vals = X.var(axis=0)
    std_vals = (var_vals+epsilon) ** (1/2)
    return (X-mean_vals)/std_vals, mean_vals, std_vals

class layer:
    def __init__(self, weights: int, nodes: int, activation_functions = None, use_BN = False):
        if activation_functions in [Activation_Functions.relu, Activation_Functions.leaky_relu]:
            standard_deviation = math.sqrt(2/weights)
        elif activation_functions in [Activation_Functions.sigmoid]:
            standard_deviation = math.sqrt(2/(weights+nodes))
        else: standard_deviation = 1
        self.weight_matrix = Matrix(2, (weights, nodes)) * standard_deviation
        self.bias_matrix = Matrix(0, (1, nodes))
        self.activation_functions = activation_functions
        self.weight_gradient = None; self.bias_gradient = None
        self.w_moving_avg_gradients = Matrix(0, (weights, nodes)); self.w_moving_avg_squared_gradients = Matrix(0, (weights, nodes))
        self.b_moving_avg_gradients = Matrix(0, (1, nodes)); self.b_moving_avg_squared_gradients = Matrix(0, (1, nodes))
        self.input_result = None; self.output_result = None
        self.Gradient_wrt_Input = None
        self.use_BN = use_BN
        if use_BN:
            self.gamma = Matrix(1, (1, nodes)); self.beta = Matrix(0, (1, nodes))
            self.g_moving_avg_gradients = Matrix(0, (1, nodes)); self.g_moving_avg_squared_gradients = Matrix(0, (1, nodes))
            self.B_moving_avg_gradients = Matrix(0, (1, nodes)); self.B_moving_avg_squared_gradients = Matrix(0, (1, nodes))
            self.normalised_result = None; self.AT_result = None; self.batch_std = None
            self.running_mean = None; self.running_std = None; self.momentum = 0.7

    #2-D inputs
    def calc(self, inputs: Matrix, is_training=True):
        # 2-D (Row: Input, Column: Nodes)
        # Path*: Input_Result -> Normalised_Result -> Affine_Transformation AT_Result -> Output_Result
        self.input_result = libN.dot(inputs, self.weight_matrix) + self.bias_matrix
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
        if (not isinstance(self.Gradient_wrt_Input, NoneType)) and libN.any(self.Gradient_wrt_Input, 0):
            print("Warning 'Delta' equals to 0")

    #dy is the Output Error Signal
    def backprop_gradients(self, dy, batch_size: int):
        if self.use_BN:
            self.d_gamma = (dy * self.normalised_result).sum(axis=0) / batch_size
            self.d_beta = dy.sum(axis=0) / batch_size
            Local_Gradient = dy * self.gamma
            term2 = Local_Gradient.mean(axis=0)
            term3 = self.normalised_result * (Local_Gradient * self.normalised_result).mean(axis=0)
            self.Gradient_wrt_Input = (1 / self.batch_std) * (Local_Gradient - term2 - term3)
        else: self.Gradient_wrt_Input = dy

    def set_lambda(self, Model_lambda, L2: bool):
        self.temp_lambda = Model_lambda
        if L2: self.sign_func = lambda x: x
        else: self.sign_func = lambda x: libN.sign(x)

    def set_ADAM_settings(self, initial_learning_rate, beta_1, beta_2, epsilon):
        self.initial_learning_rate = initial_learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def update_rule(self): #Call after weight and bias gradient are set
        #Loss regularization
        self.weight_gradient += self.sign_func(self.weight_matrix) * self.temp_lambda
        self.bias_gradient += self.sign_func(self.bias_matrix) * self.temp_lambda
        #ADAM
        self.w_moving_avg_gradients = self.w_moving_avg_gradients * self.beta_1 + self.weight_gradient * (1-self.beta_1)
        self.w_moving_avg_squared_gradients = self.w_moving_avg_squared_gradients * self.beta_2 + (self.weight_gradient**2) * (1-self.beta_2)
        self.b_moving_avg_gradients = self.b_moving_avg_gradients * self.beta_1 + self.bias_gradient * (1-self.beta_1)
        self.b_moving_avg_squared_gradients = self.b_moving_avg_squared_gradients * self.beta_2 + (self.bias_gradient**2) * (1-self.beta_2)
        m_hat_w = self.w_moving_avg_gradients/(1-self.beta_1); v_hat_w = self.w_moving_avg_squared_gradients/(1-self.beta_2)
        m_hat_b = self.b_moving_avg_gradients/(1-self.beta_1); v_hat_b = self.b_moving_avg_squared_gradients/(1-self.beta_2)
        self.weight_matrix -= libN.elementwise_div(m_hat_w*self.initial_learning_rate, (v_hat_w+self.epsilon)**(1/2))
        self.bias_matrix -= libN.elementwise_div(m_hat_b*self.initial_learning_rate, (v_hat_b+self.epsilon)**(1/2))
        if self.use_BN:
            self.g_moving_avg_gradients = self.beta_1 * self.g_moving_avg_gradients + (1 - self.beta_1) * self.d_gamma
            self.g_moving_avg_squared_gradients = self.beta_2 * self.g_moving_avg_squared_gradients + (1 - self.beta_2) * (self.d_gamma ** 2)
            self.B_moving_avg_gradients = self.beta_1 * self.B_moving_avg_gradients + (1 - self.beta_1) * self.d_beta
            self.B_moving_avg_squared_gradients = self.beta_2 * self.B_moving_avg_squared_gradients + (1 - self.beta_2) * (self.d_beta ** 2)
            m_hat_g = self.g_moving_avg_gradients / (1 - self.beta_1); v_hat_g = self.g_moving_avg_squared_gradients / (1 - self.beta_2)
            m_hat_B = self.B_moving_avg_gradients / (1 - self.beta_1); v_hat_B = self.B_moving_avg_squared_gradients / (1 - self.beta_2)
            self.gamma -= (self.initial_learning_rate * m_hat_g) / (v_hat_g + self.epsilon)**(1/2)
            self.beta -= (self.initial_learning_rate * m_hat_B) / (v_hat_B + self.epsilon)**(1/2)

class Model_Initialize_Information:
    def __init__(self, layer_structure: list[int] = None, activation_functions = Activation_Functions.leaky_relu, loss_functions = Loss_Functions.MSE, L_type: int = 0, N_type: int = 0, use_BN = False, is_regression: bool = True):
        self.layer_structure = layer_structure
        #Activation function (and its derivative)
        self.activation_functions = activation_functions
        #Loss function (and its derivative)
        self.loss_functions = loss_functions
        # L_type as in, Loss regularization type (0: None, 1: L1, 2: L2)
        self.L_type = L_type
        # N_type as in, Input Normalization type (0: None, 1: Min-Max, 2: Z-Score)
        self.N_type = N_type
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
        #N_type
        if self.N_type not in [0, 1, 2]: raise Exception(f"self.N_type value {self.N_type} is not recognised.\nInput regularization type 0: None, 1: Min-Max, 2: Z-Score.")
        #use_BN
        if not isinstance(self.use_BN, bool): raise Exception(f"self.use_BN must be a boolean value (True or False).")
        #is_regression
        if type(self.is_regression) is not bool: raise Exception("self.is_regression must be a boolean value (True or False).")
        return True

class Training_Set:
    def __init__(self, Data_Set: list[list[Matrix]], validation_set_percentage = 0):
        validation_amount_goal = int(len(Data_Set[0])*(validation_set_percentage/100))
        self.data_set = Data_Set
        self.validation_set = [[], []]
        for i in range(0, validation_amount_goal):
            random_idx = random.randint(0, len(self.data_set[0])-1)
            self.validation_set[0].append(Data_Set[0][random_idx]); self.validation_set[1].append(Data_Set[1][random_idx])
            self.data_set[0].pop(random_idx); self.data_set[1].pop(random_idx)

def deepcopy_layers(i_layers: list[layer]):
    o_layers = []
    for i_layer in i_layers:
        o_layer = layer(1, 1)
        # Note: the "+ 0" is to deepcopy the matrix in C
        o_layer.weight_matrix = i_layer.weight_matrix + 0
        o_layer.bias_matrix = i_layer.bias_matrix + 0
        o_layer.activation_functions = i_layer.activation_functions
        if i_layer.use_BN:
            print("Deepcopy WIP")
        o_layers.append(o_layer)
    return o_layers
class Checkpoint_Model:
    def __init__(self, layers: list[layer]):
        self.layers = deepcopy_layers(layers)
        self.validation_error = math.inf
        self.epoch = None
    def reassign_layers(self, layers: list[layer]):
        self.layers = deepcopy_layers(layers)
    def clear(self):
        self.validation_error = math.inf
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
        for Layer in self.Layers: amount_parameters += Layer.weight_matrix.elements + Layer.bias_matrix.elements
        match self.MII.L_type:
            case 0: self.temp_lambda = 0
            case 1: self.temp_lambda = standard_lambda/amount_parameters
            case 2: self.temp_lambda = L2_lambda/amount_parameters
        for Layer in self.Layers: Layer.set_lambda(self.temp_lambda, True if self.MII.L_type != 1 else False)
        self.NV_all = None # [NV_I, NV_L]
        match self.MII.N_type:
            case 0: pass
            case 1: self.input_normalization = convert_minmax
            case 2: self.input_normalization = convert_zscore
            case _: raise Exception(f"Input Regularization option: '{self.MII.N_type}' is not valid.")
        self.set_ADAM_settings()
        self.feature_scale_I = None; self.feature_scale_L = None

    def set_ADAM_settings(self, initial_learning_rate = 1, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
        self.initial_learning_rate = initial_learning_rate
        self.beta_1 = beta_1; self.beta_2 = beta_2; self.epsilon = epsilon
        for Layer in self.Layers: Layer.set_ADAM_settings(self.initial_learning_rate, self.beta_1, self.beta_2, self.epsilon)

    def forward_pass(self, inputs: Matrix):
        if inputs.size[1] != self.input_length: raise Exception(f"Input size doesn't match input nodes {inputs.size[1]} != {self.input_length}")
        self.Layers[0].calc(inputs)
        for i in range(1, self.length_l):
            self.Layers[i].calc(self.Layers[i-1].output_result)
        return self.Layers[-1].output_result

    def error_calc(self, expected_output: Matrix):
        if expected_output.ndim in [1, 2]:
            if expected_output.size[1] != self.Layers[-1].bias_matrix.size[1]:
                raise Exception(f"Expected output size doesn't match output nodes: {expected_output.size} != {self.Layers[-1].bias_matrix.size}")
        else: raise Exception(f"Unexpected {expected_output.ndim}-Dimension")
        error_rate = Loss_Functions.RMSE[0](self.Layers[-1].output_result, expected_output)
        return error_rate

    def error_derivative(self, expected_output: Matrix):
        output_layer_activations = self.Layers[-1].output_result
        if self.MII.is_regression: error_derivative = self.MII.loss_functions[1](output_layer_activations, expected_output)
        else: return None
        return error_derivative #2-D array

    def backward_pass(self, inputs: Matrix, error_rate, expected_output: Matrix, batch_size: int = 1):
        #Find 2-D "deltas" for outer layer
        if self.MII.is_regression: dy = libN.elementwise_mul(error_rate, self.Layers[-1].activation_functions[1](self.Layers[-1].output_result))
        else: dy = self.Layers[-1].output_result - expected_output
        self.Layers[-1].backprop_gradients(dy, batch_size)
        #Compute "deltas" for the rest of the layers
        for i in range(self.length_l-2, -1, -1):
            dy = libN.elementwise_mul(self.Layers[i+1].Gradient_wrt_Input @ self.Layers[i+1].weight_matrix.T(), self.Layers[i].activation_functions[1](self.Layers[i].output_result))
            self.Layers[i].backprop_gradients(dy, batch_size)
        #Update parameters
        for i in range(self.length_l-1, 0, -1):
            weight_gradients = libN.dot(self.Layers[i-1].output_result.T(), self.Layers[i].Gradient_wrt_Input) / batch_size
            bias_gradients = (self.Layers[i].Gradient_wrt_Input).sum(axis=0) / batch_size
            if self.use_BN:
                gamma_gradients = self.Layers[i].d_gamma; beta_gradients = self.Layers[i].d_beta
                scaling_factor = gradient_clipping(weight_gradients, bias_gradients, gamma_gradients, beta_gradients)
                self.Layers[i].d_gamma *= scaling_factor; self.Layers[i].d_beta *= scaling_factor
            else: scaling_factor = gradient_clipping(weight_gradients, bias_gradients)
            weight_gradients *= scaling_factor; bias_gradients *= scaling_factor
            self.Layers[i].weight_gradient = weight_gradients; self.Layers[i].bias_gradient = bias_gradients
            self.Layers[i].update_rule()
        weight_gradients = libN.dot(inputs.T(), self.Layers[0].Gradient_wrt_Input) / batch_size
        bias_gradients = (self.Layers[0].Gradient_wrt_Input).sum(axis=0) / batch_size
        if self.use_BN:
            gamma_gradients = self.Layers[0].d_gamma; beta_gradients = self.Layers[0].d_beta
            scaling_factor = gradient_clipping(weight_gradients, bias_gradients, gamma_gradients, beta_gradients)
            self.Layers[0].d_gamma *= scaling_factor; self.Layers[0].d_beta *= scaling_factor
        else: scaling_factor = gradient_clipping(weight_gradients, bias_gradients)
        weight_gradients *= scaling_factor; bias_gradients *= scaling_factor
        self.Layers[0].weight_gradient = weight_gradients; self.Layers[0].bias_gradient = bias_gradients
        self.Layers[0].update_rule()

    #Given a Dataset list[list[Matrix]] return list[list[list[Matrix]]] where [[batch],...,[batch]] and batch = [inputs, labels]
    def batch_set(self, Dataset: list[list[Matrix]], batch_type, batch_size):
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
    def validation_error(self, validation_set: list[list[Matrix]]):
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
        DataSet = training_obj.data_set; ValidationSet = training_obj.validation_set

        #Set Input Normalisation
        if self.MII.N_type != 0:
            if self.NV_all is None:
                Full_Data = [DataSet[0]+ValidationSet[0], DataSet[1]+ValidationSet[1]]
                NV_I, NV_L = self.input_normalization(Full_Data)
                self.NV_all = [NV_I, NV_L]
            for i in range(0, len(DataSet[0])):
                DataSet[0][i] = libN.normalize(DataSet[0][i], self.NV_all[0])
                DataSet[1][i] = libN.normalize(DataSet[1][i], self.NV_all[1])
            for i in range(0, len(ValidationSet[0])):
                ValidationSet[0][i] = libN.normalize(ValidationSet[0][i], self.NV_all[0])
                ValidationSet[1][i] = libN.normalize(ValidationSet[1][i], self.NV_all[1])

        #Set up
        Error_history_batch = []; Error_history_global = []; ErrorValidation_epoch = []
        Amount = len(DataSet[0]); Steps = epoch * Amount
        best_loss = float('inf'); patience_counter = 0

        print(f"Steps needed until completion: {Steps}, Epochs: {epoch}, Amount: {Amount}")
        #Epoch loops
        for i in range(epoch):
            #Iterate through batches
            Batches = self.batch_set(self.shuffle_dataset(DataSet), batch_type, batch_size)
            for Batch in Batches:
                # Convert to 2D
                Inputs = libN.stack_all(Batch[0]); Labels = libN.stack_all(Batch[1])
                #Send batch in
                self.forward_pass(Inputs)
                error = self.error_calc(Labels) #Float
                Error_history_batch.append(error)
                self.backward_pass(Inputs, self.error_derivative(Labels), Labels, Inputs.size[0])

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
    def run(self, inputs: Matrix):
        if (inputs.size)[1] != self.input_length: raise Exception(f"Input size doesn't match input nodes {inputs.size} != {self.input_length}")
        #Input regularization
        if self.feature_scale_I != None: ...
        #Forward Pass
        self.Layers[0].calc(inputs, is_training=False)
        for i in range(1, self.length_l):
            self.Layers[i].calc(self.Layers[i-1].output_result, is_training=False)
        #Output regularization
        output = self.Layers[-1].output_result
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
                print(f"Weight Matrix: \n {i.weight_matrix.format()} \n Bias Matrix: \n {i.bias_matrix.format()}")
                if self.use_BN:
                    print(f"Gamma Matrix: \n {i.gamma} \n Beta Matrix: \n {i.beta}")
                print(f"\n 'Delta' Matrix: \n {i.Gradient_wrt_Input.format()} \n",
                      f"Input Result: \n {i.input_result.format()} \n Output Result: \n {i.output_result.format()} \n")
        else:
            print(f"Layer 0 (input layer): Nodes: {self.input_length}")
            for idx in range(1, self.length_l+1):
                print(f"Layer: {idx}, Nodes: {self.Layers[idx-1].bias_matrix.size}, Weights amount: {self.Layers[idx-1].weight_matrix.size}")
        if self.MII.N_type != 0:
            print("Format: ")
            print("Feature Scale Input:", self.feature_scale_I, "Feature Scale Label:", self.feature_scale_L)
