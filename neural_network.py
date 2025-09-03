import numpy as np

class NeuralNetwork:
    def __init__(self, input_func, config, label, num_inputs, num_outputs, num_layers, num_neurons, learning_rate, forgetting_factor):
        # General parameters
        self.time_step_delta = config['time_step_delta']
        self.time_steps = int(config['final_time'] / self.time_step_delta)
        self.input_func = input_func
        self.label = label
        
        # Neural network parameters
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.forgetting_factor = forgetting_factor
        
        # Neural network weights
        self.initialize_weights()
        self.neural_network_gradient_wrt_weights = None
        
        self.learning_rate = learning_rate
        self.neural_network_output = None

    def initialize_weights(self):
        np.random.seed(0)
        weights_list = []
        weights_list.append(self.kaiming_he_initialization(self.num_inputs, self.num_neurons))
        for _ in range(self.num_layers - 1):
            weights_list.append(self.kaiming_he_initialization(self.num_neurons, self.num_neurons))
        weights_list.append(self.kaiming_he_initialization(self.num_neurons, self.num_outputs))
        self.weights = np.vstack(weights_list)
        
    def kaiming_he_initialization(self, input_size, output_size):
        return np.random.normal(0, np.sqrt(2 / input_size), output_size * (input_size + 1)).reshape(-1,1)
    
    def construct_transposed_weight_matrices(self):
        weight_matrices = []
        current_index = 0
        biased_num_inputs = self.num_inputs + 1
        biased_num_neurons = self.num_neurons + 1
        
        # Create V1.T (num_neurons x num_inputs + 1 matrix)
        matrix = np.array(self.weights[current_index:current_index + biased_num_inputs * self.num_neurons]).reshape(biased_num_inputs, self.num_neurons, order='F')
        weight_matrices.append(matrix.T)
        current_index += biased_num_inputs * self.num_neurons

        # Create V2.T to VL.T (num_neurons x num_neurons + 1 matrices)
        for _ in range(1, self.num_layers):
            matrix = np.array(self.weights[current_index:current_index + biased_num_neurons * self.num_neurons]).reshape(biased_num_neurons, self.num_neurons, order='F')
            weight_matrices.append(matrix.T)
            current_index += biased_num_neurons * self.num_neurons

        # Create V(L+1).T (num_outputs x num_neurons + 1 matrix)
        matrix = np.array(self.weights[current_index:current_index + biased_num_neurons * self.num_outputs]).reshape(biased_num_neurons, self.num_outputs, order='F')
        weight_matrices.append(matrix.T)
        return weight_matrices
    
    def perform_forward_propagation(self, transposed_weight_matrices, neural_network_input_with_bias):
        activated_output_layers = [neural_network_input_with_bias]
        unactivated_output_layers = []
        activated_output = []
        for layer_index in range(0, self.num_layers + 1):
            product = transposed_weight_matrices[layer_index] @ activated_output_layers[layer_index]
            unactivated_output_layers.append(product)
            if layer_index != self.num_layers:
                if layer_index == self.num_layers - 1:
                    activated_output = self.apply_activation_function_and_bias(product, 'tanh')
                else:
                    activated_output = self.apply_activation_function_and_bias(product, 'swish')
                activated_output_layers.append(activated_output)

        return activated_output_layers, unactivated_output_layers

    def perform_backward_propagation(self, activated_output_layers, unactivated_output_layers, transposed_weight_matrices):
        neural_network_gradient_wrt_weights, product = None, None
        for layer in range(self.num_layers, -1, -1):
            if layer == self.num_layers:
                transposed_layer_outputs = activated_output_layers[layer].T
                neural_network_gradient_wrt_weights = np.kron(np.eye(self.num_outputs), transposed_layer_outputs)
                product = transposed_weight_matrices[layer] @ self.apply_activation_function_derivative_and_bias(unactivated_output_layers[layer-1], 'tanh')
            else:
                transposed_layer_outputs = activated_output_layers[layer].T
                neural_network_gradient_wrt_weights = np.hstack((product @ np.kron(np.eye(self.num_neurons), transposed_layer_outputs), neural_network_gradient_wrt_weights))

                if layer != 0:
                    product = product @ transposed_weight_matrices[layer] @ self.apply_activation_function_derivative_and_bias(unactivated_output_layers[layer-1], 'swish')
        self.neural_network_gradient_wrt_weights = neural_network_gradient_wrt_weights

    # --- CORRECTED FUNCTION SIGNATURE ---
    def compute_neural_network_output(self, step, tracking_error, network_input):
        # Add bias term to the input provided by the controller
        input_with_bias = np.append(network_input, 1).reshape(-1, 1)

        transposed_weight_matrices = self.construct_transposed_weight_matrices()
        
        # Perform forward propagation using the input with bias
        activated_output_layers, unactivated_output_layers = self.perform_forward_propagation(transposed_weight_matrices, input_with_bias)
        
        self.perform_backward_propagation(activated_output_layers, unactivated_output_layers, transposed_weight_matrices)
        nn_output = unactivated_output_layers[-1]
        self.update_neural_network_weights(tracking_error)
        self.neural_network_output = nn_output
        return nn_output

    def update_neural_network_weights(self, tracking_error):           
        loss = tracking_error.reshape(-1, 1)
        if self.label == "DNN1":
           weights_dot =  self.learning_rate * (self.neural_network_gradient_wrt_weights.T @ loss - self.forgetting_factor * self.weights) 
        if self.label == "DNN2":
            weights_dot =  0.5 * self.learning_rate * ( (loss.T @ loss) * self.neural_network_gradient_wrt_weights.T - self.forgetting_factor * self.weights)
        if self.label == "DNN3":
            weights_dot =  self.learning_rate * (self.neural_network_gradient_wrt_weights.T @ loss - self.forgetting_factor * self.weights)
        if self.label == "NN1":
            weights_dot =  self.learning_rate * (self.neural_network_gradient_wrt_weights.T @ loss - self.forgetting_factor * self.weights) 
        if self.label == "NN2":
            weights_dot =  0.5 * self.learning_rate * ( (loss.T @ loss) * self.neural_network_gradient_wrt_weights.T - self.forgetting_factor * self.weights)
        if self.label == "NN3":
            weights_dot =  self.learning_rate * (self.neural_network_gradient_wrt_weights.T @ loss - self.forgetting_factor * self.weights)
        
        self.weights += self.time_step_delta * weights_dot
        
    @staticmethod
    def apply_activation_function_and_bias(x, activation_function):
        if activation_function == 'tanh':
            result = np.tanh(x)
        elif activation_function == 'swish':
            result = x * (1.0 / (1.0 + np.exp(-x)))
        return np.vstack((result, [[1]]))

    @staticmethod
    def apply_activation_function_derivative_and_bias(x, activation_function):
        if activation_function == 'tanh':
            result = 1 - np.tanh(x)**2
        elif activation_function == 'swish':
            sigmoid = 1.0 / (1.0 + np.exp(-x))
            swish = x * sigmoid
            result = swish + sigmoid * (1 - swish)
        diag_result = np.diag(result.flatten())
        return np.vstack((diag_result, np.zeros(diag_result.shape[1])))
