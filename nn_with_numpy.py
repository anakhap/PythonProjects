#https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
#building an operational neural network using Numpy
#need to be able to train our network and make predictions with it

nn_architecture = [
    {"input_dim": 2, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
]

def init_layers(nn_architecture, seed = 99):
	np.random.seed(seed)
	number_of_layers = len(nn_architecture)
	params_values = {}

	for idx, layer in enumerate(nn_architecture):
		layer_idx = idx+1
		layer_input_size = layer["input_dim"]
		layer_output_size = layer["output_dim"]

		params_values['W' + str(layer_idx)] = np.random.randn(layer_output_size, layer_input_size) * 0.1
		params_values['b' + str(layer_idx)] = np.random.randn(layer_output_size, 1) * 0.1

	return params_values

#initiation of layers parameters
#weight values are initialized with different random numbers as to avoid the breaking symmetry problem
#if weights are the same, no matter what was the input X, all units in the hidden layer will be the same too

#Activation Functions
#gives non-linearity and expressiveness 
#forward and backward propogation

def sigmoid(Z):
	return 1/(1+np.exp(-Z))

def relu(Z):
	return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
	sig = sigmoid(Z)
	return dA * sig * (1-sig)

def relu_backward(da, Z):
	dZ = np.array(dA, copy = True)
	dZ[Z <= 0] = 0;
	return dZ;

#Forward propogation
def singly_layer_forward_propogation(A_prev, W_curr, b_curr, activation="relu"):
	Z_curr = np.dot(W_curr, A_prev) + b_curr

	if activation is "relu":
		activation_func = relu
	elif activation is "sigmoid": 
		activation_func = sigmoid
	else:
		raise Exception('Non-supported activation function')

	return activation_func(Z_curr), Z_curr

#using a pre-preapred one layer step forward function, can now build a whole forward propogation step
#role: perform predictions and organize collection of intermediate values
def full_forward_propogation(X, params_values, nn_architecture):
	memory={}
	A_curr=X

	for idx, layer in enumerate(nn_architecture):
		layer_idx = idx+1
		A_prev = A_curr

		activ_function_curr = layer["activation"]
		w_curr = params_values["W" + str(layer_idx)]
		b_curr = params_values["b" + str(layer_idx)]
		A_curr, Z_curr = single_layer_forward_propogation(A_prev, W_curr, b_curr, activ_function_curr)

		memory["A" + str(idx)] = A_prev
		memory["Z" + str(layer_idx)] = Z_curr

	return A_curr, memory

#Loss Function: designed to show how far we are from the 'ideal' solution
def get_cost_value(Y_hat, Y):
	m = Y_hat.shape[1]
	cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1-Y, np.log(1-Y_hat).T))
	return np.squeeze(cost)

def get_accuracy_value(Y_hat, Y):
	Y_hat_ = convert_prob_into_class(Y_hat)
	return (Y_hat == Y).all(axis=0).mean()

#Backward Propogation
#recursive use of a chain rule known from differential calculus - calculate a derivative of functions created
#by assembling other functions, whose derivatives we already know

def single_layer_backward_propogation(da_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
	m = A.prev_shape[1]

	if activation is "relu":
		backward_activation_func = relu_backward
	elif activation is "sigmoid":
		backward_activation_func = sigmoid_backward
	else:
		raise Exception('Non-supported activation function')

	dZ_curr = backward_activation_func(dA_curr, Z_curr)
	dW_curr = np.dot(dZ_curr, A_prev.T) / m
	db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
	dA_prev = np.dot(W_curr.T, dZ_curr)

	return dA_prev, dW_curr, db_curr

def full_backward_propogation(Y_hat, Y, memory, params_values, nn_architecture):
	grads_values = {}
	m = Y.shape[1]
	Y = Y.reshape(Y_hat.shape)

	dA_prev = -(np.divide(Y, Y_hat) - np.divide(1-Y, 1-Y_hat));

	for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
		layer_idx_curr = layer_idx_prev + 1
		activ_function_curr = layer["activation"]

		dA_curr = dA_prev

		A_prev = memory["A" + str(layer_idx_prev)]
		Z_curr = memory["Z" + str(layer_idx_curr)]
		W_curr = params_values["W" + str(layer_idx_curr)]
		b_curr = params_values["b" + str(layer_idx_curr)]

		dA_prev, dW_curr, db_curr = single)single_layer_backward_propogation(
			dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

		grads_values["dW" + str(layer_idx_curr)] = dW_curr
		grads_values["db" + str(layer_idx_curr)] = db_curr

	return grads_values

#params_values: stores the current values of parameters
#grads_values: stores cost function derivatives calculated wrt these parameters

def update(params_values, grads_values, nn_architecture, learning_rate):
	for layer_idx, layer in enumerate(nn_architecture):
		params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]
		params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

	return params_values;

def train(X, Y, nn_architecture, epochs, learning_rate):
	params_values = init_layers(nn_architecture, 2)
	cost_history = []
	accuracy_history = []

	for i in range(epochs):
		Y_hat, cashe = full_forward_propogation(X, params_values, nn_architecture)
		cost = get_cost_value(Y_hat, Y)
		cost_history.append(cost)
		accuracy = get_accuracy_value(Y_hat, Y)
		accuracy_history.append(accuracy)

		grads_values = full_backward_propogation(Y_hat, Y, cashe, params_values, nn_architecture)
		params_values = update(params_values, grads_values, nn_architecture, learning_rate)

	return params_valuesm cost_history, accuracy_history







