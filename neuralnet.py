from numpy import exp, array, random, dot, tanh

train_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
train_outputs = array([[0, 1, 0, 1]]).T

random.seed(1)

# 3 input connections, 1 output connection (3x1) matrix
#for values from a to b, random function defined as : (b - a) * random_sample() + a
#here the range is from -1 to 1, so a = -1 and b = 1

synaptic_weights = 2 * random.random((3, 1)) - 1

def train(train_inputs, train_outputs, iterations):
        for iteration in range(iterations):

            global synaptic_weights

            #getting output
            x = dot(train_inputs, synaptic_weights)
            output = activate(x)

         	#calculating error
            error = train_outputs - output

            #calculating the adjustment
            adjustment = dot(train_inputs.T, error * activate_grad(x))
            synaptic_weights += adjustment


def current_activator(x):
    return tanh(x)

def current_grad(x):
    return tanh_grad(x)

def activate(x):
    fx = current_activator(x)

    # return squared function if range is not positive
    if (fx[0] > 0):
        return fx
    else:
        return fx ** 2

def activate_grad(x):
    fx = current_activator(x)
    dfx = current_grad(x)

    # return squared function if range is not positive
    if (dfx[0] > 0):
        return dfx
    else:
        return fx * dfx * 2

def sigmoid(x):
	return 1 / (1 + exp(-x))

def gaussian(x):
    return exp(-(x ** 2))

def sig_grad(x):
    return sigmoid(x) * (1 - sigmoid(x))

def gauss_grad(x):
    return -(2 * x * gaussian(x))

def softsign(x):
    return x / (1 + abs(x))

def softsign_grad(x):
    return x / ((1 + abs(x)) ** 2)

def tanh_grad(x):
    return 1 - (tanh(x) ** 2)


print("Random starting synaptic weights - ")
print(synaptic_weights)


train(train_inputs, train_outputs, 10000)


print("synaptic weights after training: ")
print(synaptic_weights)

# Testing neural network 
print("For testing output [1, 0, 0]:  ")
fx = activate(dot(array([1,1,0]), synaptic_weights))
print(fx)
