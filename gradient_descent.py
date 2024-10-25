from matplotlib import pyplot as plt
import numpy as np

from mse_function import mean_squared_error


def gradient_descent(x, y, epochs = 10, learning_rate = 0.0001, 
					stopping_threshold = 1e-6):
	
	# Initializing weight, bias, learning rate and epochs
	m = np.random.randn()
	c = np.random.randn()
	epochs = epochs
	learning_rate = learning_rate
	n = float(len(x))
	
	costs = []
	weights = []
	previous_error = None
	
	# Estimation of optimal parameters 
	for i in range(epochs):
		
		# Making predictions
		y_predicted = (m * x) + c
		
		# Calculating the current cost
		current_error = mean_squared_error(y, y_predicted)
       

		# If the change in cost is less than or equal to 
		# stopping_threshold we stop the gradient descent
		if previous_error and abs(previous_error-current_error)<=stopping_threshold:
			break
		print(f"Iteration {i+1}: error {current_error}")
		previous_error = current_error

		costs.append(current_error)
		weights.append(m)
		
		# Calculating the gradients
		weight_derivative = -(2/n) * sum(x * (y-y_predicted))
		bias_derivative = -(2/n) * sum(y-y_predicted)
		
		# Updating weights and bias
		m = m - (learning_rate * weight_derivative)
		c = c - (learning_rate * bias_derivative)
				
	
	
	
	return m, c