# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradient_descent as gd


def main():
	
	# Data

    data = pd.read_csv('Nairobi Office Price Ex.csv')
    X = data['SIZE']
    Y = data['PRICE']
	
	# Estimating weight and bias using gradient descent
    m, c = gd.gradient_descent(X, Y, epochs=10)
    print(f"Estimated m: {m}\nEstimated c: {c}")

	# Making predictions using estimated parameters
    Y_pred = m*X + c
    
    
    office_size = 100
    predicted_price = m* office_size + c
    print(f'Predicted price for 100 sq.ft office is: {predicted_price}')
    

	# Plotting the regression line
    plt.figure(figsize = (8,6))
    plt.scatter(X, Y, marker='o', color='red')
    plt.plot(X, Y_pred, color='blue', label='Regression Line')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    
    
    

	
if __name__=="__main__":
	main()
