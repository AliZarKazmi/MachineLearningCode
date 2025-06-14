'''
    we have learn that 'Forward propagation alg' is used to make prediction in neural network

        * 'Weights' values have significant importance in neural network for model predictions

        * Loss Function which aggregate all the errors into a single measure of the model's predictive performance
        ** Lower Loss Function value means a better model, so our GOAL  is to find the weights that give the lowest value   
        for loss function.
        *** We do this with the Algorithmm called "Gradient descent"



        Remeber : Optimization of Netural Network is done by "Backward Propogation"

        Q: What is Backward Propogation:
        Ans: Backward propagation helps adjust the weights of a neural network to minimize the difference between the predicted output and the actual output, thereby improving the model's accuracy. 

         

Key points to Remeber:

1. The importance of model weights in making accurate predictions. Adjusting weights can significantly change the model's output.
2. The concept of a loss function, which aggregates all prediction errors into a single measure, helping to evaluate the model's performance.
3. Gradient descent, an algorithm used to find the set of weights that minimizes the loss function. It involves starting with random weights, calculating the slope (or gradient) of the loss function at those weights, and then adjusting the weights in the direction that reduces the loss.
4. Practical exercises where you calculated model errors, understood how changing weights affects model accuracy, and coded weight adjustments to see their impact on accuracy.
'''


#       -> Code to calculate the Slopes and Update the Weights
import numpy as np

weights = np.array([1,2])

input_data = np.array([3,4])

target = 6 

# Set the learning rate: learning_rate
learning_rate = 0.01

# Calculate the predictions: preds
preds = (weights * input_data).sum

# Calculate the error: error
error = preds -target

# Print the original error
print('Original Error ' + error)


#calculating Slope / Gradient
gradient = 2 * input_data * error

# Now we can use this value (i.e 'slop value' or 'gradient value' to improve the weights of the model)
print(gradient)  



''''
If you add the slopes to your weights, you will move in the right direction. However, it's possible to move too far in that direction. 
So you will want to take a small step in that direction first, using a lower learning rate, and verify that the model is improving.
'''
# Update the weights: weights_updated
weights_updated = weights - learning_rate * gradient 

# Get updated predictions: preds_updated
preds_updated = (weights_updated * input_data).sum()

# Calculate updated error: error_updated
error_updated = preds_updated - target 


'''
Remeber : ***** Updating the model weights did indeed decrease the error!
'''
# Print the Updated error
print("Updated Error" +error_updated)
