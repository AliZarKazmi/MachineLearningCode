'''
    We used Gradient descent to optimize weights in a simple model 

    Now we will add a technique called "Back Propogation " to calculate the slopes you need to optimize 
    more complex deep learning models,

    *Working Mechanism: 
    "Forward Propogation " sends input data through the hidden layers and into the  output layer,
    "backward propogation" takes the error from the output layer propogates it backward through the hidden layers 
    towared the input layers. It allows gradient descent to update all weights in neural network (by getting gradients for all weights). 


    Backward propogation focous on the general structure of the algorithm rather than trying to memorize every mathematical details.

    ** The main purpose of Backward Propogation is to estimate the slope of the loss function w.r.t each weight.
    ** We always do forward propogation to make prediction and calculate an error before we do backward propogation.

    *** For backward propogation we go back one layer at a time and each time we go back a layer , we'll use a formula for slopes that we already learned.

    *** 3 things we need to multiply to get the slope for that weight are   
        1. The value at the weights, input node (i.e Node value feeding into that weight)
        2. The slope from plotting the loss function against that weight's output node (i.e Slope of loss function w.r.t node it feeds into)
        3. The slope of the activation function at the weight's output (i.e Slope of the activation functiont at the node it feeds into) 



    Backpropogation : Recap 
    1. Start at some random set of weights
    2. Use forward propogation to make a prediction
    3. Use backward propogation to calculate the slope of the loss function w.r.t each weight
    4. Multiply that slope by the learining rate and subtract from the current weights
    5. Keep going with that cycle untill we get to a flat part




    We learned about utilizing models built with the Keras library for deep learning tasks. This included how to save a trained model, reload it for future use, make predictions with the reloaded model, and verify its architecture. Specifically, you explored:

1. Saving and Reloading Models: You discovered that models can be saved using the .save() method with a filename, typically with an .h5 extension for the HDF5 format. Reloading a saved model into memory is done with the load_model function.
2. Making Predictions: You learned how to use a reloaded model to make predictions. For classification models, predictions output the probability distribution across classes. You practiced extracting the probability of a specific outcome using NumPy indexing.
3. Verifying Model Architecture: You found out how to print a summary of a model's architecture with the .summary() method to ensure it matches expectations.







** Creating predictions on new data using the .predict() method.
** Extracting the predicted probability of a specific event being true with predicted_prob_true = predictions[:,1].



This code snippet demonstrates the process:

# Calculate predictions: predictions
predictions = model.predict(pred_data)

# Calculate predicted probability of survival: predicted_prob_true
predicted_prob_true = predictions[:,1]

# Print predicted_prob_true
print(predicted_prob_true)
'''