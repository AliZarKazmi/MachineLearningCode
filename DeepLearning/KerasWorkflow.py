'''
    The keras workflow has 4 steps:
    1. First we "specify the architecture" : which is things like : how many layers do you want? how many nodes in each layer? What activation function do you want to use in each layer?
    2. "Compile the model" : this specifies the loss function and some details about how optimization works
    3. "Fit the model" : which is that cycle of back-propogation and optimization of model weights with your data
    4. "Prediction" : want to use your model to make prediction

    Below is the code for each Workflow
'''
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

''''
                            -> 1. Specify the Architecture (Model Specification)
'''

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

#reading the data so we can find the numbers of ndoes in the input layer , that is stored as the variable 'n_cols'
''''
    we always need to specify how many columns are in the input when building a Keras model, because that is the number
    of nodes in the input layer.
'''
predictors = np.loadtxt('predictors_data.csv', delimiter = ',')
# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

'''
There are 2 ways to build up a model and we are focusing on sequential which is the Easier way  to build a model

    Sequential models require that each layer has weights or connections only to the 1 layer comming direclty after it in the network digram
'''

# Set up the model: model
model = Sequential()

#Adding layers using ".add()" & type of layer we used which is standard  layer type is called "Dense Layer" 
# it is called Dense layer because all the nodes in the previous layer connect to all of the nodes in the current layer 

#in each layer we specify the number of ndoes as the first positional argument and the activation function we want to use in that layer using the keyword argument "activation"
#in first layer we need to specify input shape as shown below that says the input will have "n_cols " columns and there is nothing after the comma, meaning it can have numbers of rows,
#that is any number of data points

# Add the first layer
model.add(Dense(100, activation='relu', input_shape = (n_cols,)))

# Add the second layer
model.add(Dense(32, activation='relu', input_shape = (n_cols,)))

model.add(Dense(100, activation = 'relu'))

#the last layer has 1 node which is the Output layer / adding output layer
model.add(Dense(1))

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

''''
                            -> 2. Compiling a Model

                After the specifing the model , the next task is to compile it which set up the network for optimization for instance creating an internal function to do back-propagation
                efficiently


                Q: Why you need to compile your model:
                Ans: The compile methods has 2 important arguments for you to choose 
                    1.Specify the Optimizer , which controls the learning rate 
                    2. Loss Function , mean_squared_error is the most common choice for regression problems
'''

#           ->Compiling a model
n_cols = predictors.shape[1]

model = Sequential()

# Add the first layer
model.add(Dense(100, activation='relu', input_shape = (n_cols,)))

model.add(Dense(100, activation = 'relu'))

#the last layer has 1 node which is the Output layer / adding output layer
model.add(Dense(1))

model.compile(optimizer = 'adam', loss ='mean_squared_error')


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
                3. Fit a Model

            Q:What is fitting a Model:
            Ans: That is applying back-propagation and 
'''