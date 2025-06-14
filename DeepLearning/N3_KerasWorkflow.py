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

# Compile the model
model.compile(optimizer = 'adam', loss ='mean_squared_error')


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
                3. Fit a Model

            Q:What is fitting a Model:
            Ans: That is applying back-propagation and gradient descent with your data to update the weights    
            The fit step looks similar to what you have seen in scikit-learn though it has more options.

            ***Scaling data before fitting can ease optimization
'''

n_cols = predictors.shape[1]

model = Sequential()

# Add the first layer
model.add(Dense(100, activation='relu', input_shape = (n_cols,)))

model.add(Dense(100, activation = 'relu'))

#the last layer has 1 node which is the Output layer / adding output layer
model.add(Dense(1))

# Compile the model
model.compile(optimizer = 'adam', loss ='mean_squared_error')

# Fit the model 
model.fit(predictors, target)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----Classification Models in Deep Learning

''''
    For Classification we do couple of things differently , the biggest change are : first you set the loss function as 'categorical_crossentrophy' instead of 'mean_squared_error'.
    This is not the only possible loss function for classification problems, but it is by far the most common.

    ***FOR categorical crossentropy loss function, a lower score is better , but it is still hard to interpret. So I've added this argument "metrics equal accuracy".

    "metrics equal accuracy" : this means I want to print out the accuracy score at the end of each epoch, which makes it easier to see and understand the models progress.

    **The second thing we do is you to modify the last layer, so it has a separate node for each potential outcome. You also change
    the activation function to "softmax".

    'softmax ' activation function ensures the predictions sum to 1 , so they can be interpreted as probabilities. 

    ***Key points to remember which are different for Classification:
    1."categorical_crossentrophy" loss function
    2. Similar to log loss i.e Lower is better
    3. Add "metrics = ['accuracy'] to compile step for easy-to understand diagnostics.
    4. Output layer has separate node for each possible outcome, and uses "softmax" activation.


Remember : In general we want to convert categoricals in Keras to a format with a separate column for each output.  
'''

#           ->Code for Classification

#we use 'to_categorical' which is used to convert the data from 1 column to multiple columns.
from tensorflow.keras.utils import to_categorical

data = pd.read_csv('basketball_shot_log.csv')

predictors = data.drop(['shot_result'], axis = 1 ).values

# Convert the target to categorical: target
target = to_categorical(data['short_result'])

model = Sequential()

model.add(Dense(100, activation = 'relu', input_shape = (n_cols ,)))

model.add(Dense(100, activation = 'relu'))

model.add(Dense(100, activation = 'relu'))

# output layer for Cassification
model.add(Dense(2, activation = 'softmax'))

# Compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fit the model
model.fit(predictors, target)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

''''
            ->Using Model
        The things you will want to do in order to use these models are
        1. "Save" a model after you have trained it
        2. "Reload" that model
        3. "Make Predictions" with that model

Code is below
'''

from tensorflow.keras.models import load_model

#models are saved in format called 'hdf5', for which h5 is common extension
model.save('model_file.h5')

#Then load the model back into memory with the laod_model function
my_model = load_model('model_file.h5')

# Calculate predictions: predictions
predictions = my_model.predict(data_to_predict_with)


# Calculate predicted probability of the target column
probability_true = predictions[:,1]

#verify model structure, we can print out a summary of the model architecture with the summary method.
my_model.summary()
