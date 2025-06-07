'''
        Neural Network has 3 layers
            1. Input layer : The input features
            2. Output Layer : Output features (the prediction from our model)
            3. Hidden Layer: The all layers that are not input or output layer are called hiden layer. They called hiden layers
                            because while the inputs and outputs correspond to visible things that happened in the world and they are stored as the data,
                            the values in the hidden layer aren't something we have data about or anything  we observe directly from the world.


                        -> (**** Each "Dot" called a 'Node' in the hiden layer, represent an aggregation of information from our input data and each node adds to the model's ability 
                        to capture interactions.) (**** So the more nodes we have , the more interactions we capture.)

'''

#-------------------------------------------------------------------------------------------------------------------------
''''
            -> Forward Propogation Algorithm (Learn how Neural Network use data to make predictions)
                * Multiply  - add process
                * Dot Product 
                * Forward propagation for one data point at a time
                * Output is the prediction for that data point 

'''

#        -> Code for Forward Propogation
import numpy as np

input_data = np.array([2,3])

weights = {'node_0' : np.array([1,1]),
            'node_1' : np.array([11,1]),
            'output' : np.array([2,-1])
            }

# Calculate node 0 value: node_0_value
node_0_value = (input_data * weights['node_0']).sum()

# Calculate node 1 value: node_1_value
node_1_value = (input_data * weights['node_1']).sum()

# Put node values into array: hidden_layer_outputs
hidden_layer_values = np.array([node_0_value, node_1_value])
print(hidden_layer_values)

# Calculate output: output
output = (hidden_layer_values * weights['output']).sum()
print(output)
#-------------------------------------------------------------------------------------------------------------------------


'''
        -> Activation Function (Much more better than Forward Propogation)
            * Activation Function allows the model to capture non-linearities 
            * Non-linearities capture patterns i.e how going from no child to 1 to 2 children may impact your banking transactions
              differently than going from 3 children to four.
            * if the relationship in data aren't straight-line relationships, we will need an activation function that capture non-linearities.


    Q: what is Activation Function:
    Ans:  (** an "activation function" is a function applied at each node. It converts the node's input into some output.)
    
    
    An activation function is something applied to the value coming into a node which then transforms it into the value stored in that node,
    or the node output

----------------------------------------------------------------------------------------------------------------------------------------------------------

                -> **** ReLU (Rectified Linear Activation Function)

                The rectified linear activation function (called ReLU) has been shown to lead to very high-performance networks. 
                This function takes a single number as an input, returning 0 if the input is negative, and the input if the input is positive.

Here are some examples:
relu(3) = 3
relu(-3) = 0
                        It has 2 linear pieces, it's surprisingly powerful when composed together through multiple succesive hidden layers.  

'''

#       - Code for Activation Function

import numpy as np

input_data = np.array([-1,2])

weights = {'node_0' : np.array([3,3]),
            'node_1' : np.array([1,5]),
            'output' : np.array([2,-1])
            }

# Calculate node 0 value: node_0_value
node_0_value = (input_data * weights['node_0']).sum()

# we applied 'tanh' function to convert the input to output.
node_0_output = np.tanh(node_0_input)

# Calculate node 1 value: node_1_value
node_1_value = (input_data * weights['node_1']).sum()

node_1_output = np.tanh(node_1_input)

# Put node values into array: hidden_layer_outputs
hidden_layer_values = np.array([node_0_value, node_1_value])
print(hidden_layer_values)

# Calculate model output: output (do not apply relu)
output = (hidden_layer_values * weights['output']).sum()
print(output)

#-------------------------------------------------------------------------------------------------------------------------
''''
            -> Multiple Hidden Layers
                    Difference between modern deep learning and historical neural network is the use 
                    of model with not just one hidden layer , but many "succesive hidden layers". We forward propagate through 
                    these succesive layers in a similar way to what you saw for a single hidden layer.


                For example:
                    we have network with 2 hidden layer , we first fill in the values for the hidden layer one as a function 
                    of the inputs . Then apply the activation function to fill in the values in these nodes. Then use values from 
                    the first hidden layer to fill in the second hidden layer. Then we make the prediction based on the
                    the outputs of hidden layer two. You use the same forward propagation process but you apply that iterative process more time 
 


''''

# - you'll write code to do forward propagation for a neural network with 2 hidden layers. Each hidden layer has two nodes. 

def predict_with_network(input_data):
    # Calculate node 0 in the first hidden layer
    node_0_0_input = (input_data * weights['node_0_0']).sum()
    node_0_0_output = relu(node_0_0_input)

    # Calculate node 1 in the first hidden layer
    node_0_1_input = (input_data * weights['node_0_1']).sum()
    node_0_1_output = relu(node_0_1_input)

    # Put node values into array: hidden_0_outputs
    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])

    # Calculate node 0 in the second hidden layer
    node_1_0_input = (hidden_0_outputs * weights['node_1_0']).sum()
    node_1_0_output = relu(node_1_0_input)

    # Calculate node 1 in the second hidden layer
    node_1_1_input = (hidden_0_outputs * weights['node_1_1']).sum()
    node_1_1_output = relu(node_1_1_input)

    # Put node values into array: hidden_1_outputs
    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])
    
    # Calculate output here: model_output
    model_output = (hidden_1_outputs * weights['output']).sum()
    
    # Return model_output
    return(model_output)

output = predict_with_network(input_data)
print(output)