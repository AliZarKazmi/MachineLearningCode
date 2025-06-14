''''
    We will understand how to choose like "Model Architecture" and "Model Optimization" arguments


    Q: Why optimization is Hard problem:
    Ans: 
        * The optimal value for any one weight depends on the values of the other weights and we are optimizing many weights
    at once. 
        * Even if the slope tells us which weights to increase and which weights to decrease our updates may not improve our
    model meaningfully.
        * Small learning rate might cause us to make such small updates to the model's weights that our model doesn't improve materially.
        * Large learning rate might take us too far in the direaction that seemed good.
        * Smart Optimizer like "Adam" helps but optimization problems still occur

        Note:
        **The easiest way to see the effect of different learning rates is to use the simplest Optimizer , "Stochastic Gradient Descent" also known as SGD. 


    Stochastic Gradient Descent: 
    This optimizer uses a fixed learning rate , learning rate around '0.01' are common. But you can specify the 
    learning rate you need with "lr" argument .
'''
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#           -> Code for     Stochastic Gradient Desccent
# Import the SGD optimizer
from tensorflow.keras.optimizers import SGD

def get_new_model(input_shape = input_shape):
    model = Sequential()
    model.add(Dense(100, activation = 'relu' , input_shape = input_shape))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(2, activation = 'softmax'))
    return(model)

# Create list of learning rates: lr_to_test
lr_to_test =[.000001, 0.01,1]

# loop over learning rates
for lr in lr_to_test:

    print('\n\nTesting model with learning rate: %f\n'%lr )

     # Build new model to test, unaffected by previous models
    model = get_new_model()
    
    # Create SGD optimizer with specified learning rate: my_optimizer
    my_optimizer = SGD(lr = lr)

# Compile the model    
    model.compile(optimizer = my_optimizer, loss = 'categorical_crossentropy')

    # Fit the model
    model.fit(predictors, target)


'''
    **Even If the learing rate is well-tuned you can run into a called "Dying -neuron " problem

    Dying - neuron:
    This problem occurs when a neuron takes a value less than 0 for all the rows of your data


    **"Vanishing Gradient": occurs when many layers have very small slopes (due to being on flat part of tanh curve)
    **In deep learning network, updates to backprop were close to 0. this is all called a Vanishing Gradient problem.

    Chaning the activation value is maay be the solution

'''

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

'''
        -> Model Validation
        We specify the split using the keyword argument "validation_split" when calling the fit method.
'''
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuarcy'] )
model.fit (predictors, target, validation_split = 0.3)


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

''''
******our goal is to have the best validation score possible , so we should keep training while validation
score is improving and then stop training when the validation score isn't improving. We do this something called
"Early Stopping" 


Now that you know how to monitor your model performance throughout optimization, you can use early stopping to stop optimization when it isn't helping any more
'''
#   ->Code for Early Stopping

from tensorflow.keras.callbacks import EarlyStopping

# patience arg means how many epochs the model can go without improving before we stop training
early_stopping_monitor = EarlyStopping(patience = 2)

model.fit(predictors, target , validation_split = 0.3, epochs = 20, callbacks = [early_stopping_monitor])

''''
    We can experient with different architectures
    *Experiments with different architectures
    *More Layers
    *Fewer Layers
    *Layers of More Nodes
    *Layers of fewer Nodes
    *Creating a great Model requires some experientation

'''