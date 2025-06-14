'''
        ->Model Capacity 
        it should be one of the key consideration to think about when deciding what models to 'try'

        'model capacity' & 'network capacity ' is closely related to the terms overfitting and underfitting.


        Underfitting : it is when model fails to find important predictives patterns in the training data. So it 
        is accurate in neither the training data nor validation data

        **Model capacity is a model's ability to capture predictive patterns in your data. 
        *** If you had a network and you increased the number of nodes or neurons in hidden layers that would increase model capacity and 
        if you add layers that will also increase capacity.

*Increasing the number of units in each hidden layer would be a good next step to try achieving even better performance.
        '''

#------------------------------------------------------------------------------------------------------------

''''
        Workflow for Optimizing Model Capacity
            1. Start with a simple network and get the validation score. (Start with the small network)
            2. Then keep adding capacity as long as the socre keeps improving (gradually increase capacity)
            3. Once it stops improving you can decrease capacity slightly but you are probably near the ideal
               (keep increasing capacity untill validation score is no longer improving)
'''