'''
    Objective : Learn how to Optimize our Model

    Remeber for Rdge/Lasso regression we choose value for "alpha" and for KNN choose 'n_neighbors'.


    Hyper Parameter: Parameters that we specify before fitting a model like alpha and n_neighbors are called Hyperparameter

    *Fundamental step for building a successfull model: is choosing the correct Hyperparameters


    -> Hyperparameter Tuning:  
        Choosing the Correct Hyperparameter:
            1. Try lots of different hyperparameters values
            2. Fit all of them separately
            3. See how well they perform
            4. Choose the best performing values

        when fitting different hyperparameter values , we use 'Cross-validation ' to avoid overfititing the hyperparameters to the test set.

        **it is essential to use cross-validation to avoid overfitting to the test set.
        ** we can split the data and perform cross-validation on the training set.


    -> Aproaches for Hyperparameter Tuning:
        1. Grid Search Cross-Validation : where we choose a grid of possible hyperparameter values to try.
        2. *Random Search Cross Validation* (Best Approach) :
            which pick random hyperparamter values rather than exhaustively searching through all options.

    -> Limitation of GridSearch and Alternative Approach
        *The number of fits is equal to the number of Hyperparameters multiplied by the number of values multiplied
        by the number of folds.Therefore it does't scale well.
            for example:
                performing 3 fold cross-validation for 1 hyperparameter with 10 total values means 30 fits.
                or performing 10 fold cross-validation for 3 hyperparameter with 30 total values means 900 fits.

            Alternative is "Random Search Cross-Validation"
'''

#-------------------------------------------------------------------------------------------------------------------------------------------------------

''''
        -> Grid Search CV in Scikit Learn (for Regression)
'''

from sklearn.model_selection import GridSearchCV

kf = KFold(n_split = 5, shuffle= True, random_state = 42)

# we then specify the names and values of the Hyperparameters we wish to tune as the keys and values of Dictionary, param_grid
param_grid = {"alpha" : np.arrange(0.0001, 1 , 10),
              "solver" : ["sag", "lsqr"]}

#instantiate our model
ridge = Ridge()

# we then call GridSearchCV and pass it our model, the grid we wish to tune over and set cv eaqul to kf
ridge_cv = GridSearchCV(ridge, param_grid, cv = kf)

# this return a GridSearch Object that we can then fit to the training data and this fit performs the actual cross-validated grid search
ridge_cv.fit(X_train, y_train)

#print the model's attributes best-params and best_score
print(ridge_cv.best_params_,ridge_cv.best_score_)

#-------------------------------------------------------------------------------------------------------------------------------------------------------

''''
        -> Random Search CV in Scikit Learn (for Regression)
'''

from sklearn.model_selection import RandomizedSearchCV

kf = KFold(n_split = 5, shuffle= True, random_state = 42)

# we then specify the names and values of the Hyperparameters we wish to tune as the keys and values of Dictionary, param_grid
param_grid = {"alpha" : np.arrange(0.0001, 1 , 10),
              "solver" : ["sag", "lsqr"]}

#instantiate our model
ridge = Ridge()

# we then call RandomSearchCV and pass it our model, the grid we wish to tune over and set cv eaqul to kf
#we can optionaly set the n_iter argument, which determines the number of hyperparameter values tested.
ridge_cv = RandomSearchCV(ridge, param_grid, cv = kf,  n_iter =2)

# this return a GridSearch Object that we can then fit to the training data and this fit performs the actual cross-validated grid search
ridge_cv.fit(X_train, y_train)

#print the model's attributes best-params and best_score
print(ridge_cv.best_params_,ridge_cv.best_score_)

#-------------------------------------------------------------------------------------------------------------------------------------------------------

''''
        -> Evaluating on Test Set
'''

test_score = ridge_cv.score(X_test, y_test)
print(test_score)
