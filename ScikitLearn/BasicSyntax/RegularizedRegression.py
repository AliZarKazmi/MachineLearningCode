'''
    KeyPoints:
        1. Regularized Regression 
            (1.a) Defination:
                            A technique to avoid "Overfitting"
            (1.b) Why Regularization:
                            *Linear Reegresion mnimizes a Loss function, it chooses a coefficients , "alpha", for each feature variable plus "beta"
                            Large coefficients  can lead to overfitting so Regularization penalize large coefficients
            
        
        2. Types of Regularization (Ridge & Lasso)
            (2.a) Ridge Regression: 
                    Ridge regression performs regularization by computing the squared values of the model parameters multiplied by alpha and adding them to the loss function.      
                        * Ridge penalize the Large positive and Negative Coefficients. 
                        * when minimizing the loss function, models are penalized for coefficients with Large positive and Negative values.
                        * alpha: when using Ridge we need to choose the aplha value in order to fit and predict.
                        * Picking alpha for Ridge is similar to picking "K" in KNN
                        * Aplha in Ridge is known as a Hyperparameter which is a variable used for selecting a model's parameters.
                        * Alpha controls model complexity. 
                            - Alpha = 0[Can lead to Overfitting ] ; ( when alpha is equal to 0 , we are performing OLS [Ordinary Least Function] where large coefficients are not penalized and overfitting may occur) 
                            -  Very high Alpha = Can lead to Underfitting ; A high Alpha  means that large coefficient are significantly penalized which can lead to underfitting, 
                            - Model Performance get worse when Alpha Increase 
            (2.b) Lasso Regression:
        
        3. Lasso Regression for Feature Selection:
                        * Lasso can be used to used to assess Feature Importance, it can select important features of a dataset
                        * This is because it tends to shrink the coefficients of less important features to 0.
                        * Shrink the coefficients of less important features to 0 
                        * Features whose coefficients are not shrunck to 0 are selected by Lasso algorithm



'''

#-----------------------------------------------------------------------------------------------------------

'''
                                -> Ridge Regression in Scikit-learn
'''

from sklearn.linear_model import Ridge

# To highlight the impact of different alpha values, we create an empty list for our scores then loop through a list of different alpha values 
score = []
for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
    
    # Create a Ridge regression model
    ridge = Ridge(alpha = alpha )
    
    # Fit the data
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)

    # Obtain R-squared
    score.append(ridge.score(X_test, y_test)) # we score the model's R-squared value to the score list

print(score) # we print the scores for the models with five different alpha values 

#-----------------------------------------------------------------------------------------------------------

'''
                                -> Lasso Regression in Scikit-learn
 its implementaion in Scikit-learn is similar to the Ridge
'''

from sklearn.linear_model import Lasso

# To highlight the impact of different alpha values, we create an empty list for our scores then loop through a list of different alpha values 
score = []
for alpha in [0.1, 1.0, 10.0, 20.0, 50.0]:
    lasso = Lasso(alpha = alpha )
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    scores.append(lasso.scores(X_test, y_test)) # we score the model's R-squared value to the score list
print(score) # we print the scores for the models with five different alpha values 

#-----------------------------------------------------------------------------------------------------------

'''
                                -> Lasso Regression for Feature Selection in Scikit-learn
                
'''

from sklearn.linear_model import Lasso

X = diabetes_df.drop("glucose", axis =1 ).values
y = diabetes_df["glucose"].values

name = diabetes_df.drop("glucose", axis =1 ).columns  # as we calculating feature importance we use the entire dataset rather than splitting it

lasso = Lasso(alpha = 0.1) # then we instantiate Lasso setting alpha to 0.1

# we fit the model to the data  and extract/compute the Coefficients using the ".coef_" attribute
lasso_coef = lasso.fit(X,y).coef_

# Plot the coefficients for each feature - this is a Sanity Check which allow us communicate results to non-technical audiences
plt.bar(name , lasso_coef)
plt.xticks( rotation = 45)
plt.show()