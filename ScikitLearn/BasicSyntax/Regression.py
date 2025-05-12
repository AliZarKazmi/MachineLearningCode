'''
    KeyPoints:
        1. Creating Features and Target Arrays
        2. Basic Syntax (for 1 Feature & Multiple Features)

        3. Measuring the Performance of Regression Model (R-Squared, MSE, Cross-Validation)
            (3.a) R-Squared : 
                "the default matric for Linear Regression is R-Squared, which quantifies the amount of variance
                 in the target variable that is explained by the features .
                 Values can Range from 0-1, with 1 meaning the features completely explain the target variance "
            
            (3.b) Mean Squared Error (MSE): 
                    "MSE is measured in target units, squared. 
                     For example : is a model is predicting a dollar value , MSE will be in "dollars Squared". To convert
                                   the dollars , we can take the "square root" , known as the Root Mean Squared eror or RMSE
                                RMSE measure in the same units at the target variable"

        4. CrossValidation (k-fold Validation)
                            fold means group
                4.a) Why we need this? 
                                1. Model performance is dependent on the way we split the data
                                2. Not representative of the model's ability to generalize the unseen data
                                3. To combat this dependence on what is essentially a random split, we use a technique  called cross-validation
                                4. By using cross-validation, we can see how performance varies depending on how the data is split i.e R-squared for each fold ranged between 0.74 and 0.77!
                4.b) What it is?
                            *Cross-validation is a vital approach to evaluating a model. It maximizes the amount of data that is available to the model, as the
                            model is not only trained but also tested on all of the available data.
                            *In it we make groups on our dataset like split dataset into 5 groups then the first group we considered it as a test dataset
                            and remaining 4 as training dataset and fit our data on that 4 dataset. then we take 2 group as testdataset and remaing 4 training and so on.
                            Then we get the 5 metrics of R-Squared values then we perform some computation on that values and conclude the model performance
'''

#-----------------------------------------------------------------------------------------------------------

'''
                                      => Creating Features and Target Arrays
 Note: 
        *In Supervise Learning we use all the features in our dataset but we "drop our traget" and store the
        values attributes as 'X' & for 'Y' we take the target column's values attributes 

        *for scikit-learn our Features must be formated as "2-Dimensional " arrays , we achieve this 1-D -> 2-D conversion
        by the help of Numpy i.e ".reshape(-1,1)"
        

i.e 
'''
X =  diabeties_df.drop("glucose", axis=1).values
Y = diabeties_df["glucose"].values

# for scikit-learn our Features [features means only our 'X' not for 'Y'] must be formated as "2-Dimensional " arrays 
''' 
this convert our 1-dimensional array into 2-D array
make sure your Features are always in 2-D array 

'''
df = df.reshape(-1,1) 

# Check the shape of the features and targets
print(X.shape,y.shape)

#-----------------------------------------------------------------------------------------------------------

'''
                        => Fitting Regression Model for 1-Feature
    *the goal is to assess the relationship between the feature and target values there is no need to split the data into training and test sets.

        
'''

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X,Y)

'''
*the goal is to assess the relationship between the feature and target values there is no need to split the data into training and test sets.
'''
predictions = reg.predict(X)  #this will give us the line of best fit for our data & as a parameter we pass the Test data i.e 'X_TEST'


plt.scatter(X,Y)#Create a scatter plot visualizing y against X
plt.plot(X,predictions) #line plot displaying the predictions against X
plt.ylabel("Blood Glucose (mg/dl)")
plt.xlabel("Body Mass Index")
plt.show()

#-----------------------------------------------------------------------------------------------------------
'''
                    => Linear Regression Using Multiple or All Features 
'''

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=42)

reg_all = LinearRegression()
reg_all.fit(X_train,y_train)

y_predict= reg_all.predict(X_test)


#-----------------------------------------------------------------------------------------------------------
'''
                            => R-Squared in Scikit Learn
    Calculate the model's R-squared score by passing the test feature values and the test target values to an appropriate method.
'''
reg_all.score(X_test,y_test)

#-----------------------------------------------------------------------------------------------------------
'''
                            => RMSE in Scikit Learn
    Calculate the model's root mean squared error using y_test and y_pred.
'''
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred, squared=False) # it will return the Square Root of MSE


#-----------------------------------------------------------------------------------------------------------
'''
                            => Cross Validation in Scikit Learn 
    Calculate the model's root mean squared error using y_test and y_pred.
'''

from sklearn.model_selection import cross_val_score, KFold  # KFold  allows us to set a seed and shuffle our data, making our results repeatable downstream.

#by default "n_splits" is initialized with 5
#"shuffle" which shuffle our dataset before splitting into folds/groups
kf = KFold(n_splits=6, shuffle=True, random_state=42)  

reg = LinearRegression()

# pass 4 parameter i.e Model, Feature Data, Target Data, Specify the Numbers of Fold
# it returns the Array of Cross-Validation, the length of the Array is the number of Folds/Groups utilized.
#    Note: that the score reported is R-Squared, as this is the default score for Linear Regression
''' 
    Note: that the score reported is R-Squared, as this is the default score for Linear Regression
'''
cv_results = cross_val_score(reg, X, Y, cv=kf) 

''''
                    Analyzing cross-validation metrics
        Now we have performed cross-validation, it's time to analyze the results.

Will will display the mean, standard deviation, and 95% confidence interval 
'''
# Now we calculate the Mean & Standard Deviation
np.mean(cv_results)
np.std(cv_results)

# Addiationally we can calculate the 95% confidence interval
np.quantile(cv_results,[0.025, 0.975]) 

