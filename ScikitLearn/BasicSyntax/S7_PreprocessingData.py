'''
    Learn how to impute missing values, convert categorical data to numeric values, scale data, evaluate multiple supervised learning models simultaneously, and build pipelines to streamline your workflow!

        ** Scikit learn requires "Numeric Data with No Missing values".
        
        *We do preprocessing before we build our model

    -> Converting Categorical Data into Numerical Features
        1. scikit-learn : OneHotEncoder()
        2. pandas : get_dummies()


        ****
            The model will be evaluated by calculating the average RMSE, but first, you will need to convert the scores for each fold to positive values and take their square root. 
            This metric shows the average error of our model's predictions, so it can be compared against the standard deviation of the target value
        ****
'''

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

 '''
                -> Encoding Dummy Variables
 '''
import pandas as pd

music_df = pd.read_csv('music.csv')

# pd.get_dummies = it will convert categorical data into numeric
music_dummies = pd.get_dummies(music_df["genre"], drop_first=True)
print(music_df.head())

# to convert the binary features back into original DataFrame we can use "pd.concat()"
# in it passing the list i.e containing the music data frame and our dummies Data frame
music_dummies = pd.concat([music_df, music_dummies], axis=1)

# lastly we can remove the original genre column using ".drop()" passing the column
music_dummies = music_dummies.drop("genre", axis = 1)

'''
    If the dataframe only has 1 categorical fetaure we can pass the entire dataframe, thus skipping the step of 
    combining variables. 
    If we don't specify the col, the new Dataframe's binary columns will have the original feature name prefixed and the
    origial Categroical col will automatically be drop
'''

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
                ->Regression with categorical features
'''
# Create X and y
X = music_dummies.drop("popularity", axis=1).values
y = music_dummies["popularity"].values

# Instantiate a ridge model
ridge = Ridge(alpha=0.2)

# Perform cross-validation
scores = cross_val_score(ridge, X, y, cv=kf, scoring="neg_mean_squared_error")

# Calculate RMSE
rmse = np.sqrt(-scores)
print("Average RMSE: {}".format(np.mean(rmse)))
print("Standard Deviation of the target array: {}".format(np.std(y)))

''''
    ***An average RMSE of approximately 8.24 is lower than the standard deviation of 
    the target variable (song popularity), suggesting the model is reasonably accurate.
'''
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
