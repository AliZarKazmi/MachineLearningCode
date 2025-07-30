'''
    In this file we will leran about the basic Structure code for Supervised Learning using Scikit-learn

    Summary:
    1. Basic Syntax
    2. Syntax Example
    3. Measuring Model Performance Technqies for Classificarion i.e ('Accuracy' & 'Model Complexity' ) 
'''


                                # General Scikit-Learn workflow Syntax:
# -------------------------------------------------------------------------------------------------------------------------------------

    # KeyPoints:
        #     1.Import Model 
        #     2.Initialize Model
        #     3.Train/Fit data into the model
        #     4.Predict the target with the trained model on new Datapoint


# Step 1: Import the Model, which  type of algorithm for our supervised learning problem from "sklearn module"
from sklearn.module import Module

model = Module #create a variable named 'model' and instantiate 'Model'

# Step 2: Import the dataset which is the data we will use to train our model from
model.fit(X,y) # it learn patterns about the features and the target variables.

# Step 3: Predict the target with the new datapoints
# X_new is the new data we want to predict the target variable for.
# predictions is the predicted target variable for the new data.
predictions = model.predict(X_new)

print(predictions)
# -------------------------------------------------------------------------------------------------------------------------------------

                                    # Example- KNeighborsClassifier Model
# -------------------------------------------------------------------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
X = churn_df[["total_day_charge", "total_eve_charge"]].values # ".values" it is used to convert the features into "Numpy array"
y = churn_df["churn"].values

knn = KNeighborsClassifier(n_neighbors=15)

knn.fit(X,y)

X_new = np.array([
    [56.8,17.5],
    [24.4,24.1],
    [50.1,10.9]
                  ])

predictions = knn.predict(X_new)
print('Predictions: {}'.format(predictions))
# -------------------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------------------------------

                        # Measuring Model Performance for Classificarion - Accuracy
# -------------------------------------------------------------------
'''

In classification, "Accuracy" is a commonly used metric for model performace 
    
        * Accuracy = Correct Predictions / Total Observation
    * We calculate the models accuracy against the Test datasets labels.

'''

'''   Train/Test Split '''

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.3, random_state=21,stratify=y)

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,Y_train)

print(knn.score(X_test,Y_test)) # to check the accuracy , we use the ".score()" method, by passing X & Y test 


#---------------------------------------------------------------------------------------------------------------------
'''                              "Model Complexity" & Over/Underfitting i.e KNN
      *Interpreting model complexity is a great way to evaluate supervised learning performance. Your aim is to produce a model that can interpret the relationship between features and the target variable, as well as generalize well when exposed to new observations.
      * finding the best value of "K"

      *you have calculated the accuracy of the KNN model on the training and test sets using various values of n_neighbors, you can create a model complexity curve to visualize how performance changes as the model becomes less complex!
'''
train_accuracies ={}
test_accuracies ={}
neighbors = np.arrange(1,26)
for neighbor in neighbors:
    knn= KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train,Y_train)
    train_accuracies[neighbor] = knn.score(X_train, Y_train)
    test_accuracies[neighbor] = knn.score(X_train, Y_train)

#Visualizing model complexity
plt.figure(figsize=(8,6))
plt.title("KNN: Varying number of Neighbors")
plt.plot(neighbors,train_accuracies.values(),label="Training Accuracy ")
plt.plot(neighbors,test_accuracies.values(),label="Testing Accuracy ")    
plt.legend()
plt.xlabel("Number of Neighbor")
plt.ylabel("Accuracy")
plt.show()