'''''
        -> Logistic Regression: 
                        * it is used for Classification problems.
                        * This model calculates the probablity "p" that an observation belongs to binary class
                        * If probability, p > 0.5:
                                ^The data is labeled  "1".
                        * If probability, p < 0.5:
                                ^The data is labeled  "0".
                        * Logistic Regression produces a "Linear Decision  Boundary " 

'''''

#----------------------------------------------------------------------------------------------------------------------------------------
'''
            ->Logostic Regression in Scikit-Learn
'''

from sklearn.linear_model  import LogisticRegression

logreg = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)

#----------------------------------------------------------------------------------------------------------------------------------------
'''
            -> Predicting Probabilities
'''

# we can predict probabilities of each instance belonging to a class by calling logisitc regression's "predict_proba method" and passing the
# the test features.

# This return a 2-dimensional array with probablities for both classes.

# we sliced the second column , representing the positive class probabilities and store the results as y_pred_probs

y_pred_probs = logreg.predict_proba(X_test)[:,1] # predicitng probabilities

#Predict the probabilities of each individual in the test set having a diabetes diagnosis, storing the array of positive probabilities as y_pred_probs.

print(y_pred_probs[:10]) # Output result i.e:  the probability of a diabetes diagnosis for the first 10 individuals in the test set ranges from 0.01 to 0.79.

#----------------------------------------------------------------------------------------------------------------------------------------
'''
            -> Probabilities Threshold

            *The default probability threshold for logistic regression in Scikit-learn is 0.5 i.e 
                       By deafult, logistic regression threshold = 0.5


        * -> What happen if we vary the Threshold? 
            We can use the "Receiver Operating Characteristics or ROC" curve to visualize how different thresholds affect true positive 
            and false positive rates. The Dotted line in it represents a "Chance Model" which randomly guesses labels 

           1. When the threshold is 0 i.e (p=0), the model predicts "1" for all observations meaning it will correctly predict all positive values 
            and incorrectly predict all negative values. 
           2. When the threshold is 1 i.e (p=1), the model predicts "0" for all observations meaning that both the true and false postive rates are 0.
           3. if we vary the threshold we get a series of different false positive and true positive rates.
'''

#----------------------------------------------------------------------------------------------------------------------------------------
'''
            ->Plotting the ROC Curve
'''

from sklearn.metrics import roc_curve

# fpr = Flase positive rate
# tpr = true positive rate

fpr, tpr, thresholds = roc_curve(y_test,y_pred_probs)

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr)

plt.xlabel("False Positive Rate")
plt.ylabel("True positive Rate")
plt.title('Logistic Regression ROC Curve')

plt.show()

''''
    Q: How do we identify the models performance based on the this plot?
    If we have a model with " 1 " for true positive rate and "0" for false positive rate, this would be the perfect model.
    Therefor we calculate the area under the ROC curve metrics knwon as "AUC" which "scores range from 0 to 1", with one being ideal. 
'''

#----------------------------------------------------------------------------------------------------------------------------------------
'''
            ->Calculating AUC in Scikit-learn
'''

from sklearn.metrics import roc_auc_score

print(roc_auc_score(y_test, y_pred_probs))
