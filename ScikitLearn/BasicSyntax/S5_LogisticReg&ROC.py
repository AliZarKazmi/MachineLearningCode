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

# Generate ROC curve values: fpr, tpr, thresholds
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

# Calculate roc_auc_score
print(roc_auc_score(y_test, y_pred_probs))

#-------------------------------------------------------------------------------------------------------------------------------------------
'''
 Key points included:

Accuracy's limitations: You discovered that accuracy might not always be the best metric, especially in cases of class imbalance, such as in fraud detection scenarios where the majority of transactions are legitimate.

Confusion Matrix: You learned how to create and interpret a confusion matrix, a 2-by-2 matrix that helps visualize the performance of a binary classifier. The matrix includes true positives, true negatives, false positives, and false negatives.

Precision and Recall: Precision (the number of true positives divided by all positive predictions) and recall (the number of true positives divided by the sum of true positives and false negatives) were introduced as crucial metrics. High precision means a lower false positive rate, while high recall indicates a lower false negative rate.

F1-Score: You learned about the F1-score, the harmonic mean of precision and recall, which is particularly useful when you need a balance between precision and recall.

Practical Application: Using scikit-learn's classification_report and confusion_matrix, you practiced evaluating a model trained on a diabetes dataset. The exercise involved fitting a KNeighborsClassifier model, making predictions, and then generating a confusion matrix and classification report.
'''

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

'''
                        -> Confusion Matrix & Classification Report
'''

# Import confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report

# Fit the model, predict, and evaluate
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Calculate the confusion matrix
print(confusion_matrix(y_test, y_pred))

# Calculate the classification report
print(classification_report(y_test, y_pred))

'''
Choosing Metrics: You explored how to decide on the most appropriate metric (precision, recall, F1-score) based on the problem context, emphasizing the importance of understanding the business or application goal to choose the right evaluation metric.

**The ROC curve is above the dotted line, so the model performs better than randomly guessing the class of each observation.
'''

