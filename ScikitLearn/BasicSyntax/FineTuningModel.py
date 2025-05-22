''''
Having trained models, now you will learn how to evaluate them. In this chapter, 
you will be introduced to several metrics along with a visualization technique for 
analyzing classification model performance using scikit-learn. You will also learn how 
to optimize classification and regression models through the use of hyperparameter tuning.
'''


''''
    Measuring Model performance and for Classification "Accuracy" is not always the good one for the evaluation 

    * What is Class Imbalance : The situation where 1 class is more frequent than the other is called class-imbalance.

    * We can use the "Confusion Matirx" for classification model evaluation. Confusion matirx for accessing classification performance 


    Q: Why we need Confusion matirx or why it is important ?
    1. Firstly, we can retirve Accuracy : it is the sum of true predction divided by the total sum of the matrix.
    2. Secondly there are other important matrix we can caluclate from the Confusion matrix i.e ("Precision" & "Recall" & "F1-Score")

        *Precision: is the true number of Positives divided by the sum of all positive predictions. It is also called the "positve predictive value "
                        true positives / (true positive + false positives )
                    ** High precision = lower false positive rate
        
        *Recal : is the number of true positives divided by the sum of true positives and false negatives. It is also called "Sensitivity"
                ** High Recall = lower false negative rate

        * F1-Score : is the harmonic mean of precision and recall.
                    F1 Score = 2 * (precision * recall / precision + recall)
            This metric gives equal weight to precision and recall, therefore factors in both the number of errors made by the model and the type of errors.
            It is a usefull metric if we are seeking a model which performs reasonably well across both metric  

As you have seen, several metrics can be useful to evaluate the performance of classification models, including accuracy, precision, recall, and F1-score.
'''

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
''''
                                    -> Confusion Matrix in Scikit-learn
'''

from sklearn.metrics import classification_report, confusion_matrix

knn = kNeighborsClassifier (n_neighbors=7)

X_Train, X_test, y_train, y_test = train_test_split(X,y, test_sized = 0.4, random_state=42)

knn.fit(X_Train , y_train)

y_pred = knn.predict(X_test)

#CONFUSION MATRIX
print(confusion_matrix(y_test, y_pred))

# CLASSIFICATION REPORT
'Classification report -> it generate report outputs of all the relevant metrics i.e Precision, Recall, F1-score, Support.'
'Support : represent the number of intances for each class within the true labels'
''
print(classification_report(y_test,y_pred))


''''
Summary :
You learned about evaluating machine learning models, focusing on classification problems. Key points included:

Accuracy's limitations: You discovered that accuracy might not always be the best metric, especially in cases of class imbalance, such as in fraud detection scenarios where the majority of transactions are legitimate.

Confusion Matrix: You learned how to create and interpret a confusion matrix, a 2-by-2 matrix that helps visualize the performance of a binary classifier. The matrix includes true positives, true negatives, false positives, and false negatives.

Precision and Recall: Precision (the number of true positives divided by all positive predictions) and recall (the number of true positives divided by the sum of true positives and false negatives) were introduced as crucial metrics. High precision means a lower false positive rate, while high recall indicates a lower false negative rate.

F1-Score: You learned about the F1-score, the harmonic mean of precision and recall, which is particularly useful when you need a balance between precision and recall.

Practical Application: Using scikit-learn's classification_report and confusion_matrix, you practiced evaluating a model trained on a diabetes dataset. The exercise involved fitting a KNeighborsClassifier model, making predictions, and then generating a confusion matrix and classification report.

Choosing Metrics: You explored how to decide on the most appropriate metric (precision, recall, F1-score) based on the problem context, emphasizing the importance of understanding the business or application goal to choose the right evaluation metric. 
''' 