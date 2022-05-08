# Credit_Risk_Analysis

## Overview of the analysis
### Purpose of the analysis:
The purpose of this analysis was to determine which model helps us best with resolving our credit card risk problem. 
### About the dataset:
The dataset we used is credit card dataset from LendingClub, a company which services the provision of loans among peers.
### Description of the analysis:
Since credit risk is an unbalanced classification problem by its very nature, i.e. the number of risky loans is easily always far less than the number of good loans, we used different resampling techniques on our dataset. Resampling makes sure that an equal percentage of the risky loans (and therefore good loans as well) are represented in both our training dataset and testing dataset. 

The general process is like so:
A dataset is divided into a training set (75%) and a testing set (25%). The training set is used for the "fitting" or training process, which results in the creation of a model. The testing set (25%) is then used to test out that model which has just been created. Without resampling, however, we run the risk of having low risk and high risk loans being misrepresented in our training and testing sets. We need to ensure that a similar proportion of good and bad loans (i.e. both classes of our target variable) are represented in the training as well as the testing data.
To make clear, as an exmaple: 75% of all good loans in our dataset and 25% of all bad loans in our dataset are present in the training set, and likewise, 75% of all good loans in our dataset and 25% of all bad loans in our dataset are present in the testing set as well.

The file [credit_risk_resampling](https://github.com/SohaT7/Credit_Risk_Analysis/blob/main/credit_risk_resampling.ipynb) contains resampling exercises followed by running each model. The RandomOverSampler and SMOTE algorithms were used to oversample our data. The ClusterCentroids algorithm was used to undersample the data. The SMOTEENN algorithm combines the oversampling and undersampling technique. 
Next, ensemble learning models - the BalanceRandomForestClassifier and EasyEnsembleClassifier - were used to help reduce bias. Ensemble learning combines multiple models in order to improve the overall robustness and accuracy of the model. The file [credit_risk_ensemble](https://github.com/SohaT7/Credit_Risk_Analysis/blob/main/credit_risk_ensemble.ipynb) consists of code that runs those models on the dataset. 

## Results
After running each model, the balanced accuracy score, a confusion matrix, and a classification report was generated. The latter two provide us with values for precision, sensitivity (also known as "recall"), as well as the F1 score, all of which tell us how good a model is. In our results, 0 denotes a high risk or bad loan, and 1 denotes a low risk or good loan.

The calculations have been done like so:

Precision = TP/[TP+FP] , where TP is True Positives and FP is False Positives.

Recall (or Sensitivity) = TP/[TP+FN] , where TP is True Positives, FP is False Positives, and FN is False Negatives.

F1 score = [2(P * R)]/[P + R] , where P denotes precision of the model and R denotes the recall or sensitivity of the model. 

Accuracy score is generally caluculated as such:
Acc. Score = [TP + TN]/Total , where TP is True Positives and TN is True Negatives.

For each model, the calculations have been shown so the reader can see which numbers from the confusion matrix have been used, and how or where exactly in these calculations.

### (1) Naive Random Oversampling with RandomOverSampler:
![Naive Random Oversampling](https://github.com/SohaT7/Credit_Risk_Analysis/blob/main/Image_Naive_Random_Oversampling.png)
The balanced accuracy score for this model is 0.64.

The precision of this model to predict high risk or bad loans is [ 67 / (67+6546) ] = 0.01, whereas the precision to predict low risk or good loans is [ 10558 / (10558+34) ] = 1.00.

The recall or sensitivity of the model to predict high risk or bad loans is [ 67 / (67+34) ] = 0.66, whereas that for low risk or good loans is [ 10558 / (10558+6546) ] = 0.62.

The F1 score of this model to predict high risk or bad loans is [2(0.01 * 0.66)]/[0.01 + 0.66] = 0.02, whereas that to predict low risk or good loans is [2(1.00 * 0.62)]/[1.00 + 0.62] = 0.76.

### (2) SMOTE Oversampling:
![SMOTE Oversampling](https://github.com/SohaT7/Credit_Risk_Analysis/blob/main/Image_SMOTE_Oversampling.png)
The balanced accuracy score for this model is 0.65.

The precision of this model to predict high risk or bad loans is [ 62 / (62+5317) ] = 0.01, whereas the precision to predict low risk or good loans is [ 11787 / (11787+39) ] = 1.00.

The recall or sensitivity of the model to predict high risk or bad loans is [ 62 / (62+39) ] = 0.61, whereas that for low risk or good loans is [ 11787 / (11787+5317) ] = 0.69.

The F1 score of this model to predict high risk or bad loans is [2(0.01 * 0.61)]/[0.01 + 0.61] = 0.02, whereas that to predict low risk or good loans is [2(1.00 * 0.69)]/[1.00 + 0.69] = 0.81.

### (3) Undersampling with ClusterCentroids:
![Undersampling with ClusterCentroids](https://github.com/SohaT7/Credit_Risk_Analysis/blob/main/Image_ClusterCentroids.png)
The balanced accuracy score for this model is 0.54.

The precision of this model to predict high risk or bad loans is [ 70 / (70+10352) ] = 0.01, whereas the precision to predict low risk or good loans is [ 6752 / (6752+31) ] = 1.00.

The recall or sensitivity of the model to predict high risk or bad loans is [ 70 / (70+31) ] = 0.69, whereas that for low risk or good loans is [ 6752 / (6752+10352) ] = 0.39.

The F1 score of this model to predict high risk or bad loans is [2(0.01 * 0.69)]/[0.01 + 0.69] = 0.01, whereas that to predict low risk or good loans is [2(1.00 * 0.39)]/[1.00 + 0.39] = 0.57.

### (4) Combination Sampling with SMOTEENN:
![Combination Sampling with SMOTEENN](https://github.com/SohaT7/Credit_Risk_Analysis/blob/main/Image_SMOTEENN_Combo.png)
The balanced accuracy score for this model is 0.66.

The precision of this model to predict high risk or bad loans is [ 76 / (76+7566) ] = 0.01, whereas the precision to predict low risk or good loans is [ 9538 / (9538+25) ] = 1.00.

The recall or sensitivity of the model to predict high risk or bad loans is [ 76 / (76+25) ] = 0.75, whereas that for low risk or good loans is [ 9538 / (9538+7566) ] = 0.56.

The F1 score of this model to predict high risk or bad loans is [2(0.01 * 0.75)]/[0.01 + 0.75] = 0.02, whereas that to predict low risk or good loans is [2(1.00 * 0.56)]/[1.00 + 0.56] = 0.72.

### (5) Balanced Random Forest Classifier:
![Balanced Random Forest Classifier](https://github.com/SohaT7/Credit_Risk_Analysis/blob/main/Image_Balanced_Random_Forest_Classifier.png)
The balanced accuracy score for this model is 0.79.

The precision of this model to predict high risk or bad loans is [ 71 / (71+2153) ] = 0.03, whereas the precision to predict low risk or good loans is [ 14951 / (14951+30) ] = 1.00.

The recall or sensitivity of the model to predict high risk or bad loans is [ 71 / (71+30) ] = 0.70, whereas that for low risk or good loans is [ 14951 / (14951+2153) ] = 0.87.

The F1 score of this model to predict high risk or bad loans is [2(0.03 * 0.70)]/[0.03 + 0.70] = 0.06, whereas that to predict low risk or good loans is [2(1.00 * 0.87)]/[1.00 + 0.87] = 0.93.

### (6) Easy Ensemble AdaBoost Classifier:
![Easy Ensemble AdaBoost Classifier](https://github.com/SohaT7/Credit_Risk_Analysis/blob/main/Image_Easy_Ensemble_AdaBoost_Classifier.png)
The balanced accuracy score for this model is 0.93.

The precision of this model to predict high risk or bad loans is [ 93 / (93+983) ] = 0.09, whereas the precision to predict low risk or good loans is [ 16121 / (16121+8) ] = 1.00.

The recall or sensitivity of the model to predict high risk or bad loans is [ 93 / (93+8) ] = 0.92, whereas that for low risk or good loans is [ 16121 / (16121+983) ] = 0.94.

The F1 score of this model to predict high risk or bad loans is [2(0.09 * 0.92)]/[0.09 + 0.92] = 0.16, whereas that to predict low risk or good loans is [2(1.00 * 0.94)]/[1.00 + 0.94] = 0.97.

## Summary
To summarize the results, the scores from each model are as follows:

######(1) Naive Random Oversampling with RandomOverSampler:
Balanced accuracy score: 0.64; Precision: (high risk: 0.01; low risk: 1.00) ; Recall: (high risk: 0.66; low risk: 0.62); F1 score: (high risk: 0.02; low risk: 0.76).

######(2) SMOTE Oversampling:
Balanced accuracy score: 0.65; Precision: (high risk: 0.01; low risk: 1.00) ; Recall: (high risk: 0.61; low risk: 0.69); F1 score: (high risk: 0.02; low risk: 0.81).

######(3) Undersampling with ClusterCentroids:
Balanced accuracy score: 0.54; Precision: (high risk: 0.01; low risk: 1.00) ; Recall: (high risk: 0.69; low risk: 0.39); F1 score: (high risk: 0.01; low risk: 0.57).

######(4) Combination Sampling with SMOTEENN:
Balanced accuracy score: 0.66; Precision: (high risk: 0.01; low risk: 1.00) ; Recall: (high risk: 0.75; low risk: 0.56); F1 score: (high risk: 0.02; low risk: 0.72).

######(5) Balanced Random Forest Classifier:
Balanced accuracy score: 0.79; Precision: (high risk: 0.03; low risk: 1.00) ; Recall: (high risk: 0.70; low risk: 0.87); F1 score: (high risk: 0.06; low risk: 0.93).

######(6) Easy Ensemble AdaBoost Classifier:
Balanced accuracy score: 0.93; Precision: (high risk: 0.09; low risk: 1.00) ; Recall: (high risk: 0.92; low risk: 0.94); F1 score: (high risk: 0.16; low risk: 0.97).

In a credit card fraud detection problem, there will be an intricate trade-off between precision and recall. If we go for more precision, that means some of the credit card fraud cases may go undetected but whichever ones are captured will very likely be actually credit card fraud cases too. The rate of false positives will be low. On the other hand, if we go for more sensitivity in our model, we will end up capturing most if not all of the credit card fraud cases. However, we run the risk of also capturing a lot of false positives, i.e. cases which were flagged as credit card fraud but actually are not. In this case, the clients' accounts (false positives) will be unnecessarily frozen and cause them inconvenience.

From among our models, the two ensemble learning models perform better than the ones that involved only resampling. From among the Balanced Random Forest Classifier and Easy Ensemble AdaBoost Classifier, the latter performs better: it has a greater accuracy score (0.93 versus 0.73 from the Balanced Random Forest Classifier), better precision with detecting bad loans (0.09 versus 0.03), and better recall when it comes to determining a bad loan (0.92 versus 0.70). Its F1 score is also the highest from among all the other models, at 0.16. The higher the score, the better the model. Therefore, it is the Easy Ensemble AdaBoost Classifier model that should be used.
