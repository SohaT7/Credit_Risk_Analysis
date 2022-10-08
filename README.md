# Credit_Risk_Analysis
## Table of Contents
- [Overview of the Analysis](#overview-of-the-analysis)
    - [Purpose](#purpose)
    - [About the Dataset](#about-the-dataset)
    - [Tools Used](#tools-used)
    - [Description](#description)
- [Results](#results)
    - [(1) Naive Random Oversampling with RandomOverSampler](#(1)-naive-random-oversampling-with-randomoversampler)
    - [(2) SMOTE Oversampling](#smote-oversampling)
    - [(3) Undersampling with ClusterCentroids](#(3)-Undersampling-with-ClusterCentroids)
    - [(4) Combination Sampling with SMOTEENN](#(4)-Combination-Sampling-with-SMOTEENN)
    - [(5) Balanced Random Forest Classifier](#(5)-Balanced-Random-Forest-Classifier)
    - [(6) Easy Ensemble AdaBoost Classifier](#(6)-Easy-Ensemble-AdaBoost-Classifier)
- [Summary](#summary)
- [Contact Information](#contact-information)

## Overview of the analysis
### Purpose:
The purpose of this analysis is to build and evaluate different machine learning models (logistic regression classifier models) to predict credit risk.

### About the Dataset:
The dataset used here is credit loan dataset from LendingClub, a FinTech company, which services peer-to-peer loans. The csv file contains about 115677 records.

### Tools Used:
 - Python (Pandas, NumPy, Scikit-learn, Imblanced-learn libraries)

### Description:
The dataset is divided into a training set (75%) and a testing set (25%); the training set will be used to train (i.e. 'fit') the logistic regression classifier model, while the testing set will be used to compare the predicted values with the target values in order to assess the predictive capability of the model.

Since credit risk is an unbalanced classification problem, i.e. the number of risky loans is easily always far less than the number of good loans, we start by resampling our dataset. Resampling techiques ensure that the risky loans are balanced (and not disproportionately represented) in both the training and testing sets. 

The different resampling techniques used here (code to be found in the file [credit_risk_resampling](https://github.com/SohaT7/Credit_Risk_Analysis/blob/main/credit_risk_resampling.ipynb)) include: 

 - Oversampling: RandomOverSampler and SMOTE algorithms
 - Undersampling: ClusterCentroids algorithm
 - A combination of oversampling and undersampling: SMOTEENN algorithm

Next, ensemble learning models were tried (code to be found in the file [credit_risk_ensemble](https://github.com/SohaT7/Credit_Risk_Analysis/blob/main/credit_risk_ensemble.ipynb)) in order to try and improve the model performance:
 - BalanceRandomForestClassifier
 - EasyEnsembleClassifier

## Results
After running each model, we generate the following for each:
 - Balanced accuracy score
 - Confusion matrix
 - Classification report

The confusion matrix and the classification report help us gauge how good a model is by providing us with a value for:
 - Precision
 - Sensitivity (aka Recall)
 - F1 Score

A note to the reader: In our results, 0 denotes a high risk or bad loan, and 1 denotes a low risk or good loan. In the formulae listed below, TP refers to True Positives, FP is False Positives, FN is False Negatives, TN is True Negatives, P is the Precision value, and R is the Recall value.

The formulae for Precision, Sensitivity (or Recall), F1 score, and Accuracy score are as follows:
 - Precision = TP/[TP+FP]
 - Recall (or Sensitivity) = TP/[TP+FN]
 - F1 score = [2(P * R)]/[P + R]
 - Accuracy score (generally caluculated as): Acc. Score = [TP + TN]/Total

### (1) Naive Random Oversampling with RandomOverSampler:
![Naive Random Oversampling](https://github.com/SohaT7/Credit_Risk_Analysis/blob/main/Images/Image_Naive_Random_Oversampling.png)

 - Balanced accuracy score = 0.64
 - Precision (to predict high risk or bad loans) = [ 67 / (67+6546) ] = 0.01
 - Precision (to predict low risk or good loans) = [ 10558 / (10558+34) ] = 1.00
 - Recall (to predict high risk or bad loans) = [ 67 / (67+34) ] = 0.66
 - Recall (to predict low risk or good loans) = [ 10558 / (10558+6546) ] = 0.62
 - F1 score (to predict high risk or bad loans) = [2(0.01 * 0.66)]/[0.01 + 0.66] = 0.02
 - F1 score (to predict low risk or good loans) = [2(1.00 * 0.62)]/[1.00 + 0.62] = 0.76

### (2) SMOTE Oversampling:
![SMOTE Oversampling](https://github.com/SohaT7/Credit_Risk_Analysis/blob/main/Images/Image_SMOTE_Oversampling.png)

- Balanced accuracy score = 0.65
- Precision (to predict high risk or bad loans) = [ 62 / (62+5317) ] = 0.01
- Precision (to predict low risk or good loans) = [ 11787 / (11787+39) ] = 1.00
- Recall (to predict high risk or bad loans) = [ 62 / (62+39) ] = 0.61
- Recall (to predict low risk or good loans) = [ 11787 / (11787+5317) ] = 0.69
- F1 score (to predict high risk or bad loans) = [2(0.01 * 0.61)]/[0.01 + 0.61] = 0.02
- F1 score (to predict low risk or good loans) = [2(1.00 * 0.69)]/[1.00 + 0.69] = 0.81

### (3) Undersampling with ClusterCentroids:
![Undersampling with ClusterCentroids](https://github.com/SohaT7/Credit_Risk_Analysis/blob/main/Images/Image_ClusterCentroids.png)

- Balanced accuracy score = 0.54
- Precision (to predict high risk or bad loans) = [ 70 / (70+10352) ] = 0.01
- Precision (to predict low risk or good loans) = [ 6752 / (6752+31) ] = 1.00
- Recall (to predict high risk or bad loans) = [ 70 / (70+31) ] = 0.69
- Recall (to predict low risk or good loans) = [ 6752 / (6752+10352) ] = 0.39
- F1 score (to predict high risk or bad loans) = [2(0.01 * 0.69)]/[0.01 + 0.69] = 0.01
- F1 score (to predict low risk or good loans) = [2(1.00 * 0.39)]/[1.00 + 0.39] = 0.57

### (4) Combination Sampling with SMOTEENN:
![Combination Sampling with SMOTEENN](https://github.com/SohaT7/Credit_Risk_Analysis/blob/main/Images/Image_SMOTEENN_Combo.png)

- Balanced accuracy score = 0.66
- Precision (to predict high risk or bad loans) = [ 76 / (76+7566) ] = 0.01
- Precision (to predict low risk or good loans) = [ 9538 / (9538+25) ] = 1.00
- Recall (to predict high risk or bad loans) = [ 76 / (76+25) ] = 0.75
- Recall (to predict low risk or good loans) = [ 9538 / (9538+7566) ] = 0.56
- F1 score (to predict high risk or bad loans) = [2(0.01 * 0.75)]/[0.01 + 0.75] = 0.02
- F1 score (to predict low risk or good loans) = [2(1.00 * 0.56)]/[1.00 + 0.56] = 0.72

### (5) Balanced Random Forest Classifier:
![Balanced Random Forest Classifier](https://github.com/SohaT7/Credit_Risk_Analysis/blob/main/Images/Image_Balanced_Random_Forest_Classifier.png)

- Balanced accuracy score = 0.79
- Precision (to predict high risk or bad loans) = [ 71 / (71+2153) ] = 0.03
- Precision (to predict low risk or good loans) = [ 14951 / (14951+30) ] = 1.00
- Recall (to predict high risk or bad loans) = [ 71 / (71+30) ] = 0.70
- Recall (to predict low risk or good loans) = [ 14951 / (14951+2153) ] = 0.87
- F1 score (to predict high risk or bad loans) = [2(0.03 * 0.70)]/[0.03 + 0.70] = 0.06
- F1 score (to predict low risk or good loans) = [2(1.00 * 0.87)]/[1.00 + 0.87] = 0.93

### (6) Easy Ensemble AdaBoost Classifier:
![Easy Ensemble AdaBoost Classifier](https://github.com/SohaT7/Credit_Risk_Analysis/blob/main/Images/Image_Easy_Ensemble_AdaBoost_Classifier.png)

- Balanced accuracy score = 0.93
- Precision (to predict high risk or bad loans) = [ 93 / (93+983) ] = 0.09
- Precision (to predict low risk or good loans) = [ 16121 / (16121+8) ] = 1.00
- Recall (to predict high risk or bad loans) = [ 93 / (93+8) ] = 0.92
- Recall (to predict low risk or good loans) = [ 16121 / (16121+983) ] = 0.94
- F1 score (to predict high risk or bad loans) = [2(0.09 * 0.92)]/[0.09 + 0.92] = 0.16
- F1 score (to predict low risk or good loans) = [2(1.00 * 0.94)]/[1.00 + 0.94] = 0.97

## Summary
In a high-risk credit loan detection problem, there will be an intricate trade-off between precision and recall. If we go for more precision, that means some of the high-risk credit loan cases may go undetected but whichever ones are captured will very likely be actually high-risk credit loan cases too. The rate of false positives will be low. On the other hand, if we go for more sensitivity in our model, we will end up capturing most, if not all, of the high-risk credit loan cases. However, we run the risk of also capturing a lot of false positives, i.e. cases which were flagged as high-risk credit loan but actually are not. In this case, those that will not defect on repaying the loan (false positives) will be unnecessarily denied loan.

From among our models, the two ensemble learning models perform better than the ones that involved only resampling. From among the Balanced Random Forest Classifier and Easy Ensemble AdaBoost Classifier, the latter performs better: it has a greater accuracy score (0.93 versus Balanced Random Forest Classifier's 0.73), better precision with detecting bad loans (0.09 versus 0.03), and better recall when it comes to determining a bad loan (0.92 versus 0.70). Its F1 score is also the highest from among all the other models, at 0.16. The higher the score, the better the model. Therefore, it is the Easy Ensemble AdaBoost Classifier model that should be used.

## Contact Information
Email: st.sohatariq@gmail.com
