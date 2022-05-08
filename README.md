# Credit_Risk_Analysis

## Overview of the analysis
### Purpose of the analysis:
The purpose of this analysis was to determine which model helps us best with resolving our credit card risk problem. 
### About the dataset:
The dataset we used is credit card dataset from LendingClub, a company which services the provision of loans among peers.
### Description of the analysis:
Since credit risk is an unbalanced classification problem by its very nature, i.e. the number of risky loans is easily always far less than the number of good loans, we used different resampling techniques on our dataset. Resampling makes sure that an equal percentage of the risky loans (and therefore good loans as well) are represented in both our training dataset and testing dataset. 
The file [credit_risk_resampling](https://github.com/SohaT7/Credit_Risk_Analysis/blob/main/credit_risk_resampling.ipynb) contains resampling exercises followed by running each model. The RandomOverSampler and SMOTE algorithms were used to oversample our data. The ClusterCentroids algorithm was used to undersample the data. The SMOTEENN algorithm combines the oversampling and undersampling technique. 
Next, ensemble learning models - the BalanceRandomForestClassifier and EasyEnsembleClassifier - were used to help reduce bias. Ensemble learning combines multiple models in order to improve the overall robustness and accuracy of the model. The file [credit_risk_ensemble](https://github.com/SohaT7/Credit_Risk_Analysis/blob/main/credit_risk_ensemble.ipynb) consists of code that runs those models on the dataset. 
## Results
After running each model, the balanced accuracy score, a confusion matrix, and a classification report was generated. The latter two provide us with values for precision, sensitivity (also known as "recall"), as well as the F1 score, all of which tell us how good a model is. In our results, 0 denotes a high risk or bad loan, and 1 denotes a low risk or good loan.

### Naive Random Oversampling with RandomOverSampler:
![Naive Random Oversampling](https://github.com/SohaT7/Credit_Risk_Analysis/blob/main/Image_Naive_Random_Oversampling.png)
The balanced accuracy score for this model is 0.64.
The precision of this model to predict high risk or bad loans is 0.01, whereas the precision to predict low risk or good loans is 1.00.
The recall or sensitivity of the model to predict high risk or bad loans is 0.66, whereas that for low risk or good loans is 0.62.

### SMOTE Oversampling:
![SMOTE Oversampling](https://github.com/SohaT7/Credit_Risk_Analysis/blob/main/Image_SMOTE_Oversampling.png)
The balanced accuracy score for this model is 0.65.
The precision of this model to predict high risk or bad loans is 0.01, whereas the precision to predict low risk or good loans is 1.00.
The recall or sensitivity of the model to predict high risk or bad loans is 0.61, whereas that for low risk or good loans is 0.69.

### Undersampling with ClusterCentroids:
![Undersampling with ClusterCentroids](https://github.com/SohaT7/Credit_Risk_Analysis/blob/main/Image_ClusterCentroids.png)
The balanced accuracy score for this model is 0.54.
The precision of this model to predict high risk or bad loans is 0.01, whereas the precision to predict low risk or good loans is 1.00.
The recall or sensitivity of the model to predict high risk or bad loans is 0.69, whereas that for low risk or good loans is 0.39.

### Combination Sampling with SMOTEENN:
![Combination Sampling with SMOTEENN](https://github.com/SohaT7/Credit_Risk_Analysis/blob/main/Image_SMOTEENN_Combo.png)
The balanced accuracy score for this model is 0.66.
The precision of this model to predict high risk or bad loans is 0.01, whereas the precision to predict low risk or good loans is 1.00.
The recall or sensitivity of the model to predict high risk or bad loans is 0.75, whereas that for low risk or good loans is 0.56.

### Balanced Random Forest Classifier:
![Balanced Random Forest Classifier](https://github.com/SohaT7/Credit_Risk_Analysis/blob/main/Image_Balanced_Random_Forest_Classifier.png)
The balanced accuracy score for this model is 0.79.
The precision of this model to predict high risk or bad loans is 0.03, whereas the precision to predict low risk or good loans is 1.00.
The recall or sensitivity of the model to predict high risk or bad loans is 0.70, whereas that for low risk or good loans is 0.87.

### Easy Ensemble AdaBoost Classifier:
![Easy Ensemble AdaBoost Classifier](https://github.com/SohaT7/Credit_Risk_Analysis/blob/main/Image_Easy_Ensemble_AdaBoost_Classifier.png)
The balanced accuracy score for this model is 0.93.
The precision of this model to predict high risk or bad loans is 0.09, whereas the precision to predict low risk or good loans is 1.00.
The recall or sensitivity of the model to predict high risk or bad loans is 0.92, whereas that for low risk or good loans is 0.94.

## Summary
