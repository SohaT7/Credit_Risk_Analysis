# Credit_Risk_Analysis

## Overview of the analysis
### Purpose of the analysis:
The purpose of this analysis was to determine which model helps us best with resolving our credit card risk problem. 
### About the dataset:
The dataset we used is credit card dataset from LendingClub, a company which services the provision of loans among peers.
### Description of the analysis:
Since credit risk is an unbalanced classification problem by its very nature, i.e. the number of risky loans is easily always far less than the number of good loans, we used different resampling techniques on our dataset. Resampling makes sure that an equal percentage of the risky loans (and therefore good loans as well) are represented in both our training dataset and testing dataset. 
The file [credit_risk_resampling]() contains resampling exercises followed by running each model. The RandomOverSampler and SMOTE algorithms were used to oversample our data. The ClusterCentroids algorithm was used to undersample the data. The SMOTEENN algorithm combines the oversampling and undersampling technique. 
Next, ensemble learning models - the BalanceRandomForestClassifier and EasyEnsembleClassifier - were used to help reduce bias. Ensemble learning combines multiple models in order to improve the overall robustness and accuracy of the model. The file [credit_risk_ensemble]() consists of code that runs those models on the dataset. 
