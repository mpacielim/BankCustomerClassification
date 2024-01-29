# Practical Assignment 17.1 - Comparing Classifiers

**Author: Melissa Paciepnik**

## Introduction
This report details the findings on classification model performance for the bank customer dataset provided from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing).  

The data represents information on bank customers collated from 17 marketing campaigns by a Portugese banking institution.

### Goal
The business objective of the modelling was to determine which attributes of a customer and the marketing campagin directed towards this customer will result in the customer subscribing to a term deposit with the bank (yes = 1, no =0), after being contacted by the campaign.

The dataset was highly imbalanced towards 'no' answers, with ~89% of customers responding no, and the remaining 11% responding yes.

## Evaluation Rationale
A rudimentary baseline score of 89% accuracy was set.  This score is what a model that simply assigned all customers to a 'no' answer (the majority class) would achieve. The goal of a well performing model is to achieve an accuracy > 89%.

However, given the high imbalance in the dataset, choosing the best model based on accuracy is not wise, as this measures the total number of correct predictions only, including both 'no' and 'yes' answers.  

In this business context, we want to correctly identify as many potential customers that will say yes to opening a deposit account.

In this binary dataset, the Positive classifier 1 = yes, to opening an account, while the Negative classifier 0 = no.  This means that we value identifying a True Positives, and want to avoid False Negatives (this would mean a potential deposit customer was classified as a 'No', and the bank would have missed identifying and reaching out to a successful candidate).  In this case we want to **maximize Recall**, which is the measure of:

- (True Positives) \ (True Positives + False Negatives)

- In otherwords, out of all the customers who would have said yes, how many did the model identify as saying yes?

Recall will be high when we identify as many True Positives as possible, and minimize the number of False Negatives.  In other words, it will be high when we correctly classify as many 'yes' customers as possible, and minimize incorrectly classifying them as 'no'.

If this comes at the expense of low precision, meaning we accidentally classify 'no' customers as 'yes', and reach out to them, that's an acceptable risk as it may indeed result in a 'yes', or otherwise the customer can simply decline the offer.


## Results
### Initial Testing
4 classification models were tested with default settings and no hyperparameter tuning or cross validation. 

They all performed poorly, with test set accuracy at or below the baseline (of 0.885), and poor recall performance of < 0.3.

The models are listed below in order of best to worst recall performance on test data:
1. Decision Tree: 0.106 recall, 0.866 accuracy
2. KNN: 0.069 recall, 0.874 accuracy
3. SVC: 0.000 recall, 0.885 accuracy
4. Logistic Regression: 0.000 recall, 0.885 accuracy

### Secondary Testing
The same 4 models were tested using GridSearchCV and 5 k-fold cross validation to select the optimal hyperparameters, and then scored on the best performing hyperparameters for each model type.

Additionally, the data fed to the model was investigated and cleaned thoroughly to:
- Remove outliers
- Remove noisy data with poor distributions (leading to poor model performance)
- Scale numerical data
- Encode ordinal and nominal category data
- Remove nonsensical data (such as duration, as recommended by the dataset description)

This led to the following far improved results for recall, but varying for accuracy. 

The models are again listed below in order of best to worst performance, with optimal hyperparameters listed:

1. Logistic Regression 2: 0.620 recall, 0.716 accuracy
- Best hyperparameters: C value: 0.001, Weight: balanced, Penalty: L2

2. KNN 2: 0.214 recall, 0.875 accuracy
- Best hyperparameters: n_neighbors = 3, weights = 'distance'

3. Decision Tree 2: 0.189 recall, 0.899 accuracy
- Best hyperparameters: criterion = 'gini', max_depth = 7, min_samples_split = 5

4. SVC 2: 0.174 recall, 0.899 accuracy
- Best hyperparameters: C value = 1
- This model also took by far the longest time to train, at 6 minutes compared with <1 minute for all other models.

The best performing model in terms of accuracy was tied with the tuned Decision Tree Model and SVC, which both barely outperformed the baseline 88.5% accuracy with 89.9% accuracy, however with respective low recalls of 0.189 and 0.174.

The best performing recall model (the Tuned Logistic Regression model) increased from the best performing Non-Tuned Model (Decision Tree) from a recall of 0.106 to 0.620.  This means that the number of correctly identified 'yes' candidates increased from 126 (10.7% of total 'yes' customers) to 681 (62%), and the number of yes candidates who were incorrectly identified as a 'no' dropped from 1,057 (89.3%) to 417 (38%).

However there was a tradeoff with precision which cause the accuracy to fall from the baseline of 88.5% to 71.6%.

## Next Steps and Recommendations
Ultimately, the cleaning of data and tuning of hyperparameters showed promising improvements in the desired metric of recall, but further tuning is required to improve both recall and accuracy.

Actions for further data gathering:
- Given how the strong imbalance of data has clearly affected the model performance, it is recommended that more campaigns be run to:
  - Increase the number of 'yes' candidates captured in the data, through more effective campaigns
  - Review additional metrics that could be included in the new campaigns and gathered for use in improving model performance. 

Suggestions from improvement include:
- Performing gridsearch over a wider array of hyperparameters.
- Specifically investigate the effect of changing probability thresholds and weighting of neighbouring samples, given the imbalance in the dataset.
- Incorporating ensemble methods or neural networks to improve performance.

## Links
[Link to Dataset](https://github.com/mpacielim/BankCustomerClassification/tree/main/data)

[Link to Technical Report](https://github.com/mpacielim/BankCustomerClassification/blob/main/Bank%20Customer%20Classifier.ipynb)
