# Build Your Data Science Side Project

![tmp](https://user-images.githubusercontent.com/77211520/214369609-0b1c2e09-f8fe-4582-8fa0-fa79130f7303.png)

*Put together as part of an introductory workshop series to demonstrate an example of a data science side project.*

---
**TL;DR**: Goal was to predict, based on customer purchase history, whether the customer would default on their credit. Selected LGBM model which achived 0.491 score on test set with an F1 score of 0.88.

The project is organized as follows:
1. Background on problem space
    - Business problem (why is this important to businesses?)
    - Data science problem (background on problem space from a data scientists perspective)
2. Data
    - Where we got the data from
    - Evaluation metrics
    - EDA takeaways
    - Feature Engineering process
3. Models
    - Training different models
4. Model Analysis
    - Performance on different splits
    - Model Interpretability
    - Feature importance
5. Limitations

# Background on problem space
### Business problem:
To quote Amex themselves, credit default prediction is central to managing risk in a consumer lending business. 

> Credit default prediction allows lenders to optimize lending decisions, which leads to a better customer experience and sound business economics. - Amex, on Kaggle competititon overview

Thus we have a very well-defined business problem that is of utmost importance to all credit card companies.

### Data science problem
From a data scientists perspective, this is a classifiction problem that deals with unbalanced data. This is because most people do, in fact, pay their credit card
off. 

# Data
### Where we got the data from
The data was released for Amex's Kaggle challenge, but upon first release it was a monstrous 50GB. Fortunately, the community got together to create a post processed
dataset that transformed any floats to integer wherever it could be done without information loss. This, along with storing in a parquet file, allowed for the data to 
be brought to a manageable 5GB, roughly.

Here is a link to the post-processed data: https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format

An important note about this dataset is we are never exactly told what each of the variables are, just the category they fall into.
Ie:
- D_* = Delinquency variables
- S_* = Spend variables
- P_* = Payment variables
- B_* = Balance variables
- R_* = Risk variables

### Evaluation metrics
The evaluation metric used for this competition was the mean of two other metrics. Specifically, it was the mean of the normalized Gini Coefficient and the default
rate, captured at 4%.
The **Normalized Gini Coefficent** is best explained with an image. 
Suppose we have the following graph:

![tmp](https://user-images.githubusercontent.com/77211520/214896228-eac118b5-c9c7-49ff-a7b8-54a4d69bb959.png)

If the Orange area is "A" and the blue area is "B", the Gini Coefficient is calculated as $G=\frac{A}{A+B}$. The Normalized Gini Coefficient is calculated by simply dividing by the maximum possible Gini Coefficient , ie: $NG=\frac{1}{G_{max}}\frac{A}{A+B}$

*Note*: the Gini Coefficient has a relationship to AUC, or Area Under the Curve. It can be calculated as $G=2*\text{AUC}-1$

The default rate captured at 4%, on the other hand, refers to the percentage of positive labels captured in the highest ranked top 4% of predictions.

### EDA Takeaways

It is always good to start with null values. While we have 189 variables, not including target and customer ID, we have 67 variables with null values.

**Correlations**

Looking at correlations, we get the following plot of correlations with the target:
![tmp](https://user-images.githubusercontent.com/77211520/214955511-2434bbc2-7332-47eb-af1d-5e3e6e008f02.png)

Most features are not siginificantly correlated with the target. However, there are many features that are correlated with each other.
![tmp](https://user-images.githubusercontent.com/77211520/214926753-78c90057-71bb-420b-a0e2-0c5240e8cf96.png)

This tells us we may want to consider removing some of these to reduce the dimensionality and improve training speed and model performance.

**Categorical variables**. Exploring the categorical variables, note most of the categorical variables are under the Delinquency variables category, meaning they're likely very important.

![tmp](https://user-images.githubusercontent.com/77211520/214927596-2de1f836-cde8-4993-9243-54561ad32705.png)

**Numerical variables**. There are a lot of numeric variables, so to examine each of the variables, we look at:

![tmp](https://user-images.githubusercontent.com/77211520/214931380-15a4c508-d200-4400-a207-3e34b13f9c3e.png)

Note almost all variables have a very small variance. In fact, only 9 of the numeric variables have a variance bigger than 1.

### Feature Engneering process:
The features have already been standardized. However, since we're dealing with data at the customer level, we potentially have multiple rows 
for each customer and we need to aggregate their data in some way. We'll aggregate numerical features by calculating their mean and categorical features by count.

# Models
### Training Different Models
Using **Lazy Classifier**, we tried 24 different models. Here are the results.

![tmp](https://user-images.githubusercontent.com/77211520/214947393-a5e1c668-2301-4afe-8edc-f151995a1ba5.png)

Note that we have a custom metric for this challenge, however. So we'll re-test with correct metric with just the top 2 classifiers.
LGB had a custom metric score of 0.4909679724727166 and XGB had a custom metric score of 0.48498807067754673. So, we choose LGB.

# Model Analysis
### Performance on Different splits
This subpoint is a "model fairness" consideration, but the features have been anonymized so we don't have any meaningful splits.
We'll analyze based on different ground truth labels instead by examining the confusion matrix.

Confusion matrix:
| Actual\Predicted | Negative | Positive |
| ----------  | ------ | ---------  |
|    Negative | 12425  | 1120       |
|    Positive | 1139   | 3318       |

So, we see the model had its struggles, failing to classify 1139 examples as positive in the test set. This has profound business implications, as this means 
1139 customers are not paying back their loans.

### Model Interpretability
Fortunately, LightGBM is an inherently interpretable model by nature of it being random forest based. 

### Feature Importance
Examining model with **SHAP**, we see the following graph:

![tmop](https://user-images.githubusercontent.com/77211520/214959834-60a3e701-3fde-4822-b1f7-6055413581fb.png)

We have only one feature that contributes hugely to the prediction of our model!

Examining in a dependence plot:

![tmp](https://user-images.githubusercontent.com/77211520/214960027-b5a9ab16-4141-4d07-b2fa-b38edf36c4b0.png)

Examining Permutation Importance as well, we have the following plot:

![tmp](https://user-images.githubusercontent.com/77211520/214964551-887e66b2-f44a-4560-ba03-b633e6eac58f.png)

As we saw earlier, there is one main feature that is important in predicting the values.

# Limitations
This project was limited by the fact that we did not try aggregating the features for each customer in many ways. We just stuck with one, which is certainly 
suboptimal. For example, imagine if we had a customer who 
