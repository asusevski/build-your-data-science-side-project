# Build Your Data Science Side Project

![tmp](https://user-images.githubusercontent.com/77211520/214369609-0b1c2e09-f8fe-4582-8fa0-fa79130f7303.png)

**TL;DR**: Goal was to predict, based on customer purchase history, whether the customer would default on their credit. Selected xgboost model which achived _% on test set. Some features that were highly predictive are:

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
![tmp](https://user-images.githubusercontent.com/77211520/214922667-31cd6874-6915-40a4-8b85-d4ebd6449c11.png)

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
for each customer and we need to aggregate their data in some way. We'll aggregate numerical features b mean and categorical features by count.

