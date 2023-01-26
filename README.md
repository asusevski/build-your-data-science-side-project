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
First, exploring just the categorical variables. Note most of the categorical variables are under the Delinquency variables category, meaning they're likely very 
important.

![cat_vars](https://user-images.githubusercontent.com/77211520/214881916-9d6545a1-93e9-405f-b64d-68c2c1d7f211.png)




