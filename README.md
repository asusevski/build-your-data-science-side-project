# Build Your Data Science Side Project

![tmp](https://user-images.githubusercontent.com/77211520/214369609-0b1c2e09-f8fe-4582-8fa0-fa79130f7303.png)

**TL;DR**: Goal was to predict, based on customer purchase history, whether the customer would default on their credit. Selected xgboost model which achived _% on test set. Some features that were highly predictive are:

The project is organized as follows:
1. Background on problem space
    - Business problem (why is this important to businesses?)
    - Data science problem (background on problem space from a data scientists perspective)
2. Data
    - Where we got the data from
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

### EDA Takeaways


data: https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format
