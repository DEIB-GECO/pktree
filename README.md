<!-- <img src="logo.png" alt="Logo" width="110" align="left" style="margin-right: 15px;">  

# PkTree:
_Incorporating Prior Knowledge into Tree-Based Models_
 -->

<table>
<tr>
<td>
<img src="logo.png" alt="Logo" width="300" style="margin-right: 15px;">
</td>
<td>
<h1 align="left">PkTree</h1>
<h4 align="left" style="font-weight: medium; color: gray;">
A Python package for incorporating prior domain knowledge into tree-based models
</h3>
</td>
</tr>
</table>


**PkTree** is a Python package that enables the integration of prior knowledge into Decision Trees (DT) and Random Forests (RF). By prioritizing relevant features, this package enhances interpretability and aligns predictive models with prior insights.  

The enhancements in **PkTree** build upon the `scikit-learn` library.

---

## **Features**  

### 1. **Prior-Informed Decision Trees**  
We introduce two key modifications to the traditional Decision Tree algorithm to prioritize relevant features during tree construction:  
- **Feature Sampling**: Weighted feature sampling during training.  
- **Impurity Improvement**: Adjusting impurity calculations based on prior knowledge scores.  

The modified models include a parameter `pk_configuration`, which can take the following values:  
- `'no_gis'`: Standard tree without knowledge.  
- `'on_feature_sampling'`: Applies prior knowledge-informed feature sampling.  
- `'on_impurity_improvement'`: Incorporates prior knowledge scores in impurity computations.  
- `'all'`: Combines both feature sampling and impurity improvement.  

---

## **Approaches**  

### **1. Feature Sampling**  
This approach replaces the standard random feature sampling with a weighted strategy:  
- Each feature is assigned a weight corresponding to its prior-knowledge relevance (`w_prior^-1`).  
- Features are sampled using a Fisher-Yates-based algorithm with weights normalized to probabilities.  
- A hyperparameter `k` controls the influence of prior knowledge on sampling.  

### **2. Impurity Improvement**  
When determining the best split, the traditional impurity calculation is modified:  
- The standard impurity improvement value is multiplied by the feature's `w_prior`.  
- An additional hyperparameter `v` controls the strength of the prior knowledge score's impact.  

---

## **Random Forest Extensions**  

### **1. Out-of-Bag (OOB) Weights**  
This approach leverages Out-of-Bag predictions for weighting individual estimators in the Random Forest ensemble:  
- For each tree, calculate:
  - `f_score`: Accuracy on OOB samples.  
  - `s_prior`: Average prior-knowledge relevance (`w_prior^-1`) of selected features.  
- Compute weights for each tree based on these scores and normalize them.  
- A hyperparameter `r` increases the weight differences across trees, enhancing the influence of prior-knowledge scores. 

### **2. Weighted Voting**  
During prediction:  
- Tree predictions are weighted based on their normalized scores.  
- Predictions are aggregated in parallel using these weights, aligning final results with prior-knowledge informed estimators.  

---

## **Getting Started**  

### **Installation**  
Install the package via `pip`:  
```bash
pip install pktree
```

### **Example Usage**  
Here’s how to use pktree to build and train a prior-knowledge informed Decision Tree or Random Forest model:

### **Creation of synthetic datasets**
First of all let's build a simple synthetic dataset, and extract a ficticious w_prior for each feature.
```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

# Function for extraction of fictitious w_prior
def assign_feature_scores(n_features = 50):
    scores = np.round(np.random.uniform(0.01, 0.99, size=n_features), 5)
    return scores

# Function for the generation of synthetic dataset
def generate_dataset(task_type, n_samples = 100, n_features = 50, noise_level = 0.1):
    
    if task_type == 'classification':
        X, y = make_classification(
            n_samples=n_samples, 
            n_features=n_features, 
            n_informative=int(n_features * 0.7), 
            n_redundant=int(n_features * 0.2), 
            n_classes=2, 
            random_state=42
        )

        # Add noise to the data frame
        X += np.random.normal(0, noise_level, X.shape)
        
    elif task_type == 'regression':
        X, y = make_regression(
            n_samples=n_samples, 
            n_features=n_features, 
            noise=noise_level, 
            random_state=42
        )

    else:
        raise ValueError("task_type must be either 'classification' or 'regression'")

    return X, y

# Obtain the scores and datasets
w_prior = assign_feature_scores()
X_classification, y_classification = generate_dataset('classification')
X_regression, y_regression = generate_dataset('regression')
        
```
### **Decision Trees**
Now that we have the data, let us see some example of usage for DecisionTreesClassifier:
```python
from pktree import tree

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_classification, y_classfication, test_size=0.2, random_state=42)

# Create a domain-informed Decision Tree model
model = tree.DecisionTreeClassifier(random_state=42, pk_configuration='all', w_prior=w_prior, k=1, v=1, pk_function='linear')

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```
and DecisionTreeRegressor:
```python
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)

# Create a domain-informed Decision Tree model
model = ensemble.RandomForestRegressor(random_state=42, pk_configuration='on_impurity_improvement', oob_score=True, on_oob=True, w_prior=w_prior, pk_function='reciprocal')

#Make predictions
y_pred = model.predict(X_test)
```
### **Random Forest**
For RandomForestClassifier:  
```python
from pktree import ensemble

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_classification, y_classfication, test_size=0.2, random_state=42)

# Create a domain-informed Random Forest model
forest = ensemble.RandomForestClassifier(random_state=42, pk_configuration='on_feature_sampling', oob_score=True, on_oob=True, w_prior=w_prior)

# Train the model
forest.fit(X_train, y_train)

# Make predictions
predictions = forest.predict(X_test)
```
and RandomForestRegressor:
```python
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)

# Create a domain-informed Random Forest model
forest = ensemble.RandomForestRegressor(random_state=42, pk_configuration='on_impurity_improvement', oob_score=True, on_oob=True, w_prior=w_prior)

# Train the model
forest.fit(X_train, y_train)

# Make predictions
predictions = forest.predict(X_test)
```

---

## **Compatibility**  
- Built on top of `scikit-learn`.  
- Compatible with both classification and regression tasks.  

---

## **License**  
This package is open-source and distributed under the [MIT License](LICENSE).  

--- 

We welcome contributions and feedback to enhance **pktree** further! 
