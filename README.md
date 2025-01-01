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
A Python package for incorporating biological prior knowledge into tree-based models
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
Here’s how to use pktree to build and train a prior-knowledge informed Decision Tree model:  
```python
from pktree import tree


# Create a biology-informed Decision Tree model
model = tree.DecisionTreeClassifier(pk_configuration='all', k=2.0, v=1.0)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

For Random Forests:  
```python
from pktree import ensemble

# Create a biology-informed Random Forest model
forest = ensemble.RandomForestClassifier(pk_configuration='all', v=0.35)

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