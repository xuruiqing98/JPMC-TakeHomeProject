# Income Prediction & Population Segmentation (Census Data)

## 📌 Project Overview

In practice, building a useful machine learning system is not about maximizing model accuracy, but about making reliable decisions under constraints.

This project reframes income prediction as a resource allocation problem, where marketing actions must be prioritized across a diverse population with limited budget. By integrating classification and segmentation, the system identifies not only who is likely to be high-value, but also how different groups should be targeted.

The focus is therefore on decision quality rather than model performance alone, ensuring that outputs are interpretable, robust, and directly applicable to real-world marketing strategy.

## Key Results

- ROC AUC: ~0.95, indicating strong ranking performance
- Precision-focused threshold reduces Customer Acquisition Cost (CAC) inefficiency
- Segmentation enables 4 actionable customer personas
- Deprioritizing low-value segments reduces unnecessary outreach while preserving high-value coverage

---

## 🎯 Objectives

- Build a **robust classification model** for income prediction
- Perform **interpretable segmentation** based on demographic and labor-market characteristics
- Ensure all analysis respects the **survey sampling structure (weight variable)**

---

## 📂 Project Structure

TakeHomeProject/
├── data/
│   ├── census-bureau.data
│   ├── census-bureau.columns
├── notebooks/
│   └── EDA.ipynb
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train_model.py
│   └── segmentation.py
├── results/
│   ├── model_metrics.json
│   ├── feature_importance.csv
├── models/
│   └── best_model.joblib
├── README.md
├── requirements.txt

---

## Run Pipeline

```bash
# Train classification model
python src/train_model.py

# Run segmentation
python src/segmentation.py```

These steps form the end-to-end pipeline for model training and segmentation analysis.
---

## 🔍 Key Design Decisions

### 1. Treating Integer Codes as Categorical

Several variables (e.g., industry codes, occupation codes, year) are stored as integers but represent categories.

→ These are explicitly handled as **categorical features**, not numeric.

---

### 2. Handling Sampling Weights

The dataset includes a `weight` variable representing population importance.

- Used in:
  - Model training (`sample_weight`)
  - Evaluation metrics
  - Segmentation summaries (weighted averages)

→ Ensures results reflect **true population distribution**

---

### 3. Avoiding Data Leakage

- Data is split **before preprocessing**
- Encoder and imputations are fitted **only on training data**
- Same transformations are applied to validation/test sets

---

### 4. Encoding Strategy

- Used `OneHotEncoder(handle_unknown="ignore")`
- Prevents feature mismatch between train/test
- Ensures production-safe pipeline

---

### 5. Handling Missing Values

- Numeric → filled with median (robust to skew)
- Categorical → filled with "Missing"

---

## 📊 Exploratory Data Analysis (EDA)

Key findings:

- Dataset is **dominated by categorical variables**
- Income distribution is **highly imbalanced**
- Financial variables (capital gains, dividends) are **extremely skewed**
- Many fields include "Not in universe" → represents structural absence (not missing)

---

## 🤖 Modeling
###  Modeling Philosophy

This project focuses not only on model accuracy, but on decision reliability.

In imbalanced classification settings, accuracy can be misleading. Therefore:
- Threshold tuning is used to optimize F1-score
- Precision-recall tradeoff is explicitly analyzed
- Model performance is evaluated from a decision-making perspective

This reflects real-world deployment scenarios where business outcomes depend on correct identification of high-value segments.
In addition, model decisions are explicitly tied to business tradeoffs.

This framing ensures that model outputs are actionable and aligned with business objectives, rather than optimized purely for statistical performance.

False positives correspond to unnecessary marketing spend, while false negatives represent missed high-income opportunities. Therefore, model threshold selection is treated as a resource allocation decision rather than purely a statistical optimization problem.

### Models Evaluated

- Logistic Regression
- Random Forest
- (Optional) XGBoost

---

### Evaluation Strategy

Due to class imbalance:

- Accuracy is **not sufficient**
- Focus on:
  - Precision
  - Recall
  - F1-score
  - ROC-AUC

---

### Threshold Optimization

Threshold selection is treated as a business decision, balancing Customer Acquisition Cost (CAC) against missed high-value opportunities.

Additionally, threshold sensitivity is evaluated:

- 0.3 → high recall
- 0.5 → balanced
- 0.7–0.85 → high precision

A higher decision threshold (0.85) is selected to prioritize precision, reducing wasted marketing effort on low-probability individuals, while accepting some loss in recall.

---

### Final Model

- Best model: Logistic Regression
- ROC-AUC: ~0.95
- F1-score: ~0.59

---

### Key Insight

Although accuracy is high (~94%), it is not meaningful due to class imbalance.

The model prioritizes identifying high-income individuals through optimized precision-recall tradeoff.

---

## 👥 Segmentation

### Method

- K-Means clustering
- Standardized numeric + encoded categorical features
- Optimal K selected via **Silhouette Score**

---

### Cluster Selection

K tested: 3–6  
Best K: 6 (highest silhouette score)

However:

- Some clusters are very small (<1%)
- Tradeoff between statistical fit and interpretability

---

### Key Segments Identified

The clustering reveals a **lifecycle-driven structure**:

- Children / Non-working population
- Core workforce (mid-age, full-time)
- High-income professionals (educated, married)
- Low participation / older groups

---

### Business Insight

Segmentation is primarily driven by labor-force participation, education level, and life stage. This indicates that income is structurally linked to employment engagement and socio-economic stability, with high-value individuals concentrated in professionally stable groups.

From a business perspective, these segments enable differentiated marketing strategies. High-income, stable professionals can be directly targeted with premium products, while early-career individuals benefit from long-term engagement strategies. Low participation groups are deprioritized due to limited expected return.

By integrating segmentation with model predictions, the system supports segment-specific playbooks where targeting decisions are informed by both predicted income probability and underlying socio-economic context. This allows marketing resources to be allocated more efficiently, improving return on investment while reducing unnecessary outreach.

---

## ⚠️ Important Considerations

### Weight Sensitivity

Ignoring `weight` would lead to:

- Biased cluster sizes
- Misleading averages

---

### High-Dimensional Encoding

One-hot encoding increases dimensionality significantly (~500 features)

→ Managed via model choice (tree-based + regularization)

---

## 🚀 How to Run

### 1. Install dependencies

pip install -r requirements.txt

---

### 2. Train model

python src/train_model.py

Outputs:

- results/model_metrics.json
- results/feature_importance.csv
- models/best_model.joblib

---

### 3. Run segmentation

python src/segmentation.py

Outputs:

- Cluster distribution (weighted)
- Numeric summaries
- Top categorical features per cluster

---

### All dependencies are pinned to stable versions to ensure reproducibility.


## 🧠 Final Takeaways

- Income prediction is highly influenced by:
  - employment intensity
  - education
  - financial activity

- Segmentation reveals:
  - clear lifecycle and labor-market structures
  - interpretable population groups

- Proper handling of sampling weight is critical for valid insights

- Model outputs should be interpreted as decision-support signals rather than absolute predictions, and used in conjunction with business constraints.

---

## ⚠️ Risk & Limitations

- The dataset is based on 1994–1995 census data, which may not reflect current labor market dynamics.
- Demographic variables (e.g., gender, occupation) may introduce bias, raising fairness considerations in marketing decisions.
- Model performance may degrade over time due to distribution shift, requiring ongoing monitoring through Population Stability Index (PSI) and feature drift tracking.

These factors should be considered before deploying the model in real-world settings.

---

## 💡 Future Improvements

- Use ordinal encoding for education
- Reduce high-cardinality categorical features
- Evaluate clustering stability across different K
- Apply dimensionality reduction (PCA)

---

## 👤 Author

Ruiqing Xu  
PhD Candidate – Machine Learning & Data Analytics
