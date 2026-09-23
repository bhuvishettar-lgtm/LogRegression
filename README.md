# MAGIC Gamma Telescope Classification using Logistic Regression

A Machine Learning pipeline for classifying Cherenkov gamma telescope signals (`g`) against background hadron noise (`h`) using Logistic Regression with standardized features and oversampling.

---

## Technical Overview

- **Algorithm**: Logistic Regression (`sklearn.linear_model.LogisticRegression`)
- **Dataset**: MAGIC Gamma Telescope Dataset (`19,020` total samples, `10` continuous features)
- **Data Preprocessing**: Standard Scaling (`StandardScaler`) & Oversampling (`RandomOverSampler`)
- **Data Splitting**: 60% Train / 20% Validation / 20% Test
- **Performance Summary**: Achieved **79% Accuracy** on the test split.

---

## Features

| Feature Column | Description |
| :--- | :--- |
| `fLength` | Major axis of ellipse [mm] |
| `fWidth` | Minor axis of ellipse [mm] |
| `fSize` | 10-log of sum of content of all pixels |
| `fConc` | Ratio of sum of two highest pixels to fSize |
| `fConc1` | Ratio of highest pixel to fSize |
| `fAsym` | Distance from highest pixel to center |
| `fM3Long` | 3rd root of 3rd moment along major axis |
| `fM3Trans` | 3rd root of 3rd moment along minor axis |
| `fAlpha` | Angle of major axis with vector to origin |
| `fDist` | Distance from origin to center of ellipse |
| `class` | Target Variable: Gamma (`g` $\rightarrow$ `1`), Hadron (`h` $\rightarrow$ `0`) |

---

## Model Evaluation

Classification report on the test evaluation dataset ($N = 3,804$):

```text
              precision    recall  f1-score   support

           0       0.68      0.72      0.70      1295
           1       0.85      0.82      0.84      2509

    accuracy                           0.79      3804
   macro avg       0.77      0.77      0.77      3804
weighted avg       0.79      0.79      0.79      3804
