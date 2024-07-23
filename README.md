# Heart Disease Prediction Project

## Project Overview

This project aims to predict the presence of heart disease in patients using various machine learning algorithms. The dataset used is the UCI Heart Disease dataset, which contains medical information about patients, including attributes such as age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, ST depression induced by exercise relative to rest, slope of the peak exercise ST segment, number of major vessels colored by fluoroscopy, and thalassemia.

## Dataset Acknowledgement
Heart Disease UCI [link](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

## Features

| Feature                | Description                                                | Units        |
|------------------------|------------------------------------------------------------|--------------|
| age                    | Age                                                        | Years        |
| sex                    | Sex                                                        | -            |
| cp                     | Chest pain type (4 values)                                 | -            |
| trestbps               | Resting blood pressure (in mm Hg on admission to the hospital) | mm Hg        |
| chol                   | Serum cholesterol                                          | mg/dl        |
| fbs                    | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)      | -            |
| restecg                | Resting electrocardiographic results (values 0 = normal, 1, 2) | -            |
| thalach                | Maximum heart rate achieved                                | bpm          |
| exang                  | Exercise induced angina (1 = yes; 0 = no)                  | -            |
| oldpeak                | ST depression induced by exercise relative to rest         | -            |
| slope                  | The slope of the peak exercise ST segment (values 1 = upsloping, 2 = flat, 3 = downsloping) | -            |
| ca                     | Number of major vessels (0-3) colored by fluoroscopy       | -            |
| thal                   | 3 = normal; 6 = fixed defect; 7 = reversible defect        | -            |

## Models and Performance

Three machine learning models were implemented and compared: Support Vector Machine (SVM), Logistic Regression, and Naive Bayes. Below is a summary of their performance:

| Metric                  | SVM       | Logistic Regression | Naive Bayes |
|-------------------------|-----------|---------------------|-------------|
| **Accuracy**            | 81.3%     | 81.3%               | 83.5%       |
| **Precision (No Disease)** | 0.80    | 0.80                | 0.78        |
| **Precision (Heart Disease)** | 0.82 | 0.82                | 0.89        |
| **Recall (No Disease)** | 0.78      | 0.78                | 0.88        |
| **Recall (Heart Disease)** | 0.84   | 0.84                | 0.80        |
| **F1-Score (No Disease)** | 0.79    | 0.79                | 0.83        |
| **F1-Score (Heart Disease)** | 0.83 | 0.83                | 0.84        |

## Explanation

### Summary of Findings:
1. **Accuracy**:
   - Both the SVM and Logistic Regression classifiers achieved an accuracy of approximately 81.3%.
   - The Naive Bayes classifier outperformed the other two with an accuracy of approximately 83.5%.

2. **Precision**:
   - For predicting "no disease", SVM and Logistic Regression have higher precision (0.80) compared to Naive Bayes (0.78).
   - For predicting "heart disease", Naive Bayes has the highest precision (0.89), significantly better than both SVM and Logistic Regression (0.82).

3. **Recall**:
   - For predicting "no disease", Naive Bayes achieves the highest recall (0.88), indicating it is very effective at identifying patients without heart disease.
   - For predicting "heart disease", SVM and Logistic Regression have higher recall (0.84) compared to Naive Bayes (0.80).

4. **F1-Score**:
   - For predicting "no disease", Naive Bayes has the highest F1-score (0.83), indicating a good balance between precision and recall.
   - For predicting "heart disease", the F1-scores are very close, with Naive Bayes slightly leading (0.84) over SVM and Logistic Regression (0.83).

### Conclusion:
Based on the comparison, the Naive Bayes classifier demonstrates superior performance overall, particularly in accuracy and precision for predicting heart disease. This makes Naive Bayes the most suitable model for this specific heart disease prediction task. While SVM and Logistic Regression also perform well and consistently, they do not surpass Naive Bayes in key metrics. Therefore, for the dataset used, Naive Bayes is recommended as the best-performing classifier.

## Next Steps

- Experimenting with more advanced machine learning models such as Random Forests, Gradient Boosting Machines (GBM), and XGBoost can provide insights into their performance compared to the current models.
- Deep learning approaches using neural networks, specifically feedforward neural networks or more sophisticated architectures like convolutional neural networks (CNNs) and recurrent neural networks (RNNs), can be explored for possibly better accuracy and predictive power.
- Conducting hyperparameter tuning using techniques such as grid search or randomized search can optimize model performance.
- Implementing cross-validation will ensure that the results are robust and generalizable to new, unseen data.

## Installation

To run this project, ensure you have the following libraries installed:

```sh
pip install pandas numpy matplotlib seaborn scikit-learn
