# Introduction to GitHub Repository: Heart Disease Prediction

Welcome to our GitHub repository dedicated to the analysis and prediction of heart disease using decision trees, random forests, and neural networks. Our goal is to provide an in-depth exploration of a dataset sourced from the [2022 annual CDC survey data of over 400k adults](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease), identifying patterns and features that are indicative of heart disease.

## Problem Formulation

Our objective is to develop a predictive model that can determine the presence of heart disease in individuals by analyzing lifestyle choices, health factors, and biodemographic information. The key questions guiding our research are:
- Which machine learning model is most effective for prediction?
- What are the most influential factors contributing to the likelihood of heart disease?

## Why is this a concern?

Heart disease is the leading cause of death, accounting for 49% of fatalities, according to the American Heart Association in 2022. This rate exceeds the combined deaths from cancer and stroke. Although heart disease cannot be cured, it can be managed, particularly with early intervention, which is most effective. However, patients in the U.S. often wait up to a month for medical appointments. To mitigate this issue, we aim to develop an AI model to expedite early detection, enabling individuals to assess their risk independently and seek timely intervention.

## Dataset

The dataset includes the following variables:
1. **Heart Disease**: Whether the respondent has ever had heart disease (Yes/No).
2. **BMI**: Body Mass Index.
3. **Smoking**: Whether the respondent has smoked 100 cigarettes in their lifetime (Yes/No).
4. **Alcohol Drinking**: Excessive alcohol consumption as defined for adults (Yes/No).
5. **Stroke**: Whether the respondent has ever had a stroke (Yes/No).
6. **Physical Health**: Number of days of poor physical health in the past 30 days.
7. **Mental Health**: Number of days of poor mental health in the past 30 days.
8. **DiffWalking**: Difficulty with walking or climbing stairs (Yes/No).
9. **Sex**: Male or Female.
10. **Age Category**: Fourteen-level age categorization.
11. **Race**: Imputed race/ethnicity.
12. **Diabetic**: Whether the respondent has ever had diabetes (Yes/No).
13. **Physical Activity**: Any physical activity or exercise outside of regular job in the past 30 days (Yes/No).
14. **GenHealth**: General health condition.
15. **SleepTime**: Average number of hours of sleep in a 24-hour period.
16. **Asthma**: Whether the respondent has ever had asthma (Yes/No).
17. **Kidney Disease**: Whether the respondent has ever had kidney disease (Yes/No).
18. **Skin Cancer**: Whether the respondent has ever had skin cancer (Yes/No).

## Methodology

Our methodology for analyzing the dataset and developing the predictive model includes the following steps:

1. **Data Preprocessing**:
   - Clean the dataset.
   - Handle missing values.
   - Encode categorical variables using Label Encoding.
   - Cast data types appropriately.
   - Remove outliers using the Interquartile Range (IQR) method.
   - Eliminate duplicate entries.
  
2. **Exploratory Data Analysis (EDA):**
   In our EDA, we delve into various aspects to understand the impact of different variables on heart disease. Here's what we cover:

- **Cramer's V Correlation**: We utilize Cramer's V to understand the strength and significance of the relationships between categorical variables.
  
  ![Cramer's V Correlation](/Pictures/Cramer.png)

- **Correlation Matrix**: A correlation matrix to observe the linear relationships between features.
  
  ![Correlation Matrix](/Pictures/Corr.png)

- **Risk Activities Analysis**: We examine how engaging in risky activities correlates with the incidence of heart disease.
  
  ![Effect of Risky Activities](/Pictures/Effect1.png)

- **Lifestyle Choices Analysis**: We explore the effects of various lifestyle choices on the incidence of heart disease.
  
  ![Effect of Lifestyle Choices](/Pictures/Effect2.png)

3. **Model Development:**
   In the model development phase, we experiment with various algorithms to build a model that best predicts heart disease. Below are visualizations representing the performance of each model during the training phase.

  ### Decision Tree
  ![Decision Tree Model](/Pictures/DecisionTree.png)

  ### Random Forest
  ![Random Forest Model](/Pictures/RandomForest.png)

  ### Neural Network
  ![Neural Network Model](/Pictures/NeuralNetwork.png)

4. **Cross Validation:** 
  To ensure the robustness of our models, we employ K-fold cross-validation. This technique allows us to assess how the models will generalize to an independent dataset. Here's the   visualization for the cross-validation process:

  ![K-fold Cross Validation](/Pictures/CrossValidation.png)

5. **Model Evaluation**
  The following table outlines the performance metrics of our models:

  | Model          | Accuracy | Precision | Recall | F1-Score | AUC   |
  |----------------|----------|-----------|--------|----------|-------|
  | Decision Tree  | 88.1%    | 63.9%     | 6.0%   | 11.0%    | 81.8% |
  | Random Forest  | 88.0%    | 79.0%     | 52.0%  | 50.0%    | 83.0% |
  | Neural Network | 91.23%   | 61.0%     | 4.0%   | 7.4%     | 83.3% |

6. **Resampling**
   During our initial model training, we encountered a significant challenge: a high False Negative Rate of approximately 95%. This rate indicates that our models were prone to predicting   'No heart disease' when, in reality, heart disease was present. In medical diagnostic contexts, such a high rate of missed positive detections is unacceptable, as it could lead to   severe consequences for patient care.

  The root cause of this issue appears to be the imbalanced dataset, a concern we noted during our Exploratory Data Analysis (EDA). To address this, we have explored and implemented   various resampling techniques to balance our dataset:

  - **Manual Undersampling**: We reduced the majority class by 70%, which improved the True Positive Rate (TPR). Though there was a drop in overall accuracy, the model became more reliable in identifying actual cases of heart disease.

  - **Random Undersampling**: We randomly decreased the number of samples in the majority class to achieve balance. This method further decreased accuracy but significantly improved TPR.

  - **Oversampling the Minority Class**: By replicating samples in the minority class, we aimed to balance the dataset. The performance was comparable to undersampling techniques.

  - **Synthetic Minority Over-sampling Technique (SMOTE)**: SMOTE helps by generating synthetic samples from the minority class. Its performance was on par with both undersampling and oversampling.

  - **K-Means SMOTE**: A variant of SMOTE, this technique employs k-means clustering before generating synthetic samples. It produced the most meaningful results, improving model accuracy, TPR, and True Negative Rate (TNR).

## Neural Network Performance After Resampling

  The following table showcases the performance of our Neural Network model after implementing various resampling techniques:

  | Sampling Method  | Accuracy | Precision | Recall | F1-Score | AUC   |
  |------------------|----------|-----------|--------|----------|-------|
  | 70% of Majority  | 74.8%    | 74.1%     | 76.4%  | 75.1%    | 82.7% |
  | Undersampling    | 74.8%    | 74.1%     | 76.4%  | 75.1%    | 82.7% |
  | Oversampling     | 74.7%    | 71.6%     | 82.1%  | 76.4%    | 82.2% |
  | SMOTE Tomek      | 74.9%    | 72.1%     | 81.5%  | 76.5%    | 82.4% |
  | K-mean SMOTE     | 91.5%    | 93.%%     | 89.1%  | 91.3%    | 96.1% |

  It is evident from the table that K-mean SMOTE has significantly enhanced the Neural Network model's performance across all metrics, especially Accuracy and AUC, indicating a robust   predictive capability.
   
7. **Outcome and insight:**

  ### Problem Definition and Model Selection

  Our exploration into various machine learning models revealed distinct capabilities and challenges associated with each. For instance:

  - **Decision Tree**: Excellent for visual representation and interpretation, but prone to overfitting when used on training datasets without proper tuning.
  - **Random Forest**: Enhances efficiency over individual decision trees, adeptly handling large datasets with numerous variables due to its ensemble approach, thereby reducing the risk of overfitting.
  - **Neural Networks**: Demonstrated the highest accuracy among our models and were particularly effective in avoiding overfitting through techniques like regularization and network architecture optimization.

  ### Resolution of Sub-Problems

  Based on our comprehensive analysis, we selected specific models for particular tasks to optimize our predictive outcomes:

  - **Primary Model**: We utilized **Neural Network** because of its **Accuracy and AUC score**.
  - **Feature Importance**:
      Through our analysis using the Neural Network model, we've identified key factors that significantly impact the likelihood of having heart disease. These include:

      - **Physical Health**: The number of days a person experiences poor physical health plays a critical role.
      - **General Health**: Overall assessments of health, whether good or bad, are strong indicators.
      - **Age**: Age is a significant determinant, as risk increases with age.

### Final Thoughts

The selection of the Neural Network was crucial due to its superior performance metrics, making it the most reliable choice for our goal of accurately predicting heart disease. This approach aligns with our commitment to enhance diagnostic processes and improve patient outcomes through advanced AI technologies.
   
## Contribution:

| Team Member | Contributions                                             |
|-------------|-----------------------------------------------------------|
| Nga Wei En  | Data Preparation, EDA, Cross Validation, Neural Network Modeling, Resampling, Fine Tuning, Github Documentation |
| Bai YiFan   | Data Preparation, EDA, Decision Tree Modeling, Fine Tuning, Presentation, Video Editing |
| Lin HongChao    | Random Forest Modeling, Presentation, Documentation                      |
