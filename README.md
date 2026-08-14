# AI-Powered-Financial-Fraud-Detection-System
## 📌 Project Overview

The AI-Powered Financial Fraud Detection System is a machine learning-based application designed to identify potentially fraudulent financial transactions.

The project combines data analysis, exploratory data analysis (EDA), feature engineering, machine learning, and Streamlit deployment to build an interactive fraud detection system.

The system analyzes transaction-related information such as transaction type, transaction amount, and account balances and predicts whether a transaction is fraudulent or not fraudulent.

## 🎯 Problem Statement

Financial fraud can cause significant financial losses and is difficult to identify manually when dealing with a large number of transactions.

The objective of this project is to develop a machine learning-based system that can learn patterns from historical transaction data and classify a transaction as:

0 → Not Fraudulent
1 → Fraudulent

The trained model is integrated into a Streamlit application so that users can enter transaction details and receive a fraud prediction.

## 🎯 Project Objectives
 - Analyze financial transaction data.
 - Perform exploratory data analysis (EDA).
 - Understand fraudulent transaction patterns.
 - Perform feature engineering.
 - Prepare the data for machine learning.
 - Handle the highly imbalanced fraud classification problem.
 - Train a machine learning classification model.
 - Evaluate the model using classification metrics.
 - Save the complete preprocessing and prediction pipeline.
 - Build an interactive Streamlit application for fraud prediction.

## ✨ Key Features
 - Financial transaction analysis
 - Exploratory data analysis and visualization
 - Fraud distribution analysis
 - Feature engineering
 - Numerical feature scaling
 - Categorical feature encoding
 - Imbalanced classification handling
 - Logistic Regression-based fraud prediction
 - Model evaluation using precision, recall and F1-score
 - Confusion matrix analysis
 - Interactive Streamlit interface
 - Saved machine learning pipeline for application deployment

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning and preprocessing |
| Joblib | Saving the trained ML pipeline |
| Streamlit | Interactive web application |

## 📊 Dataset

The project uses a financial transaction dataset containing 6,362,620 transactions and 11 columns.

The original dataset contains the following columns:

 - step
 - type
 - amount
 - nameOrig
 - oldbalanceOrg
 - newbalanceOrig
 - nameDest
 - oldbalanceDest
 - newbalanceDest
 - isFraud
 - isFlaggedFraud

The target variable is:

isFraud

where:

0 → Not Fraudulent
1 → Fraudulent

The dataset contains:

 - 6,354,407 non-fraudulent transactions
 - 8,213 fraudulent transactions

Fraudulent transactions represent approximately 0.13% of the complete dataset, making this a highly imbalanced classification problem.

## 🔎 Exploratory Data Analysis

The notebook performs exploratory analysis to understand the transaction data and identify patterns related to fraud.

### Data Inspection

The dataset was examined using:

 - head()
 - info()
 - columns
 - shape
 - value_counts()
 - missing-value analysis

The dataset contains 6,362,620 rows and 11 columns.

### Missing Values

Missing values were checked using:

- df.isnull().sum()

The analysis showed 0 missing values across the dataset.

### Fraud Distribution

The target variable was analyzed using:

- df['isFraud'].value_counts()

The analysis revealed a strong imbalance between fraudulent and non-fraudulent transactions.

### Transaction Type Analysis

Transaction types were visualized using bar plots to understand the distribution of different transaction categories.

### Fraud Analysis by Transaction Type

The analysis also focused on TRANSFER and CASH_OUT transactions to investigate the distribution of fraudulent transactions across these transaction types.

## 🧮 Feature Engineering

Two additional features were created to capture balance differences during a transaction:

### Origin Account Balance Difference

 balance_diffOrg = oldbalanceOrg - newbalanceOrig
 
### Destination Account Balance Difference

 balance_diffDest = oldbalanceDest - newbalanceDest

These features provide additional information about how account balances change during a transaction.

## 🧹 Feature Selection

The following columns were removed before model training:

 - nameOrig
 - nameDest
 - isFlaggedFraud

The modeling dataset therefore focuses on transaction type, transaction amount, account balances, and engineered balance differences.

The target variable isFraud was separated from the input features.

## ⚙️ Data Preprocessing

The project uses a Scikit-learn preprocessing pipeline.

### Categorical Feature

The type column is categorical and is processed using:

OneHotEncoder(drop="first")

### Numerical Features

The following numerical features are standardized using StandardScaler:

- amount
- oldbalanceOrg
- newbalanceOrig
- oldbalanceDest
- newbalanceDest
  
### Train-Test Split

The dataset was divided into training and testing sets using:

  train_test_split(X,y,test_size=0.3,random_state=42)

This results in a 70% training and 30% testing split.

## 🤖 Machine Learning Model

The project uses Logistic Regression as the classification algorithm.

Because fraudulent transactions are highly underrepresented in the dataset, the model uses:

   LogisticRegression(class_weight="balanced",max_iter=1000)

Using class_weight="balanced" helps the model give greater importance to the minority fraud class.


## 🔗 Machine Learning Pipeline

The project uses a **Scikit-learn Pipeline** to combine data preprocessing and machine learning into a single workflow.

### Pipeline Workflow

```text
Input Transaction Data
        ↓
   ColumnTransformer
        ↓
 ┌──────────────────────────────┐
 │                              │
 │ Numerical Features           │
 │ → StandardScaler              │
 │                              │
 │ Categorical Feature          │
 │ → OneHotEncoder               │
 │                              │
 └──────────────────────────────┘
        ↓
 Logistic Regression
        ↓
 Fraud Prediction
        ↓
 0 → Not Fraudulent
 1 → Fraudulent
```

The complete pipeline is trained on the training data and then saved using **Joblib** for use in the Streamlit application.

## 📈 Model Evaluation

The model was evaluated using:

- Precision
- Recall
- F1-score
- Accuracy
- Confusion Matrix
  
### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Not Fraudulent (0) | 1.00 | 0.95 | 0.97 |
| Fraudulent (1) | 0.02 | 0.93 | 0.04 |

The overall accuracy obtained on the test set was approximately:

94.56%

However, because the dataset is highly imbalanced, accuracy alone should not be considered the main performance indicator.

For fraud detection, recall for the fraudulent class is particularly important because missing an actual fraudulent transaction can be costly.

The fraud class achieved a recall of approximately 0.93 in the test results.

## 🔲 Confusion Matrix

The confusion matrix obtained from the test set was:

```text
[[1802592, 103759],
 [     169,   2266]]
```

| Actual / Predicted | Not Fraudulent (0) | Fraudulent (1) |
|---|---:|---:|
| **Not Fraudulent (0)** | 1,802,592 | 103,759 |
| **Fraudulent (1)** | 169 | 2,266 |

This represents:

 - True Negatives: 1,802,592
 - False Positives: 103,759
 - False Negatives: 169
 - True Positives: 2,266

The model successfully identified a large proportion of fraudulent transactions, but it also produced a relatively large number of false positives.

## 💾 Model Saving

After training and evaluation, the complete machine learning pipeline was saved using Joblib:

  import joblib

  joblib.dump(pipeline,"fraud_detection_pipeline.pkl")

The saved pipeline contains both the preprocessing steps and the trained Logistic Regression model.

## 🌐 Streamlit Application

The trained pipeline is integrated into a Streamlit application.

The application allows users to enter transaction details including:

 - Transaction Type
 - Amount
 - Old Balance of Origin Account
 - New Balance of Origin Account
 - Old Balance of Destination Account
 - New Balance of Destination Account

The application creates a DataFrame from these inputs and passes it directly to the trained pipeline for prediction.

### Prediction Output

The model returns:

1 → FRAUDULENT
0 → NOT FRAUDULENT

The Streamlit application displays the result using an error message for fraudulent transactions and a success message for non-fraudulent transactions.

## 📂 Project Structure

```text
AI-Financial-Fraud-Detection/
│
├── FraudDetection.ipynb
├── FraudDetectApp.py
├── fraud_detection_pipeline.pkl
├── AIML Dataset.csv
├── requirements.txt
└── README.md
```

## 🚀 Installation
1. Clone the Repository
  git clone <YOUR-GITHUB-REPOSITORY-URL>
3. Navigate to the Project Directory
  cd <YOUR-PROJECT-DIRECTORY>
4. Install Dependencies
  pip install -r requirements.txt

If you do not have a requirements.txt file yet, the main dependencies used by the project are:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- streamlit
- joblib
  
## ▶️ Run the Streamlit Application

Run:

 streamlit run FraudDetectApp.py

The Streamlit application will open in your browser.

## 🖥️ Application Workflow

```text
User enters transaction details
            ↓
Streamlit collects the input
            ↓
Input data is converted into DataFrame
            ↓
Saved ML Pipeline receives the input
            ↓
Data preprocessing is applied
            ↓
Logistic Regression model makes prediction
            ↓
       Prediction
        ↙       ↘
       0         1
       ↓         ↓
Not Fraudulent  Fraudulent
       ↓         ↓
Success Message Error Message
```
 
## 📌 Results & Key Insights

The analysis revealed that fraudulent transactions form only a very small proportion of the complete dataset.

The machine learning model achieved approximately 94.56% test accuracy and a 0.93 recall for the fraudulent class.

The use of class_weight="balanced" was important because of the severe class imbalance.

The analysis also focused on transaction types such as TRANSFER and CASH_OUT when examining fraud patterns.

## ⚠️ Limitations

- The dataset is highly imbalanced.
- The model generates a considerable number of false positives.
- Logistic Regression is a relatively simple classification model.
- The system should not be treated as a production financial fraud prevention system without additional validation.
- The model's predictions depend on the patterns present in the training dataset.
  
## 🚀 Future Improvements

Possible improvements include:

- Experimenting with advanced models such as Random Forest, XGBoost, or other ensemble methods.
- Hyperparameter tuning.
- More extensive feature engineering.
- Threshold optimization for fraud detection.
- Handling class imbalance using additional techniques.
- Adding prediction probabilities to the Streamlit application.
- Adding interactive EDA dashboards.
- Deploying the Streamlit application online.
- Adding model monitoring and periodic retraining.
- Improving false-positive reduction.
 
## 🏁 Conclusion

This project demonstrates an end-to-end machine learning workflow for financial fraud detection, starting from exploratory data analysis and feature engineering to model training, evaluation, model serialization, and deployment through Streamlit.

The final system provides an interactive interface where transaction information can be entered and classified as potentially fraudulent or non-fraudulent using the trained machine learning pipeline.

The project demonstrates practical skills in:

- Data Analysis
- Data Visualization
- Feature Engineering
- Data Preprocessing
- Machine Learning
- Model Evaluation
- Model Deployment
- Streamlit Application Development
  
## 👩‍💻 Author

Vanshika Singh Kalakoti

B.Tech Computer Science

## ⭐ If you found this project useful

Feel free to explore the repository, review the notebook, and experiment with the fraud detection application.          

