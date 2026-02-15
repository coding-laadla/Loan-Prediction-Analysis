import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/content/loan_prediction.csv")

df.info()
df.isnull().sum()

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Numerical → Median
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Categorical → Most Frequent
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

label_encoder = LabelEncoder()

for col in cat_cols:
    df[col] = label_encoder.fit_transform(df[col])

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y.value_counts(normalize=True)

df.head()
# Apply SMOTE only on training data
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print("Before SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())



undersampler = RandomUnderSampler(random_state=42)

X_train_under, y_train_under = undersampler.fit_resample(X_train_scaled, y_train)

print("Before Undersampling:")
print(y_train.value_counts())

print("\nAfter Undersampling:")
print(pd.Series(y_train_under).value_counts())

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

def evaluate_models(X_train, y_train, X_test, y_test, dataset_name):
    
    results = []
    
    print(f"\n===== Results using {dataset_name} =====\n")
    
    for name, model in models.items():
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        report = classification_report(y_test, y_pred, output_dict=True)
        
        precision = report['1']['precision']
        recall = report['1']['recall']
        f1 = report['1']['f1-score']
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results.append([name, precision, recall, f1, roc_auc])
        
        print(f"{name}")
        print(classification_report(y_test, y_pred))
        print("ROC-AUC:", roc_auc)
        print("-"*50)
        
    return pd.DataFrame(results, columns=["Model", "Precision", "Recall", "F1-Score", "ROC-AUC"])

results_original = evaluate_models(
    X_train_scaled, y_train,
    X_test_scaled, y_test,
    "Original Data"
)

results_smote = evaluate_models(
    X_train_smote, y_train_smote,
    X_test_scaled, y_test,
    "SMOTE Data"
)

results_original["Dataset"] = "Original"
results_smote["Dataset"] = "SMOTE"

final_results = pd.concat([results_original, results_smote])

final_results 

plt.figure(figsize=(12,6))
sns.barplot(data=final_results, x="Model", y="F1-Score", hue="Dataset")
plt.title("Model Comparison (F1-Score)")
plt.show()

plt.figure(figsize=(8,6))

for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=name)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (SMOTE Data)")
plt.legend()
plt.show()
