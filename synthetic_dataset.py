import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import joblib                                                                                                
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


np.random.seed(87)
random.seed(87)


n_samples = 1000000
fraud_ratio= 0.02
fraud_sample = int(n_samples * fraud_ratio)
legit_sample = n_samples-fraud_sample


customer_ids = [f"Cust_{i:05d}" for i in range(1000)]


def generate_transactions ( n, fraud = False):
    data = []
    for _ in range (n):
        customer_id = random.choice(customer_ids)
        transaction_time = datetime(2025,2,1)+timedelta(minutes = random.randint(0,60*24*90))
        transaction_amount =  round(np.random.exponential(200),2)
        transaction_type = np.random.choice(["POS","ATM","ONLINE","BRANCH"], p=[0.4,0.2,0.3,.1])
        device_type = np.random.choice(["Mobile","Desktop","POS-Terminal"],p=[0.5,.3,.2])
        location_distance_km = np.random.exponential(2) if not fraud else np.random.exponential(20)
        is_foreign_transaction = int(np.random.rand() <(0.1 if not fraud else 0.5))
        is_high_risk_country = int(np.random.rand()<(0.1 if not fraud else 0.5))
        is_weekend = int(transaction_time.weekday()>=5)
        hour_of_day =  transaction_time.hour
        previous_fraud_flag = int(np.random.rand() < (0.01 if not fraud else 0.2))
        is_fraud = int(fraud)

        data.append([customer_id, transaction_time, transaction_amount, transaction_type, device_type,location_distance_km,is_foreign_transaction,
                     is_high_risk_country,is_weekend,hour_of_day,previous_fraud_flag,is_fraud])
        
    return data


fraud_data = generate_transactions(fraud_sample, True)
legit_data = generate_transactions(legit_sample, False)

all_data = fraud_data+legit_data


random.shuffle(all_data)

columns = ["customer_id","transaction_time","transaction_amount","transaction_type","device_type","location_distance_km","is_foreign_transaction",
           "is_high_risk_country","is_weekend","hour_of_day","previous_fraud_flag","is_fraud"]

df = pd.DataFrame(all_data, columns=columns)

df["transaction_id"] = [f"TXN{i:07d}"for i in range(len(df))]
df = df[["transaction_id"]+columns]
df.to_csv(r"datasets\synthetic_fraud_data.csv", index = False)
df_data = pd.read_csv(r"datasets\synthetic_fraud_data.csv", parse_dates=["transaction_time"])
df_data["day_of_week"] = df_data["transaction_time"].dt.day_name()
df_data["month"]=df_data["transaction_time"].dt.month
df_data["hour"] = df_data["transaction_time"].dt.hour


plt.figure(figsize=(10,5))
sns.histplot(df_data, x = "transaction_amount", hue="is_fraud", bins = 20, kde=True, stat = "density")
plt.xlim(0,2000)
sns.countplot(df_data, x="transaction_type", hue="is_fraud")
sns.countplot(df_data, x= "device_type", hue="is_fraud")
plt.figure(figsize=(10,5))
sns.boxplot(data=df_data, x="is_fraud", y="location_distance_km")
plt.yscale('log')
plt.title("Location Distance by Fraud Flag")
plt.show()


fraud_by_hour = df_data.groupby("hour")["is_fraud"].mean()
fraud_by_hour.plot(kind="bar", figsize=(10,4), title="Fraud Rate by Hour of Day")
plt.ylabel("Fraud Rate")
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(12,4))
sns.countplot(data=df_data, x="is_foreign_transaction", hue="is_fraud", ax=ax[0])
ax[0].set_title("Foreign Transactions")
sns.countplot(data=df_data, x="is_high_risk_country", hue="is_fraud", ax=ax[1])
ax[1].set_title("High-Risk Country Transactions")
plt.tight_layout()
plt.show()



plt.figure(figsize=(10,6))
numeric_cols = df_data.select_dtypes(include=['float64', 'int64']).drop(columns=["is_fraud"])
sns.heatmap(df_data[numeric_cols.columns.tolist() + ["is_fraud"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()



df = df_data
df["transaction_time"] = pd.to_datetime(df["transaction_time"])

# Sort by customer and time
df = df.sort_values(["customer_id", "transaction_time"])

# Average transaction amount per customer (rolling)
df["avg_transaction_amount_user"] = df.groupby("customer_id")["transaction_amount"].transform(lambda x: x.rolling(10, min_periods=1).mean())

# Large transaction flag
df["is_large_transaction"] = (df["transaction_amount"] > 3 * df["avg_transaction_amount_user"]).astype(int)

# Count of transactions in last 24 hours





# Time since last transaction (in seconds)
df["time_since_last_txn"] = (
    df.groupby("customer_id")["transaction_time"]
    .diff().dt.total_seconds().fillna(999999)
)

# Night transaction
df["is_night"] = df["hour_of_day"].apply(lambda x: 1 if x < 6 else 0)

# Location anomaly (sudden jump > 30km)
df["location_change_flag"] = df.groupby("customer_id")["location_distance_km"].diff().abs().fillna(0)
df["location_change_flag"] = (df["location_change_flag"] > 30).astype(int)

# Frequency of each device type used by a customer
device_freq = df.groupby(["customer_id", "device_type"])["transaction_id"].count().reset_index()
device_freq.rename(columns={"transaction_id": "device_usage_count"}, inplace=True)
df = df.merge(device_freq, on=["customer_id", "device_type"], how="left")



model_features = [
    "transaction_amount", "transaction_type", "device_type", "location_distance_km",
    "is_foreign_transaction", "is_high_risk_country", "is_weekend", "hour_of_day",
    "previous_fraud_flag", "avg_transaction_amount_user", "is_large_transaction",
    "time_since_last_txn", "is_night", "location_change_flag","device_usage_count"
]



# Define feature list and target
model_features = [
    "transaction_amount", "transaction_type", "device_type", "location_distance_km",
    "is_foreign_transaction", "is_high_risk_country", "is_weekend", "hour_of_day",
    "previous_fraud_flag", "avg_transaction_amount_user", "is_large_transaction",
    "time_since_last_txn", "is_night", "location_change_flag", "device_usage_count"
]
target_col = "is_fraud"

# Copy working dataset
df_model = df[model_features + [target_col]].copy()

# Encode categorical columns
categorical_cols = ["transaction_type", "device_type"]
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    encoders[col] = le  # Save encoders for future inverse_transform

# Fill any remaining NaNs
df_model = df_model.fillna(0)

X = df_model[model_features]
y = df_model[target_col]


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=87)
X_test.to_csv(r"datasets\X_test.csv", index= False)
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)[:, 1]

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("\nROC AUC Score:", roc_auc_score(y_test, y_proba))

xgb.plot_importance(xgb_model, max_num_features=15, height=0.6)
plt.title("Top Feature Importances - XGBoost")
plt.show()

joblib.dump(xgb_model, r"models\xgb_fraud_model.pkl")

import shap

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

joblib.dump(explainer, r"models\shap_explainer.pkl")

shap.summary_plot(shap_values, X_test, plot_type='bar')

shap.summary_plot(shap_values , X_test)

i=8
shap.force_plot(explainer.expected_value, shap_values[i],X_test.iloc[i],matplotlib=True)

shap.decision_plot(explainer.expected_value, shap_values[i], X_test.iloc[i])

# Dependence plot for 'transaction_amount'
shap.dependence_plot("transaction_amount", shap_values, X_test)


# With interaction feature (e.g., 'is_large_transaction')
shap.dependence_plot("transaction_amount", shap_values, X_test, interaction_index="is_large_transaction")

# Pick an index to explain (e.g., index 10)
index = 10

# Get single instance values
expected_value = explainer.expected_value
instance = X_test.iloc[index]
instance_shap = shap_values[index]

# Waterfall plot
shap.plots._waterfall.waterfall_legacy(expected_value, instance_shap, instance)


import joblib


# Save XGBoost model


# Save SHAP explainer




