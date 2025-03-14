import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from sklearn.metrics import accuracy_score, roc_auc_score

# Step 1: Load and Preprocess Data
df = pd.read_csv("/content/kidney_disease.csv")

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(exclude=np.number).columns

# Handling missing values separately for numerical and categorical features
# For numerical features, use mean imputation
num_imputer = SimpleImputer(strategy='mean')
df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

# For categorical features, use most frequent imputation
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# Normalize numerical features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Encode categorical features
# Assuming 'category_column' is a categorical column
# If you have other categorical columns, include them in the list
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') #sparse=False to get a numpy array
categorical_features = encoder.fit_transform(df[['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']])
# Assuming 'category_column' is a categorical column; replace with your actual categorical column name
# Removing the original categorical columns and adding the encoded features
encoded_feature_names = encoder.get_feature_names_out(['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'])

encoded_features_df = pd.DataFrame(categorical_features, columns=encoded_feature_names, index=df.index)
df = df.drop(columns=['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'])  # Replace 'category_column' with your actual categorical column name
df = pd.concat([df, encoded_features_df], axis=1)

# Ensure all columns are numeric before converting to numpy array
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, replace errors with NaN
        except ValueError:
            print(f"Column '{col}' could not be converted to numeric.")
            # Consider further handling for columns that cannot be converted

df = df.fillna(0)  # Replace NaN with 0 to ensure numeric dtype

df = df.values  # Convert dataframe to numpy array

# Create time-series sequences
sequence_length = 10
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

data_sequences = create_sequences(df, sequence_length)
labels = df[sequence_length:, -1]  # Assuming last column is the target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data_sequences, labels, test_size=0.2, random_state=42)

# Step 2: Build and Compare Models

# LSTM Model
lstm_model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(sequence_length, X_train.shape[2])),
    BatchNormalization(),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# GRU Model
gru_model = Sequential([
    GRU(128, return_sequences=True, input_shape=(sequence_length, X_train.shape[2])),
    BatchNormalization(),
    GRU(64, return_sequences=True),
    Dropout(0.3),
    GRU(32, return_sequences=False),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 3: Train Models
lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
gru_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
rf_model.fit(X_train[:, -1, :], y_train)  # Flattened input for Random Forest

# Step 4: Evaluate Models
def evaluate_model(model, X_test, y_test, model_type):
    if model_type == "rf":
        y_pred = model.predict(X_test[:, -1, :])
    else:
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)
    print(f"{model_type.upper()} Model - Accuracy: {accuracy:.2f}, AUC-ROC: {auc_roc:.2f}")

# Evaluate all models
evaluate_model(lstm_model, X_test, y_test, "lstm")
evaluate_model(gru_model, X_test, y_test, "gru")
evaluate_model(rf_model, X_test, y_test, "rf")

# Step 5: Choosing the Best Model
# Based on the evaluation, GRU is preferred if it performs better in AUC-ROC, otherwise LSTM is chosen
# Random Forest is used as a baseline traditional model