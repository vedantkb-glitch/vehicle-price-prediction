import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
data = pd.read_csv("/content/Car_Price_Prediction.csv")

# Separate input and output
X = data.drop("Price", axis=1)
y = data["Price"]

# Categorical and numerical columns
categorical_cols = ["Make", "Model", "Fuel Type", "Transmission"]
numerical_cols = ["Year", "Engine Size", "Mileage"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# Model
model = RandomForestRegressor(n_estimators=200, random_state=42)

# Pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
pipeline.fit(X_train, y_train)

# Save model
pickle.dump(pipeline, open("vehicle_price_model.pkl", "wb"))

print("✅ Model trained and saved successfully")