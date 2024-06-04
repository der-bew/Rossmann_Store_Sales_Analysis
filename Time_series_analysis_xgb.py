#!/usr/bin/env python

# import libraries
import pandas as pd
import numpy as np
from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pickle
import logging
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb


warnings.filterwarnings("ignore")


# Initialize the logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


def load_data(path):
    try:
        df = pd.read_csv(path, low_memory=False)
        return df
    except Exception as e:
        logger.info(f"Error on Loding Data {e}")


path = "../data/02_intermediate/train_clean.csv"
df = load_data(path)


df.head()


logger.info("Droping unecessary coulmuns")
df = df.drop(["Unnamed: 0", "Date"], axis=1)


# Preprocessing pipeline
def create_preprocessing_pipeline(numeric_features, categorical_features):
    logger.info("Creating preprocessing pipeline")
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


# Train and evaluate the model
def train_evaluate_model(df, target_column):
    logger.info("Starting model training and evaluation")

    # Define features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Define numeric and categorical features
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)

    # Preprocess the features
    logger.info("Preprocessing the features")
    X_preprocessed = preprocessor.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_preprocessed, y, test_size=0.2, random_state=42
    )

    # Create regression matrices
    dtrain_reg = xgb.DMatrix(X_train, y_train)
    dtest_reg = xgb.DMatrix(X_test, y_test)

    # Define hyperparameters
    params = {"objective": "reg:squarederror", "tree_method": "hist"}
    n = 5000

    evals = [(dtest_reg, "validation"), (dtrain_reg, "train")]

    # Train XGBoost model
    logger.info("Training XGBoost model")
    model = xgb.train(
        params=params,
        dtrain=dtrain_reg,
        num_boost_round=n,
        evals=evals,
        verbose_eval=50,
        early_stopping_rounds=50,
    )

    # Make predictions
    logger.info("Making predictions")
    y_pred = model.predict(dtest_reg)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    logger.info(f"Root_Mean Squared Error: {rmse}")

    return model, y_pred, y_test, X


df["Sales"] = df["Sales"].fillna(df["Sales"].mean())
model, y_pred, y_test, X = train_evaluate_model(df, "Sales")


rcParams["figure.figsize"] = 35, 25
xgb.plot_tree(model, rankdir="LR", num_trees=1)


xgb.plot_tree(model, rankdir="LR", num_trees=20)


#  Interpret Feature Importances
feature_importance = model.get_score(importance_type="gain")
sorted_feature_importance = sorted(
    feature_importance.items(), key=lambda x: x[1], reverse=True
)

for feature, importance in sorted_feature_importance[:10]:
    print(f"{feature}: {importance}")


# Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_test.index, residuals)
plt.axhline(y=0, color="r", linestyle="-")
plt.title("Residuals")
plt.show()


# # Serialize The Model

timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
filename = f"xgbmodel_{timestamp}.pkl"
pickle.dump(model, open(filename, "wb"))
