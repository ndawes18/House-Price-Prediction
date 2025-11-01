"""House Price Prediction (toy demo)
Run: python house_price_pipeline.py
"""
import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

def create_sample_df():
    data = [
        [1,8450,7,2003,1710,"CollgCr",208500],
        [2,9600,6,1976,1262,"Veenker",181500],
        [3,11250,7,2001,1786,"CollgCr",223500],
        [4,9550,7,1915,1717,"NAmes",140000],
        [5,14260,8,2000,2198,"CollgCr",250000],
    ]
    cols = ["Id","LotArea","OverallQual","YearBuilt","GrLivArea","Neighborhood","SalePrice"]
    return pd.DataFrame(data, columns=cols)

def prepare_features(df):
    df = df.copy()
    df['house_age'] = 2025 - df['YearBuilt']
    X = df.drop(columns=['Id','SalePrice'])
    y = df['SalePrice']
    return X, y

def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object','string']).columns.tolist()
    num_pipe = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    cat_pipe = Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([('num', num_pipe, numeric_cols), ('cat', cat_pipe, categorical_cols)], remainder='drop', sparse_threshold=0)
    return preprocessor

def train_models(X_train, y_train, X_test, y_test):
    models = {
        'Linear': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, verbosity=0)
    }
    results = {}
    for name, model in models.items():
        pre = build_preprocessor(X_train)
        pipe = Pipeline([('pre', pre), ('model', model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)
        results[name] = {'rmse': rmse, 'r2': r2, 'pipeline': pipe}
    return results

def main():
    os.makedirs('models', exist_ok=True)
    df = create_sample_df()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = train_models(X_train, y_train, X_test, y_test)
    best = min(results.items(), key=lambda kv: kv[1]['rmse'])
    best_name, best_info = best
    print('Best model:', best_name, 'metrics:', {'rmse': best_info['rmse'], 'r2': best_info['r2']})
    joblib.dump(best_info['pipeline'], 'models/house_price_pipeline.pkl')
    print('Saved pipeline to models/house_price_pipeline.pkl')

if __name__ == '__main__':
    main()
