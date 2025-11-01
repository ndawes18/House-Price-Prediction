import os, joblib
from house_price_pipeline import create_sample_df, prepare_features, main

def test_create_sample_df():
    df = create_sample_df()
    assert 'SalePrice' in df.columns

def test_prepare_features():
    df = create_sample_df()
    X, y = prepare_features(df)
    assert 'house_age' in X.columns

def test_main_creates_model(tmp_path):
    out = tmp_path / 'model.pkl'
    model_file = main(out_model_path=str(out))
    assert os.path.exists(model_file)
    mdl = joblib.load(model_file)
    assert hasattr(mdl, 'predict')
