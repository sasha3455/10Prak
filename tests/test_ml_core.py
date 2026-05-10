import os, sys, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_core import mask_phone, mask_name, load_data, predict_sentiment, score_metrics, recommend_for_user
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def test_mask_phone():
    assert mask_phone('+79001234567').startswith('+790') and mask_phone('+79001234567').endswith('4567')

def test_mask_name():
    assert '*' in mask_name('Клиент_123')

def test_load_data_fields():
    df = load_data(os.path.join(os.path.dirname(__file__), '..', 'bmw_sales.csv'), limit=250)
    assert len(df) >= 200
    assert {'text','rating','product','date','user_id'}.issubset(df.columns)

def test_metrics():
    m = score_metrics([1,0,1,0], [1,0,0,0])
    assert set(['Accuracy','Precision','Recall','F1']).issubset(m.keys())

def test_predict_sentiment():
    model = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])
    model.fit(['отличный автомобиль рекомендую', 'плохой автомобиль не рекомендую'], [1, 0])
    label, conf = predict_sentiment('отличный автомобиль, рекомендую', model)
    assert label in ['позитивный','негативный'] and 0 <= conf <= 100

def test_recommendations():
    df = load_data(os.path.join(os.path.dirname(__file__), '..', 'bmw_sales.csv'), limit=300)
    recs = recommend_for_user(df.user_id.iloc[0], 3, df)
    assert isinstance(recs, list) and len(recs) > 0
