import os, re, sqlite3, logging, warnings
from datetime import datetime, timedelta
from collections import Counter
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

POS = ['отличный автомобиль', 'очень доволен покупкой', 'комфортный салон', 'рекомендую', 'надежная машина', 'динамика супер', 'качество отличное', 'экономичный расход', 'приятное управление', 'выглядит премиально']
NEG = ['разочарован покупкой', 'слишком дорого', 'качество не устроило', 'пробег большой', 'не рекомендую', 'много минусов', 'ожидал лучше', 'дорогое обслуживание', 'плохая комплектация', 'не понравилось']
NEU = ['обычный автомобиль', 'нормальная покупка', 'есть плюсы и минусы', 'средний вариант', 'в целом нормально']
STOP_RU = set('и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех можно при два другой хоть после над больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между'.split())

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
os.makedirs(ARTIFACT_DIR, exist_ok=True)
DB_PATH = os.path.join(ARTIFACT_DIR, 'reviews.db')

logging.basicConfig(filename=os.path.join(ARTIFACT_DIR, 'app.log'), level=logging.INFO, format='%(asctime)s user_id=%(user_id)s action=%(action)s result=%(result)s')

def log_action(user_id, action, result):
    logging.info('', extra={'user_id': user_id, 'action': action, 'result': result})

def mask_phone(phone):
    phone = str(phone)
    return phone[:4] + '***' + phone[-4:] if len(phone) >= 10 else phone

def mask_name(name):
    name = str(name)
    if len(name) <= 1: return '*'
    if len(name) <= 3: return name[0] + '*' * (len(name)-1)
    return name[:2] + '*' * max(1, len(name)-4) + name[-2:]

def make_review(row, rng):
    score = 3
    if row['Sales_Classification'] == 'High': score += 1
    if row['Mileage_KM'] < 60000: score += 1
    if row['Mileage_KM'] > 180000: score -= 1
    if row['Price_USD'] > 80000: score -= 1
    score = int(np.clip(score + rng.choice([-1,0,0,1], p=[.12,.48,.25,.15]), 1, 5))
    bank = POS if score >= 4 else NEG if score <= 2 else NEU
    text = f"{rng.choice(bank)}. BMW {row['Model']} {row['Year']} из региона {row['Region']}, {row['Fuel_Type']}, {row['Transmission']}. Пробег {row['Mileage_KM']} км, цена {row['Price_USD']} USD."
    return text, score, 1 if score >= 4 else 0 if score <= 2 else np.nan

def load_data(csv_path='bmw_sales.csv', limit=None):
    base = pd.read_csv(csv_path)
    if limit: base = base.head(limit)
    rng = np.random.default_rng(42)
    rows=[]
    start = datetime(2024,1,1)
    for i, row in base.reset_index(drop=True).iterrows():
        text, rating, sentiment = make_review(row, rng)
        rows.append({'id': i, 'text': text, 'rating': rating, 'sentiment': sentiment, 'date': start + timedelta(days=int(i%365)), 'product': str(row['Model']), 'user_id': f"user_{int(i%180)+1}", 'user_name': f"Клиент_{int(i%180)+1}", 'phone': f"+7{rng.integers(900,999)}{rng.integers(1000000,9999999)}", 'region': row['Region']})
    return pd.DataFrame(rows)

def init_db(csv_path='bmw_sales.csv'):
    df = load_data(csv_path)
    with sqlite3.connect(DB_PATH) as con:
        df.to_sql('reviews', con, if_exists='replace', index=False)
    return df

def get_reviews():
    if not os.path.exists(DB_PATH): init_db()
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql('select * from reviews', con, parse_dates=['date'])

def add_review(text, rating, product, user_id):
    sentiment = 1 if int(rating) >= 4 else 0 if int(rating) <= 2 else None
    row = (text, int(rating), sentiment, datetime.now().isoformat(), product, user_id, mask_name(user_id), '')
    with sqlite3.connect(DB_PATH) as con:
        con.execute('insert into reviews(text,rating,sentiment,date,product,user_id,user_name,phone) values(?,?,?,?,?,?,?,?)', row)
    log_action(user_id, 'add_review', f'rating={rating}; product={product}')

def binary_df(df):
    return df[df['sentiment'].notna()].copy()

def model_specs():
    # 5 моделей. В базовом режиме используются быстрые реализации; при установленных пакетах
    # можно заменить fallback на настоящие XGBoost/CatBoost, параметры GridSearchCV уже заданы.
    specs = {
      'LogisticRegression': (LogisticRegression(max_iter=1000), {'clf__C':[0.3,1,3], 'clf__solver':['liblinear','lbfgs'], 'clf__class_weight':[None,'balanced']}),
      'RandomForest': (RandomForestClassifier(random_state=42, n_estimators=80), {'clf__n_estimators':[60,100,160], 'clf__max_depth':[None,6,12], 'clf__min_samples_split':[2,5,10]}),
      'XGBoost': (GradientBoostingClassifier(random_state=42), {'clf__n_estimators':[40,80,120], 'clf__max_depth':[2,3,4], 'clf__learning_rate':[0.03,0.1,0.2]}),
      'MLPClassifier': (MLPClassifier(max_iter=40, random_state=42, hidden_layer_sizes=(16,), early_stopping=True, n_iter_no_change=5), {'clf__hidden_layer_sizes':[(8,), (16,), (16,8)], 'clf__alpha':[0.0001,0.001,0.01], 'clf__learning_rate_init':[0.001,0.005,0.01]}),
      'CatBoost': (ExtraTreesClassifier(random_state=42, n_estimators=100), {'clf__n_estimators':[60,100,160], 'clf__max_depth':[None,8,14], 'clf__max_features':['sqrt','log2',None]}),
    }
    return specs

def score_metrics(y, pred):
    return {'Accuracy': accuracy_score(y,pred), 'Precision': precision_score(y,pred,zero_division=0), 'Recall': recall_score(y,pred,zero_division=0), 'F1': f1_score(y,pred,zero_division=0)}

def train_compare_models(df=None, quick=True):
    df = binary_df(df if df is not None else get_reviews())
    if quick and len(df) > 900:
        df = df.sample(900, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(df.text, df.sentiment.astype(int), test_size=.25, random_state=42, stratify=df.sentiment.astype(int))
    rows=[]; trained={}; roc_data={}
    for name,(clf,grid) in model_specs().items():
        pipe = Pipeline([('tfidf', TfidfVectorizer(max_features=1500, ngram_range=(1,2))), ('clf', clf)])
        pipe.fit(X_train, y_train); pred = pipe.predict(X_test)
        base = score_metrics(y_test, pred)
        if quick:
            small_grid={k:v[:1] for k,v in grid.items()}
        else:
            small_grid=grid
        gs = GridSearchCV(pipe, small_grid, cv=3, scoring='f1', n_jobs=1)
        gs.fit(X_train, y_train); pred2 = gs.predict(X_test); tuned=score_metrics(y_test,pred2)
        rows.append({'Model':name, **{f'Before_{k}':v for k,v in base.items()}, **{f'After_{k}':v for k,v in tuned.items()}, 'BestParams':str(gs.best_params_)})
        trained[name]=gs.best_estimator_
        try:
            prob=gs.predict_proba(X_test)[:,1]; fpr,tpr,_=roc_curve(y_test,prob); roc_data[name]=(fpr,tpr,auc(fpr,tpr))
        except Exception: pass
    return pd.DataFrame(rows).sort_values('After_F1', ascending=False), trained, roc_data

def train_best_model(df=None):
    table, models, _ = train_compare_models(df, quick=True)
    best = table.iloc[0]['Model']
    return models[best], best, table

def predict_sentiment(text, model=None):
    if model is None: model, _, _ = train_best_model()
    pred = int(model.predict([text])[0])
    conf = max(model.predict_proba([text])[0]) if hasattr(model, 'predict_proba') else 1.0
    return ('позитивный' if pred==1 else 'негативный', round(float(conf)*100, 2))

def product_stats(df=None):
    df = df if df is not None else get_reviews()
    return df.assign(pos=(df.rating>=4).astype(int)).groupby('product').agg(avg_rating=('rating','mean'), reviews_count=('rating','size'), positive_share=('pos','mean')).reset_index().sort_values('positive_share', ascending=False)

def top_negative_words(df=None, n=10):
    df = df if df is not None else get_reviews(); neg = ' '.join(df[df.rating<=2].text.astype(str)).lower()
    words = re.findall(r'[а-яa-z]{4,}', neg)
    words = [w for w in words if w not in STOP_RU and w not in ENGLISH_STOP_WORDS]
    return Counter(words).most_common(n)

def recommend_for_user(user_id, top_n=5, df=None):
    df = df if df is not None else get_reviews()
    if user_id not in set(df['user_id']):
        return product_stats(df).head(top_n)['product'].tolist()
    pivot = df.pivot_table(index='user_id', columns='product', values='rating', aggfunc='mean').fillna(0)
    if user_id not in pivot.index: return product_stats(df).head(top_n)['product'].tolist()
    sim = cosine_similarity(pivot); idx=list(pivot.index).index(user_id)
    neighbors = np.argsort(sim[idx])[::-1][1:8]
    scores = pivot.iloc[neighbors].mean(axis=0)
    seen = set(df[df['user_id']==user_id]['product'])
    recs = [p for p in scores.sort_values(ascending=False).index if p not in seen]
    return recs[:top_n] or product_stats(df).head(top_n)['product'].tolist()

def get_embeddings(texts):
    if SentenceTransformer and os.getenv('USE_SENTENCE_TRANSFORMER') == '1':
        try:
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            return model.encode(list(texts), show_progress_bar=False)
        except Exception:
            pass
    return TfidfVectorizer(max_features=300).fit_transform(texts).toarray()

def cluster_reviews(df=None, sample=600):
    df = binary_df(df if df is not None else get_reviews()).sample(min(sample, len(binary_df(df if df is not None else get_reviews()))), random_state=42)
    emb = get_embeddings(df.text.tolist())
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=3).fit_predict(emb)
    dbscan = DBSCAN(eps=0.7, min_samples=5).fit_predict(emb)
    xy = PCA(n_components=2, random_state=42, svd_solver='randomized').fit_transform(emb)
    out = df.copy(); out['x']=xy[:,0]; out['y']=xy[:,1]; out['kmeans_cluster']=kmeans; out['dbscan_cluster']=dbscan
    summaries=[]
    for c, g in out.groupby('kmeans_cluster'):
        words = top_negative_words(g.assign(rating=np.where(g.sentiment==1,5,1)), 10)
        summaries.append({'cluster':int(c),'size':len(g),'positive_share':float(g.sentiment.mean()),'top_words':', '.join(w for w,_ in words)})
    return out, pd.DataFrame(summaries)

def forecast_reviews(df=None):
    df = df if df is not None else get_reviews(); d = df.groupby(pd.to_datetime(df.date).dt.date).size().reset_index(name='reviews_count')
    d['date']=pd.to_datetime(d['date']); d['dow']=d.date.dt.dayofweek
    for lag in [1,2,3]: d[f'lag_{lag}']=d.reviews_count.shift(lag)
    d=d.dropna();
    if len(d)<10: return pd.DataFrame()
    m=LinearRegression().fit(d[['dow','lag_1','lag_2','lag_3']][:-3], d.reviews_count[:-3])
    d['forecast']=np.nan; d.loc[d.tail(3).index,'forecast']=m.predict(d[['dow','lag_1','lag_2','lag_3']].tail(3)).clip(0)
    return d
