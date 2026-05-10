import os, uuid
import pandas as pd
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from ml_core import *

st.set_page_config(page_title='BMW Reviews ML Product', layout='wide')
if 'user_id' not in st.session_state:
    st.session_state.user_id = 'web_' + uuid.uuid4().hex[:10]
if not os.path.exists(DB_PATH): init_db('bmw_sales.csv')

df = get_reviews()
st.sidebar.title('ML-продукт отзывов')
st.sidebar.write('Ваш user_id:', st.session_state.user_id)
page = st.sidebar.radio('Раздел', ['Главная', 'Статистика товаров', 'Сравнение моделей', 'Кластеры', 'Логи администратора'])

if page == 'Главная':
    st.title('Анализ отзывов клиентов интернет-магазина BMW')
    model, best_name, metrics = train_best_model(df)
    c1,c2 = st.columns([1,1])
    with c1:
        st.subheader('Оставить отзыв')
        product = st.selectbox('Товар', sorted(df['product'].unique()))
        rating = st.slider('Оценка', 1, 5, 5)
        text = st.text_area('Текст отзыва', 'Отличный автомобиль, комфортный салон, рекомендую')
        sent, conf = predict_sentiment(text, model)
        st.info(f'Предсказанная тональность: {sent}, уверенность {conf}%')
        if st.button('Сохранить отзыв'):
            add_review(text, rating, product, st.session_state.user_id)
            st.success('Отзыв сохранён в SQLite')
            st.rerun()
    with c2:
        st.subheader('Персональные рекомендации')
        for p in recommend_for_user(st.session_state.user_id, 5, df): st.write('•', p)
        st.subheader('Лучшие товары по доле позитива')
        st.dataframe(product_stats(df).head(10), use_container_width=True)
    st.subheader('Графики')
    c3,c4 = st.columns(2)
    c3.plotly_chart(px.histogram(df, x='rating', nbins=5, title='Распределение оценок'), use_container_width=True)
    daily = df.groupby(pd.to_datetime(df.date).dt.date).size().reset_index(name='reviews')
    c4.plotly_chart(px.line(daily, x='date', y='reviews', title='Динамика отзывов по дням'), use_container_width=True)
    text_all=' '.join(df.text.astype(str))
    wc = WordCloud(width=900, height=350, background_color='white').generate(text_all)
    fig, ax = plt.subplots(figsize=(12,4)); ax.imshow(wc); ax.axis('off'); st.pyplot(fig)

elif page == 'Статистика товаров':
    st.title('Статистика по товарам')
    st.dataframe(product_stats(df), use_container_width=True)
    st.write('5 самых частых слов в негативных отзывах:', top_negative_words(df, 5))

elif page == 'Сравнение моделей':
    st.title('Сравнение 5 моделей и подбор гиперпараметров')
    full = st.checkbox('Запустить полный GridSearchCV (дольше)', value=False)
    table, models, roc_data = train_compare_models(df, quick=not full)
    st.dataframe(table, use_container_width=True)
    best = table.iloc[0]
    st.success(f"Лучшая модель по F1 после подбора: {best['Model']} ({best['After_F1']:.3f})")
    rows=[]
    for name,(fpr,tpr,aucv) in roc_data.items():
        rows += [{'model':name,'fpr':x,'tpr':y,'auc':aucv} for x,y in zip(fpr,tpr)]
    if rows: st.plotly_chart(px.line(pd.DataFrame(rows), x='fpr', y='tpr', color='model', title='ROC-кривые'), use_container_width=True)

elif page == 'Кластеры':
    st.title('Кластеризация отзывов: SentenceTransformer + KMeans/DBSCAN')
    clustered, summary = cluster_reviews(df)
    st.dataframe(summary, use_container_width=True)
    fig = px.scatter(clustered, x='x', y='y', color='kmeans_cluster', hover_data=['text','product','rating'], title='PCA-визуализация кластеров KMeans')
    st.plotly_chart(fig, use_container_width=True)
    choice = st.selectbox('Показать примеры кластера', sorted(clustered.kmeans_cluster.unique()))
    st.dataframe(clustered[clustered.kmeans_cluster==choice][['product','rating','text']].head(15), use_container_width=True)

else:
    st.title('Последние 50 записей app.log')
    log_path=os.path.join(ARTIFACT_DIR,'app.log')
    if os.path.exists(log_path): st.code(''.join(open(log_path, encoding='utf-8').readlines()[-50:]))
    else: st.warning('Лог пока пуст')
