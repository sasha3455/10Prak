from ml_core import *
import pandas as pd, matplotlib.pyplot as plt, os
from wordcloud import WordCloud

df = init_db('bmw_sales.csv')
table, models, roc = train_compare_models(df, quick=True)
clustered, clusters = cluster_reviews(df, sample=200)
prod = product_stats(df)
forecast = forecast_reviews(df)

print('Модели:'); print(table[['Model','Before_F1','After_F1','BestParams']])
print('\nТоп товаров по доле позитива:'); print(prod.head())
print('\n5 частых слов в негативных отзывах:', top_negative_words(df,5))

fig, axes = plt.subplots(2,3, figsize=(16,10))
fig.suptitle('student_report: BMW reviews ML product', fontsize=16)
axes[0,0].hist(df.rating, bins=5, edgecolor='black'); axes[0,0].set_title('Распределение оценок')
daily=df.groupby(pd.to_datetime(df.date).dt.date).size(); axes[0,1].plot(pd.to_datetime(daily.index), daily.values); axes[0,1].set_title('Динамика отзывов')
axes[0,2].bar(prod.head(8)['product'], prod.head(8)['positive_share']); axes[0,2].tick_params(axis='x', rotation=45); axes[0,2].set_title('Топ товаров по позитиву')
axes[1,0].axis('off'); axes[1,0].table(cellText=table[['Model','Before_F1','After_F1']].round(3).values, colLabels=['Model','Before F1','After F1'], loc='center')
axes[1,1].scatter(clustered.x, clustered.y, c=clustered.kmeans_cluster); axes[1,1].set_title('Кластеры PCA/KMeans')
wc=WordCloud(width=600,height=350,background_color='white').generate(' '.join(df.text)); axes[1,2].imshow(wc); axes[1,2].axis('off'); axes[1,2].set_title('Облако слов')
plt.tight_layout(); plt.savefig('student_report.png', dpi=160, bbox_inches='tight')
print('Создан student_report.png')
