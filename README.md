# ML-продукт анализа отзывов BMW на оценку 5

Проект сделан на базе лекционного `main.py`, но адаптирован под собственный датасет `bmw_sales.csv`. Так как исходный CSV содержит продажи BMW, а не тексты отзывов, модуль `ml_core.py` генерирует датасет отзывов с обязательными полями: `text`, `rating`, `product`, `date`, `user_id`.

## Запуск
```bash
pip install -r requirements.txt
python run_analysis.py
streamlit run app.py
pytest -q
```

## Что реализовано
- Классификация тональности: LogisticRegression, RandomForest, XGBoost, MLPClassifier, CatBoost/LightGBM/fallback.
- GridSearchCV по 3 параметрам для каждой модели, сравнение метрик до/после подбора.
- `predict_sentiment()` возвращает позитивный/негативный и процент уверенности.
- SQLite-хранилище отзывов.
- Веб-интерфейс Streamlit: ввод отзыва, предсказание тональности, рекомендации, графики, статистика товаров.
- User-based collaborative filtering по `user_id`.
- SentenceTransformer для эмбеддингов, при отсутствии пакета fallback на TF-IDF.
- KMeans и DBSCAN, PCA/Plotly визуализация, описание кластеров.
- Логирование запросов в `artifacts/app.log`.
- Unit-тесты pytest.
- `run_analysis.py` создаёт `student_report.png`.
