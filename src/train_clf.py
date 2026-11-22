"""Этап 4: классификация премиум-сегмента аренды (price >= 300000). Запуск: python src/train_clf.py"""
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
params = json.loads((ROOT / "params_clf.json").read_text(encoding="utf-8"))

df = pd.read_csv(ROOT / "data" / "astana_rent_clean.csv")

# Бинарная цель: премиум-сегмент (аренда от 300 000 тг/мес)
y = (df["price"] >= 300000).astype(int)

# Исключаем price и price_m2, чтобы не было утечки данных
num_cols = ["rooms", "area", "floor", "floors", "year"]
cat_cols = ["district", "furniture", "floor_type", "age_group"]
X = df[num_cols + cat_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params["test_size"],
    random_state=params["random_state"],
    stratify=y
)

dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
dummy_acc = accuracy_score(y_test, dummy.predict(X_test))

prep = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
    (make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat_cols)
)

classifiers = {
    "logreg": LogisticRegression(C=params["C"], class_weight=params["class_weight"], max_iter=1000),
    "tree": DecisionTreeClassifier(
        max_depth=4,
        class_weight=params["class_weight"],
        random_state=params["random_state"]
    ),
}

model = make_pipeline(prep, classifiers[params["model"]])

# Кросс-валидация с учётом порога из params_clf.json
cv_proba = cross_val_predict(model, X_train, y_train, cv=5, method="predict_proba")[:, 1]
cv_pred = (cv_proba >= params["threshold"]).astype(int)
cv_f1 = f1_score(y_train, cv_pred)
cv_auc = roc_auc_score(y_train, cv_proba)

model.fit(X_train, y_train)
test_proba = model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= params["threshold"]).astype(int)

metrics = {
    "dummy_accuracy": round(float(dummy_acc), 4),
    "cv_f1": round(float(cv_f1), 4),
    "cv_roc_auc": round(float(cv_auc), 4),
    "test_accuracy": round(float(accuracy_score(y_test, test_pred)), 4),
    "test_precision": round(float(precision_score(y_test, test_pred, zero_division=0)), 4),
    "test_recall": round(float(recall_score(y_test, test_pred, zero_division=0)), 4),
    "test_f1": round(float(f1_score(y_test, test_pred, zero_division=0)), 4),
}

(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(model, ROOT / "models" / "model_clf.joblib")
(ROOT / "metrics_clf.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
print(params["model"], metrics)