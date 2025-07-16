import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, ParameterSampler, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from joblib import dump
from tqdm import tqdm

# Rutas de los datasets unificados
DATA_PATH         = 'dataset_stats.csv'
ELO_PATH          = 'dataset_elo.csv'
MODEL_OUTPUT_PATH = 'xgboost_model.joblib'

# --------------------------------------------------
# Funciones del pipeline adaptadas del original
# --------------------------------------------------

def load_data(path):
    df = pd.read_csv(path)
    if 'year' not in df.columns:
        if 'tourney_date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
            except (ValueError, TypeError):
                df['date'] = pd.to_datetime(df['tourney_date'], errors='coerce')
            df['year'] = df['date'].dt.year
            df['year'] = df['year'].fillna(2022)
        else:
            raise KeyError("No se encontró columna 'year' ni 'tourney_date'.")
    return df


def preprocess(df):
    bins  = [0, 32, 64, 96, 128, np.inf]
    labels= ['32', '64', '96', '128', 'GrandSlams+']
    df['draw_cat'] = pd.cut(df['draw_size'], bins=bins, labels=labels)
    df = pd.get_dummies(df, columns=['draw_cat'], prefix='draw')

    base_features = [
        'diff_rank','diff_age','diff_height','grand_slam',
        'h2h','h2h_surface','diff_elo','diff_elo_surface',
        'diff_1stWon','diff_2ndWon','diff_bp_ratio',
        'diff_surface_win_pct','diff_avg_games_per_set','diff_recent_matches'
    ]
    draw_cols = [c for c in df.columns if c.startswith('draw_')]
    features  = base_features + draw_cols

    df = df.dropna(subset=features + ['year']).copy()
    return df, features


def split_temporal(df):
    return df[df['year'] < 2023].copy(), df[df['year'] >= 2023].copy()


def make_symmetric(df_part, features):
    df_inv = df_part.copy()
    # Invertir diffs y head-to-head para data augmentation
    cols_to_flip = [f for f in features if f.startswith('diff_') or f in ['h2h','h2h_surface']]
    for col in cols_to_flip:
        df_inv[col] = -df_inv[col]
    df_part['y'] = 1
    df_inv['y'] = 0
    return pd.concat([df_part, df_inv], ignore_index=True)


def random_search_manual(X, y, base_params, param_dist, n_iter, cv):
    sampler     = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))
    best_score  = -np.inf
    best_params = None
    for params in tqdm(sampler, desc='Buscando hiperparámetros'):
        all_params = base_params.copy()
        all_params.update(params)
        model = XGBClassifier(**all_params)
        scores= cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        if scores.mean() > best_score:
            best_score  = scores.mean()
            best_params = params
    return best_params, best_score


def determine_optimal_rounds(X, y, best_params):
    dtrain = xgb.DMatrix(X, label=y)
    cv_params = {k: v for k, v in best_params.items() if k != 'n_estimators'}
    cv_params.update({'objective':'binary:logistic','eval_metric':'auc'})
    cv_results = xgb.cv(
        params=cv_params,
        dtrain=dtrain,
        num_boost_round=500,
        nfold=5,
        metrics='auc',
        early_stopping_rounds=20,
        seed=42,
        as_pandas=True,
        verbose_eval=False
    )
    return len(cv_results)


def train_final_model(X, y, best_params, optimal_rounds, base_params):
    params = base_params.copy()
    params.update(best_params)
    params.pop('n_estimators', None)
    params.pop('eval_metric', None)
    params.pop('verbosity', None)
    params.pop('n_jobs', None)
    model = XGBClassifier(
        **params,
        n_estimators=optimal_rounds,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


def evaluate_model(model, X, y):
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:,1]
    print("\n=== Reporte de clasificación ===")
    print(classification_report(y, y_pred))
    print("\n=== Matriz de confusión ===")
    print(confusion_matrix(y, y_pred))
    print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
    print(f"AUC-ROC:  {roc_auc_score(y, y_proba):.4f}\n")

if __name__ == '__main__':
    df_raw = load_data(DATA_PATH)
    df, features = preprocess(df_raw)
    df_train, df_test = split_temporal(df)
    df_train_final = make_symmetric(df_train, features)
    df_test_final  = make_symmetric(df_test, features)
    X_train, y_train = df_train_final[features], df_train_final['y']
    X_test,  y_test  = df_test_final[features],  df_test_final['y']

    base_params = {
        'objective':'binary:logistic'
    }
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2, 0.5],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [1, 1.5, 2]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_params, best_score = random_search_manual(X_train, y_train, base_params, param_dist, n_iter=50, cv=cv)
    optimal_rounds = determine_optimal_rounds(X_train, y_train, best_params)
    model_final = train_final_model(X_train, y_train, best_params, optimal_rounds, base_params)
    dump(model_final, MODEL_OUTPUT_PATH)
    print(f"Modelo guardado en {MODEL_OUTPUT_PATH}")
