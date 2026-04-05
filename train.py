import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, ParameterSampler, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from joblib import dump
from tqdm import tqdm

DATA_PATH         = 'dataset_stats.csv'
ELO_PATH          = 'dataset_elo.csv'
MODEL_OUTPUT_PATH = 'xgboost_model.joblib'

# Bins compartidos con predict.py — NO cambiar sin actualizar ambos lados
DRAW_BINS   = [0, 32, 64, 96, 128, np.inf]
DRAW_LABELS = ['32', '64', '96', '128', 'GrandSlams+']


def load_data(path):
    df = pd.read_csv(path, parse_dates=['tourney_date'], low_memory=False)
    df['year']  = df['tourney_date'].dt.year.astype(int)
    df['month'] = df['tourney_date'].dt.month.astype(int)
    return df


def preprocess(df):
    df['draw_size'] = pd.to_numeric(df['draw_size'], errors='coerce')
    df['draw_cat']  = pd.cut(df['draw_size'], bins=DRAW_BINS, labels=DRAW_LABELS)

    # Surface one-hot: fit sobre datos reales para capturar categorías exactas
    surf_ohe  = OneHotEncoder(sparse_output=False, drop='first')
    surf_feat = surf_ohe.fit_transform(df[['surface']])
    surf_cols = [f"surf_{c}" for c in surf_ohe.categories_[0][1:]]
    df[surf_cols] = surf_feat

    draw_dummies = pd.get_dummies(df['draw_cat'], prefix='draw')
    df = pd.concat([df, draw_dummies], axis=1)

    features = [
        'diff_rank', 'diff_age', 'diff_height',
        'diff_elo', 'diff_elo_surface',
        'diff_1stWon', 'diff_2ndWon', 'diff_bp_ratio',
        'h2h', 'h2h_surface',
        'diff_surface_win_pct', 'diff_avg_games_per_set', 'diff_recent_matches',
        'month',
    ] + surf_cols + list(draw_dummies.columns)

    df = df.dropna(subset=features + ['year']).copy()
    return df, features, surf_ohe


def split_temporal(df, split_year=2023):
    return df[df['year'] < split_year].copy(), df[df['year'] >= split_year].copy()


def augment_symmetry(df, features):
    inv  = df.copy()
    flip = [c for c in features if c.startswith('diff_') or c in ['h2h', 'h2h_surface']]
    inv[flip] = -inv[flip]
    df['y']  = 1
    inv['y'] = 0
    return pd.concat([df, inv], ignore_index=True)


def random_search(X, y, base_params, param_dist, n_iter, cv):
    sampler    = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))
    best_score = -np.inf
    best_params = None
    for p in tqdm(sampler, desc='HParam search'):
        params = {**base_params, **p}
        score  = cross_val_score(XGBClassifier(**params), X, y,
                                 cv=cv, scoring='roc_auc', n_jobs=-1).mean()
        if score > best_score:
            best_score, best_params = score, p
    return best_params, best_score


def find_optimal_rounds(X, y, params):
    dtrain = xgb.DMatrix(X, label=y)
    cvp    = {k: v for k, v in params.items() if k != 'n_estimators'}
    cvp.update(objective='binary:logistic', eval_metric='auc')
    results = xgb.cv(cvp, dtrain, num_boost_round=500, nfold=5,
                     early_stopping_rounds=20, seed=42,
                     as_pandas=True, verbose_eval=False)
    return len(results)


def train_final(X, y, best_params, rounds, base_params):
    params = {**base_params, **best_params}
    for k in ['n_estimators', 'eval_metric', 'verbosity', 'n_jobs']:
        params.pop(k, None)
    model = XGBClassifier(**params, n_estimators=rounds, n_jobs=-1)
    model.fit(X, y)
    return model


def evaluate(model, X, y):
    y_pred  = model.predict(X)
    y_score = model.predict_proba(X)[:, 1]
    print('Classification report')
    print(classification_report(y, y_pred))
    print('Confusion matrix')
    print(confusion_matrix(y, y_pred))
    print(f"Accuracy: {accuracy_score(y, y_pred):.4f}, AUC: {roc_auc_score(y, y_score):.4f}")


if __name__ == '__main__':
    df = load_data(DATA_PATH)
    df, features, surf_ohe = preprocess(df)
    df_tr, df_te = split_temporal(df)
    df_tr_aug = augment_symmetry(df_tr, features)
    df_te_aug = augment_symmetry(df_te, features)

    X_tr, y_tr = df_tr_aug[features], df_tr_aug['y']
    X_te, y_te = df_te_aug[features],  df_te_aug['y']

    base = {'objective': 'binary:logistic'}
    dist = {
        'n_estimators':     [100, 200, 300],
        'max_depth':        [3, 4, 6],
        'learning_rate':    [0.05, 0.1],
        'subsample':        [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3],
        'gamma':            [0, 0.1],
        'reg_alpha':        [0, 0.1],
        'reg_lambda':       [1, 2],
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    best_p, best_s = random_search(X_tr, y_tr, base, dist, n_iter=30, cv=cv)
    print(f"Mejor AUC CV: {best_s:.4f} | params: {best_p}")

    rounds      = find_optimal_rounds(X_tr, y_tr, best_p)
    final_model = train_final(X_tr, y_tr, best_p, rounds, base)
    evaluate(final_model, X_te, y_te)

    # Guardar artefacto completo: modelo + metadata necesaria para predict.py
    artifact = {
        'model':           final_model,
        'features':        features,
        'surf_categories': surf_ohe.categories_[0].tolist(),  # e.g. ['Clay','Grass','Hard']
        'draw_bins':       DRAW_BINS,
        'draw_labels':     DRAW_LABELS,
    }
    dump(artifact, MODEL_OUTPUT_PATH)
    print(f"Modelo guardado en {MODEL_OUTPUT_PATH} ({len(features)} features)")
