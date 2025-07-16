import pandas as pd
import numpy as np
from joblib import load
import shap
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

# -------------------------
# Configuración de rutas unificadas
# -------------------------
MODEL_PATH      = 'xgboost_model.joblib'
ELO_PATH        = 'dataset_elo.csv'
MATCHES_PATH    = 'dataset_stats.csv'
PLAYERS_FILE    = 'jugadores_unicos.txt'

# -------------------------
# Cargar modelo y datos
# -------------------------
model = load(MODEL_PATH)

df_elo = pd.read_csv(ELO_PATH)
df_elo['jugador'] = df_elo['jugador'].str.lower()

df_matches = pd.read_csv(MATCHES_PATH)
for col in ['winner_name', 'loser_name', 'surface']:
    if col in df_matches.columns:
        df_matches[col] = df_matches[col].str.lower()

# Calcular medias para variables diff
diff_feats = [
    'diff_rank','diff_age','diff_height',
    'diff_1stWon','diff_2ndWon','diff_bp_ratio',
    'diff_surface_win_pct','diff_avg_games_per_set','diff_recent_matches'
]
mean_features = {feat: df_matches[feat].mean() if feat in df_matches.columns else 0.0 for feat in diff_feats}

# -------------------------
# Cargar lista de jugadores únicos para autocompletado
# -------------------------
with open(PLAYERS_FILE, 'r', encoding='utf-8') as f:
    jugadores_unicos = [line.strip() for line in f if line.strip()]
# Set de nombres en minúscula para validación
jugadores_set = set([j.lower() for j in jugadores_unicos])
player_completer = WordCompleter(
    jugadores_unicos,
    ignore_case=True,
    sentence=True,
    match_middle=True
)

# Completer para superficies
surface_completer = WordCompleter(['hard','clay','grass'], ignore_case=True)

# -------------------------
# Función H2H
# -------------------------
def head_to_head(j1, j2, surf=None):
    d = df_matches
    mask = (((d['winner_name'] == j1) & (d['loser_name'] == j2)) |
            ((d['winner_name'] == j2) & (d['loser_name'] == j1)))
    d2 = d[mask]
    if surf:
        d2 = d2[d2['surface'] == surf]
    if d2.empty:
        return 0
    return int((d2['winner_name'] == j1).sum() - (d2['winner_name'] == j2).sum())

# -------------------------
# Construcción de features
# -------------------------
def build_features(j1, j2, surf, draw_size):
    j1l, j2l = j1.lower(), j2.lower()
    feats = {}
    # Stats diffs o medias
    for feat in ['rank','age','height','1stWon','2ndWon','bp_ratio',
                 'surface_win_pct','avg_games_per_set','recent_matches']:
        feats[f'diff_{feat}'] = mean_features.get(f'diff_{feat}', 0.0)
    # Grand slam flag
    feats['grand_slam'] = 1.0 if draw_size == 128 else 0.0
    # Head-to-head
    feats['h2h'] = float(head_to_head(j1l, j2l))
    feats['h2h_surface'] = float(head_to_head(j1l, j2l, surf))
    # ELO diff
    e1 = df_elo[df_elo['jugador'] == j1l]
    e2 = df_elo[df_elo['jugador'] == j2l]
    if not e1.empty and not e2.empty and 'elo_actual_total' in df_elo.columns:
        feats['diff_elo'] = float(e1['elo_actual_total'].iloc[-1] - e2['elo_actual_total'].iloc[-1])
    else:
        feats['diff_elo'] = 0.0
    # ELO surface diff
    surf_col = f'elo_actual_{surf}'
    if not e1.empty and not e2.empty and surf_col in df_elo.columns:
        feats['diff_elo_surface'] = float(e1[surf_col].iloc[-1] - e2[surf_col].iloc[-1])
    else:
        feats['diff_elo_surface'] = 0.0
    # Tamaño de cuadro numérico
    feats['draw_size'] = float(draw_size)
    # Draw one-hot
    categories = ['32','64','96','128','GrandSlams+']
    for cat in categories:
        feats[f'draw_{cat}'] = 1.0 if str(draw_size) == cat else 0.0
    # Definir orden de columnas según el modelo
    cols_order = [
        'diff_rank','diff_age','diff_height','grand_slam',
        'h2h','h2h_surface','diff_elo','diff_elo_surface',
        'diff_1stWon','diff_2ndWon','diff_bp_ratio',
        'diff_surface_win_pct','diff_avg_games_per_set','diff_recent_matches',
        'draw_size'
    ] + [f'draw_{c}' for c in categories]
    X = pd.DataFrame([feats], columns=cols_order)
    return X

# -------------------------
# Predicción e interfaz
# -------------------------
def predecir_partido():
    print("🎾 Predicción de Partido Futuro (Unificado) 🎾")
    # Leer y validar Jugador 1
    while True:
        j1 = prompt("Jugador 1: ", completer=player_completer).strip()
        if j1.lower() in jugadores_set:
            break
        print(f"❌ Jugador '{j1}' no encontrado. Intenta de nuevo.")
    # Leer y validar Jugador 2
    while True:
        j2 = prompt("Jugador 2: ", completer=player_completer).strip()
        if j2.lower() in jugadores_set:
            break
        print(f"❌ Jugador '{j2}' no encontrado. Intenta de nuevo.")
    # Fijar orden consistente: orden alfabético
    if j1.lower() > j2.lower():
        j1, j2 = j2, j1
    surf = prompt("Superficie (hard/clay/grass): ", completer=surface_completer).strip().lower()
    ds = int(prompt("Tamaño de cuadro (número): ").strip())
    X = build_features(j1, j2, surf, ds)
    prob1 = model.predict_proba(X)[0,1]
    prob2 = 1 - prob1
    print("\n--- Resultado ---")
    print(f"{j1} tiene {prob1*100:.2f}% de probabilidad de ganar")
    print(f"{j2} tiene {prob2*100:.2f}% de probabilidad de ganar")
    # SHAP
    expl = shap.Explainer(model)
    sv = expl(X)
    print("\nContribuciones SHAP:")
    for feat,val in zip(X.columns, sv.values[0]):
        print(f" {feat}: {val:.4f}")
    shap.plots.bar(sv[0], show=True)

if __name__ == '__main__':
    predecir_partido()
