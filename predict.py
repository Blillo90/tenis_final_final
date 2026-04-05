import pandas as pd
import numpy as np
from joblib import load
import shap
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

MODEL_PATH   = 'xgboost_model.joblib'
ELO_PATH     = 'dataset_elo.csv'
MATCHES_PATH = 'dataset_stats.csv'
PLAYERS_FILE = 'jugadores_unicos.txt'

# Mapeo superficie (entrada usuario en minúscula) → columna ELO en español
SURF_ELO_MAP = {
    'hard':  'elo_actual_dura',
    'clay':  'elo_actual_tierra',
    'grass': 'elo_actual_hierba',
}

# -------------------------
# Cargar artefacto del modelo
# -------------------------
artifact = load(MODEL_PATH)
if not isinstance(artifact, dict):
    raise RuntimeError(
        "El modelo cargado no es compatible. "
        "Por favor, re-entrena con train.py para generar el artefacto actualizado."
    )

model          = artifact['model']
FEATURES       = artifact['features']
SURF_CATS      = artifact['surf_categories']   # e.g. ['Clay', 'Grass', 'Hard']
DRAW_BINS      = artifact['draw_bins']
DRAW_LABELS    = artifact['draw_labels']

# -------------------------
# Cargar datos
# -------------------------
df_elo = pd.read_csv(ELO_PATH, encoding='latin1')
df_elo['jugador'] = df_elo['jugador'].str.lower().str.strip()

df_matches = pd.read_csv(MATCHES_PATH, parse_dates=['tourney_date'], low_memory=False)
for col in ['winner_name', 'loser_name', 'surface']:
    if col in df_matches.columns:
        df_matches[col] = df_matches[col].str.lower().str.strip()
# Ordenar cronológicamente para que iloc[-1] dé el partido más reciente
df_matches = df_matches.sort_values('tourney_date').reset_index(drop=True)

with open(PLAYERS_FILE, 'r', encoding='utf-8') as f:
    jugadores_unicos = [line.strip() for line in f if line.strip()]
jugadores_set     = {j.lower() for j in jugadores_unicos}
player_completer  = WordCompleter(jugadores_unicos, ignore_case=True, match_middle=True)
surface_completer = WordCompleter(['hard', 'clay', 'grass'], ignore_case=True)


# -------------------------
# Helpers para construir features reales del jugador
# -------------------------

def _player_rows(player: str) -> pd.DataFrame:
    """Todos los partidos de un jugador (ya ordenados por fecha)."""
    return df_matches[
        (df_matches['winner_name'] == player) |
        (df_matches['loser_name']  == player)
    ]


def head_to_head(j1: str, j2: str, surf: str = None) -> int:
    d    = df_matches
    mask = (((d['winner_name'] == j1) & (d['loser_name'] == j2)) |
            ((d['winner_name'] == j2) & (d['loser_name'] == j1)))
    d2   = d[mask]
    if surf:
        d2 = d2[d2['surface'] == surf]
    if d2.empty:
        return 0
    return int((d2['winner_name'] == j1).sum() - (d2['winner_name'] == j2).sum())


def _safe_diff(a, b) -> float:
    """Resta dos valores numéricos; devuelve 0.0 si alguno es NaN/None."""
    try:
        fa, fb = float(a), float(b)
        return 0.0 if (np.isnan(fa) or np.isnan(fb)) else fa - fb
    except (TypeError, ValueError):
        return 0.0


def get_basic_stats(player: str) -> dict:
    """Rank, edad y altura del partido más reciente del jugador."""
    rows = _player_rows(player)
    if rows.empty:
        return {}
    last = rows.iloc[-1]
    if last['winner_name'] == player:
        return {
            'rank':   pd.to_numeric(last.get('winner_rank'), errors='coerce'),
            'age':    pd.to_numeric(last.get('winner_age'),  errors='coerce'),
            'height': pd.to_numeric(last.get('winner_ht'),   errors='coerce'),
        }
    return {
        'rank':   pd.to_numeric(last.get('loser_rank'), errors='coerce'),
        'age':    pd.to_numeric(last.get('loser_age'),  errors='coerce'),
        'height': pd.to_numeric(last.get('loser_ht'),   errors='coerce'),
    }


def get_serve_stats(player: str, n: int = 20) -> dict:
    """Media de estadísticas de saque en los últimos n partidos."""
    rows = _player_rows(player).tail(n)
    first_won, second_won, bp_ratios = [], [], []
    for _, r in rows.iterrows():
        if r['winner_name'] == player:
            first_won.append(r.get('w_1stWon'))
            second_won.append(r.get('w_2ndWon'))
            bpf = r.get('w_bpFaced') or 0
            bps = r.get('w_bpSaved') or 0
        else:
            first_won.append(r.get('l_1stWon'))
            second_won.append(r.get('l_2ndWon'))
            bpf = r.get('l_bpFaced') or 0
            bps = r.get('l_bpSaved') or 0
        bp_ratios.append(bps / bpf if bpf > 0 else np.nan)

    def _mean(lst):
        vals = [float(v) for v in lst if pd.notna(v)]
        return float(np.mean(vals)) if vals else 0.0

    return {
        '1stWon':   _mean(first_won),
        '2ndWon':   _mean(second_won),
        'bp_ratio': _mean(bp_ratios),
    }


def get_surface_win_pct(player: str, surf: str) -> float:
    """Porcentaje de victorias en la superficie dada (histórico completo)."""
    rows      = _player_rows(player)
    surf_rows = rows[rows['surface'] == surf]
    if surf_rows.empty:
        return 0.5
    wins = (surf_rows['winner_name'] == player).sum()
    return float(wins / len(surf_rows))


def get_avg_games_per_set(player: str, n: int = 30) -> float:
    """Media de juegos por set (columna avg_w/avg_l pre-calculada)."""
    rows = _player_rows(player).tail(n)
    vals = []
    for _, r in rows.iterrows():
        col = 'avg_w' if r['winner_name'] == player else 'avg_l'
        v = r.get(col)
        if pd.notna(v):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else 0.0


def get_recent_match_count(player: str) -> int:
    """Partidos jugados en los últimos 365 días."""
    rows   = _player_rows(player)
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=365)
    return len(rows[rows['tourney_date'] >= cutoff])


def draw_size_to_cat(ds: int) -> str:
    """Convierte draw_size numérico a categoría usando los mismos bins que train.py."""
    for lo, hi, label in zip(DRAW_BINS[:-1], DRAW_BINS[1:], DRAW_LABELS):
        if lo < ds <= hi:
            return label
    return DRAW_LABELS[-1]


# -------------------------
# Construcción del vector de features
# -------------------------

def build_features(j1: str, j2: str, surf: str, draw_size: int) -> pd.DataFrame:
    j1l, j2l = j1.lower(), j2.lower()
    feats = {}

    # --- Rank, edad, altura (stats reales del último partido) ---
    s1 = get_basic_stats(j1l)
    s2 = get_basic_stats(j2l)
    feats['diff_rank']   = _safe_diff(s1.get('rank'),   s2.get('rank'))
    feats['diff_age']    = _safe_diff(s1.get('age'),    s2.get('age'))
    feats['diff_height'] = _safe_diff(s1.get('height'), s2.get('height'))

    # --- ELO (desde dataset_elo.csv con nombres de columna en español) ---
    e1 = df_elo[df_elo['jugador'] == j1l]
    e2 = df_elo[df_elo['jugador'] == j2l]
    if not e1.empty and not e2.empty:
        feats['diff_elo'] = float(
            e1['elo_actual_total'].iloc[0] - e2['elo_actual_total'].iloc[0]
        )
        surf_col = SURF_ELO_MAP.get(surf, '')
        feats['diff_elo_surface'] = (
            float(e1[surf_col].iloc[0] - e2[surf_col].iloc[0])
            if surf_col and surf_col in df_elo.columns else 0.0
        )
    else:
        feats['diff_elo'] = feats['diff_elo_surface'] = 0.0

    # --- Estadísticas de saque (media de últimos 20 partidos) ---
    srv1 = get_serve_stats(j1l)
    srv2 = get_serve_stats(j2l)
    feats['diff_1stWon']   = srv1['1stWon']   - srv2['1stWon']
    feats['diff_2ndWon']   = srv1['2ndWon']   - srv2['2ndWon']
    feats['diff_bp_ratio'] = srv1['bp_ratio'] - srv2['bp_ratio']

    # --- Head-to-head ---
    feats['h2h']         = float(head_to_head(j1l, j2l))
    feats['h2h_surface'] = float(head_to_head(j1l, j2l, surf))

    # --- Stats agregadas de historial ---
    feats['diff_surface_win_pct']    = get_surface_win_pct(j1l, surf) - get_surface_win_pct(j2l, surf)
    feats['diff_avg_games_per_set']  = get_avg_games_per_set(j1l) - get_avg_games_per_set(j2l)
    feats['diff_recent_matches']     = get_recent_match_count(j1l) - get_recent_match_count(j2l)

    # --- Variables temporales ---
    feats['month'] = pd.Timestamp.today().month

    # --- Surface one-hot (mismas categorías que en entrenamiento) ---
    # SURF_CATS[0] es la categoría eliminada (drop='first'); las demás se codifican
    for cat in SURF_CATS[1:]:
        feats[f'surf_{cat}'] = 1.0 if surf == cat.lower() else 0.0

    # --- Draw size one-hot (mismos bins que pd.cut en entrenamiento) ---
    draw_cat = draw_size_to_cat(draw_size)
    for label in DRAW_LABELS:
        feats[f'draw_{label}'] = 1.0 if draw_cat == label else 0.0

    # Construir DataFrame con el orden exacto de features del modelo
    df_pred = pd.DataFrame([feats], columns=FEATURES)
    df_pred = df_pred.fillna(0.0)
    return df_pred


# -------------------------
# Interfaz de predicción
# -------------------------

def predecir_partido():
    print("=== Prediccion de Partido ===")
    while True:
        j1 = prompt("Jugador 1: ", completer=player_completer).strip()
        if j1.lower() in jugadores_set:
            break
        print(f"'{j1}' no encontrado. Prueba con el autocompletado.")
    while True:
        j2 = prompt("Jugador 2: ", completer=player_completer).strip()
        if j2.lower() in jugadores_set and j2.lower() != j1.lower():
            break
        print(f"'{j2}' invalido o igual a jugador 1.")

    # Orden canónico: alfabético para evitar ambigüedad de dirección
    if j1.lower() > j2.lower():
        j1, j2 = j2, j1

    surf = prompt("Superficie (hard/clay/grass): ", completer=surface_completer).strip().lower()
    if surf not in SURF_ELO_MAP:
        print(f"Superficie '{surf}' no reconocida. Usando 'hard'.")
        surf = 'hard'

    ds = int(prompt("Draw size (32, 64, 96, 128; >128 para GrandSlams+): ").strip())

    X  = build_features(j1, j2, surf, ds)
    p1 = model.predict_proba(X)[0, 1]
    p2 = 1.0 - p1

    print(f"\n--- Resultado ---")
    print(f"{j1}: {p1 * 100:.2f}%")
    print(f"{j2}: {p2 * 100:.2f}%")

    # SHAP con TreeExplainer (eficiente para XGBoost)
    explainer = shap.TreeExplainer(model)
    sv        = explainer.shap_values(X)
    contributions = sorted(zip(FEATURES, sv[0]), key=lambda x: abs(x[1]), reverse=True)
    print("\nContribuciones SHAP (top 10):")
    for feat, val in contributions[:10]:
        print(f"  {feat:35s} {val:+.4f}")


if __name__ == '__main__':
    predecir_partido()
