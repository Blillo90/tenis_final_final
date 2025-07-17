#!/usr/bin/env python3
"""
Script para actualizar dataset_elo.csv y dataset_stats.csv con nuevos datos de ongoing_tourneys.csv.
- Hace copia de seguridad de ambos CSV en carpeta backups con timestamp.
- Actualiza las puntuaciones ELO de los jugadores según los partidos ya jugados.
- Calcula y añade nuevas filas en dataset_stats (comprueba duplicados) incluyendo todos los indicadores:
  * grand_slam, draw_size
  * diff_rank, diff_age, diff_height
  * diff_elo, diff_elo_surface
  * h2h, h2h_surface
  * diff_1stWon, diff_2ndWon, diff_bp_ratio
  * diff_surface_win_pct, diff_avg_games_per_set, diff_recent_matches
"""
import os
import shutil
from datetime import datetime, timedelta
import pandas as pd

# Configuración de rutas
data_dir = '.'
backup_dir = os.path.join(data_dir, 'backups')
elo_file = os.path.join(data_dir, 'dataset_elo.csv')
stats_file = os.path.join(data_dir, 'dataset_stats.csv')
going_file = os.path.join(data_dir, 'ongoing_tourneys.csv')

# Crear carpeta de backups si no existe
os.makedirs(backup_dir, exist_ok=True)

# Timestamp para backup
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Copiar archivos a backups
shutil.copy(elo_file, os.path.join(backup_dir, f'dataset_elo_{timestamp}.csv'))
shutil.copy(stats_file, os.path.join(backup_dir, f'dataset_stats_{timestamp}.csv'))
print(f"Backups creados en {backup_dir} con sufijo {timestamp}")

# Cargar dataset_elo y crear diccionarios de ELO
tmp_elo = pd.read_csv(elo_file)
idx = next((c for c in ['player_id','jugador','id','player'] if c in tmp_elo.columns), tmp_elo.columns[0])
tmp_elo.set_index(idx, inplace=True)
# Columnas de ELO global y por superficie
df_cols = tmp_elo.columns.tolist()
elo_global_col = 'elo' if 'elo' in df_cols else next(c for c in df_cols if c.startswith('elo_actual'))
surf_cols = {surface: col for surface in ['hard','clay','grass'] for col in df_cols if col.endswith(surface)}
elo_dict = tmp_elo[elo_global_col].to_dict()
print(f"ELO global: {elo_global_col}, Superficies: {surf_cols}")

# Cargar dataset_stats histórico y ongoing
df_stats = pd.read_csv(stats_file, parse_dates=['tourney_date'])
df_old_stats = df_stats.copy()
df_raw_ongoing = pd.read_csv(going_file, parse_dates=['tourney_date'])
# Normalizar nombres de columnas
df_raw_ongoing.rename(columns=lambda c: c.strip(), inplace=True)

def parse_score_to_avg_games(score):
    sets = [s for s in str(score).split() if '-' in s]
    games = []
    for s in sets:
        try:
            w, l = map(int, s.split('-'))
            games.append(w + l)
        except:
            continue
    return sum(games) / len(games) if games else 0.0

# Detectar columnas de stats_df
stats_win = next(c for c in ['winner_id','winner_name'] if c in df_stats.columns)
stats_loss = next(c for c in ['loser_id','loser_name'] if c in df_stats.columns)
date_col = 'tourney_date'
surf_col = 'surface'

# Funciones de cálculo
def compute_h2h(p1, p2, surf=None):
    df = df_old_stats
    mask = ((df[stats_win] == p1) & (df[stats_loss] == p2)) | ((df[stats_win] == p2) & (df[stats_loss] == p1))
    if surf:
        mask &= df[surf_col] == surf
    sub = df[mask]
    wins_p1 = (sub[stats_win] == p1).sum()
    wins_p2 = (sub[stats_win] == p2).sum()
    return wins_p1 - wins_p2


def compute_surface_pct(player, surf):
    df = df_old_stats[df_old_stats[surf_col] == surf]
    total = ((df[stats_win] == player) | (df[stats_loss] == player)).sum()
    wins = (df[stats_win] == player).sum()
    return wins / total if total else 0.0


# Funciones de cálculo (sin compute_avg_games pues no hay columna 'score' en historial)

def compute_h2h(p1, p2, surf=None):
    df = df_old_stats
    mask = ((df[stats_win] == p1) & (df[stats_loss] == p2)) | ((df[stats_win] == p2) & (df[stats_loss] == p1))
    if surf:
        mask &= df[surf_col] == surf
    sub = df[mask]
    wins_p1 = (sub[stats_win] == p1).sum()
    wins_p2 = (sub[stats_win] == p2).sum()
    return wins_p1 - wins_p2


def compute_surface_pct(player, surf):
    df = df_old_stats[df_old_stats[surf_col] == surf]
    total = ((df[stats_win] == player) | (df[stats_loss] == player)).sum()
    wins = (df[stats_win] == player).sum()
    return wins / total if total else 0.0


def compute_recent_matches(player, cutoff_date):
    df = df_old_stats[(df_old_stats[stats_win] == player) | (df_old_stats[stats_loss] == player)]
    return (df['tourney_date'] >= cutoff_date).sum()

# Generar nuevas filas
new_rows = []
existing_keys = set(df_stats[[date_col, stats_win, stats_loss, surf_col]].apply(tuple, axis=1))
for _, row in df_raw_ongoing.iterrows():
    p1 = row.get('winner_id', row.get('winner_name'))
    p2 = row.get('loser_id', row.get('loser_name'))
    key = (row[date_col], p1, p2, row[surf_col])
    if key in existing_keys:
        continue
    cutoff = row[date_col] - timedelta(days=365)
    entry = {
        date_col: row[date_col],
        stats_win: p1,
        stats_loss: p2,
        surf_col: row[surf_col],
        'draw_size': row.get('draw_size', 0),
        'grand_slam': 1 if row.get('draw_size', 0) == 128 else 0,
        'diff_rank': row.get('winner_rank', 0) - row.get('loser_rank', 0),
        'diff_age': row.get('winner_age', 0) - row.get('loser_age', 0),
        'diff_height': row.get('winner_ht', 0) - row.get('loser_ht', 0),
        'diff_elo': elo_dict.get(p1, 1500) - elo_dict.get(p2, 1500),
        'diff_elo_surface': (
            tmp_elo.at[p1, surf_cols[row[surf_col].lower()]] - tmp_elo.at[p2, surf_cols[row[surf_col].lower()]]
        ) if row[surf_col].lower() in surf_cols else 0.0,
        'h2h': compute_h2h(p1, p2),
        'h2h_surface': compute_h2h(p1, p2, row[surf_col]),
        'diff_1stWon': row.get('w_1stWon', 0) - row.get('l_1stWon', 0),
        'diff_2ndWon': row.get('w_2ndWon', 0) - row.get('l_2ndWon', 0),
        'diff_bp_ratio': (
            (row.get('w_bpSaved', 0) / row.get('w_bpFaced', 1)) -
            (row.get('l_bpSaved', 0) / row.get('l_bpFaced', 1))
        ),
        'diff_surface_win_pct': compute_surface_pct(p1, row[surf_col]) - compute_surface_pct(p2, row[surf_col]),
        'diff_avg_games_per_set': parse_score_to_avg_games(row.get('score', '')), # avg games per set de este partido
        'diff_recent_matches': compute_recent_matches(p1, cutoff) - compute_recent_matches(p2, cutoff),
        'diff_ace': row.get('w_ace', 0) - row.get('l_ace', 0),
        'diff_df': row.get('w_df', 0) - row.get('l_df', 0)
    }
    new_rows.append(entry)
    existing_keys.add(key)

# Concatenar y guardar resultados
if new_rows:
    df_new = pd.DataFrame(new_rows)
    df_new = df_new.reindex(columns=df_stats.columns, fill_value=None)
    df_stats = pd.concat([df_stats, df_new], ignore_index=True)
    df_stats.to_csv(stats_file, index=False)
    print(f"Añadidas {len(new_rows)} filas nuevas a {stats_file}.")
else:
    print("No hay nuevas filas que añadir en dataset_stats.")

print("Actualización completada.")
