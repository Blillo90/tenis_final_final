#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para generar un dataset de stats históricas correctamente calculadas
Partiendo de los CSV de cada año en ./data (2000.csv, 2001.csv, ...), este script:
 1. Lee y concatena todos los archivos desde 2000.csv
 2. Calcula juegos por set a partir del marcador
 3. Calcula diferencia histórica de avg games per set
 4. Calcula diferencia histórica de win pct por superficie
 5. Calcula diferencia histórica de partidos jugados en los últimos 365 días
 6. Muestra barra de progreso al leer los archivos
y exporta dataset_stats.csv al directorio raíz.
"""
import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm

# 1. Función auxiliar para parsear score y calcular avg games per set
import re
def parse_score_to_avg_games(score_str):
    # score_str ejemplo: '6-3 4-6 7-5'
    if not score_str or score_str.isspace():
        return np.nan
    sets = re.findall(r"(\d+)-(\d+)", score_str)
    if not sets:
        return np.nan
    total_games = sum(int(a) + int(b) for a,b in sets)
    return total_games / len(sets)

# 2. Leer y concatenar todos los CSV desde 2000.csv
csv_files = sorted(glob.glob(os.path.join('data', '*.csv')))
# Filtrar solo años >= 2000.csv
csv_files = [f for f in csv_files if os.path.basename(f).split('.')[0].isdigit() and int(os.path.basename(f).split('.')[0]) >= 2000]

dfs = []
print(f"Leyendo {len(csv_files)} archivos de datos...")
for fp in tqdm(csv_files, desc='Archivos procesados'):
    df = pd.read_csv(fp, parse_dates=['tourney_date'])
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
# Asegurarnos de orden cronológico
df = df.sort_values('tourney_date').reset_index(drop=True)

# 3. Calcular games_per_set
df['games_per_set'] = df['score'].apply(parse_score_to_avg_games)

# 4. diff_avg_games_per_set
# Promedio histórico acumulado (shifted) de gps para ganador y perdedor
df['avg_gps_winner'] = df.groupby('winner_id')['games_per_set'] \
                          .apply(lambda s: s.shift().expanding().mean())
df['avg_gps_loser']  = df.groupby('loser_id')['games_per_set']  \
                          .apply(lambda s: s.shift().expanding().mean())

df['diff_avg_games_per_set'] = df['avg_gps_winner'] - df['avg_gps_loser']

# 5. diff_surface_win_pct
# Cálculo histórico de % victorias por superficie
df['win_flag'] = 1

# Para cada jugador y superficie, acumulado de victorias y totales
surface_w = df.groupby(['winner_id', 'surface', ])['win_flag'] \
               .apply(lambda s: s.shift().cumsum())
surface_t = df.groupby(['winner_id', 'surface'])['win_flag'] \
               .apply(lambda s: s.shift().cumcount() + (s.shift()>=0).astype(int))
# Para oponente: invertimos roles
surface_w2 = df.groupby(['loser_id', 'surface'])['win_flag'] \
                .apply(lambda s: s.shift().cumsum())
surface_t2 = df.groupby(['loser_id', 'surface'])['win_flag'] \
                .apply(lambda s: s.shift().cumcount() + (s.shift()>=0).astype(int))

# Reindexar series al df original
surface_w.index = df.index
surface_t.index = df.index
surface_w2.index = df.index
surface_t2.index = df.index

df['pct_win_winner_s'] = surface_w / surface_t
df['pct_win_loser_s']  = surface_w2 / surface_t2

df['diff_surface_win_pct'] = df['pct_win_winner_s'] - df['pct_win_loser_s']

# 6. diff_recent_matches (365 días)
# Generar DataFrame largo (row_id, player_id, tourney_date)
# Duplicamos filas: una para winner y otra para loser
long = pd.DataFrame({
    'row_id': np.repeat(df.index.values, 2),
    'player_id': np.concatenate([df['winner_id'].values, df['loser_id'].values]),
    'tourney_date': np.concatenate([df['tourney_date'].values, df['tourney_date'].values])
})
# Ordenar por fecha e indexar
long = long.sort_values('tourney_date').set_index('tourney_date')
# Contar partidos jugados en ventana móvil de 365D (shifted)
long['recent_count'] = long.groupby('player_id')['player_id'] \
                         .apply(lambda s: s.rolling('365D').count().shift())
# Asignar rol p1/p2 en cada par duplicado
long = long.reset_index()
long['role'] = long.groupby('row_id').cumcount().map({0: 'p1', 1: 'p2'})
# Pivot para obtener recent_p1 y recent_p2
recent = long.pivot(index='row_id', columns='role', values='recent_count')
# Unir al df original
df['recent_p1'] = recent['p1'].values
df['recent_p2'] = recent['p2'].values

df['diff_recent_matches'] = df['recent_p1'] - df['recent_p2']

# 7. Exportar resultado
out_cols = [
    'tourney_id', 'tourney_date', 'winner_id', 'loser_id', 'surface', 'score',
    'diff_surface_win_pct', 'diff_avg_games_per_set', 'diff_recent_matches'
]
df[out_cols].to_csv('dataset_stats_2.csv', index=False)
print("dataset_stats_2.csv generado correctamente.")
