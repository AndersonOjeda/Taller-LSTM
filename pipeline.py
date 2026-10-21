"""Preparación y predicción compartidas por entrenamiento y aplicación."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data/dataset_demanda_energia_LSTM_2025_2026.csv'
NUM = ['demanda_mw', 'temperatura_c', 'humedad_pct', 'viento_kmh',
       'radiacion_wm2', 'precipitacion_mm', 'precio_kwh']
REQUIRED = ['timestamp', *NUM, 'festivo']
FEATURES = [*NUM, 'festivo', 'fin_semana', 'hora_sin', 'hora_cos',
            'semana_sin', 'semana_cos', 'mes_sin', 'mes_cos']


def clean(raw):
    """Sin interpolación, relleno, clipping ni estadísticos de datos futuros."""
    missing = set(REQUIRED) - set(raw.columns)
    if missing:
        raise ValueError(f'Faltan columnas: {sorted(missing)}')
    d = raw.copy()
    audit = {'filas_originales': len(d), 'duplicados_exactos': int(d.duplicated().sum())}
    d = d.drop_duplicates()
    d['timestamp'] = pd.to_datetime(d['timestamp'], errors='coerce')
    invalid = d.timestamp.isna() | d.timestamp.ne(d.timestamp.dt.floor('h'))
    audit['timestamps_invalidos_o_no_horarios'] = int(invalid.sum())
    d = d.loc[~invalid].sort_values('timestamp').copy()
    if d.timestamp.duplicated().any():
        raise ValueError('Hay horas duplicadas con datos distintos: resolver con la fuente.')
    if len(d) < 49:
        raise ValueError('Se requieren al menos 49 horas para preparar el taller.')
    for c in [*NUM, 'festivo', 'demanda_objetivo']:
        if c in d:
            d[c] = pd.to_numeric(d[c].astype(str).str.strip().str.replace(',', '.', regex=False), errors='coerce')
            d[c] = d[c].replace([np.inf, -np.inf], np.nan)
    audit['faltantes_antes'] = {c: int(d[c].isna().sum()) for c in NUM + ['festivo']}
    audit['fisicamente_invalidos'] = {}
    rules = {'humedad_pct': ~d.humedad_pct.between(0, 100),
             'festivo': ~d.festivo.isin([0, 1])}
    # No se fija un máximo arbitrario para demanda, temperatura, viento o precio.
    for c in ['demanda_mw', 'viento_kmh', 'radiacion_wm2', 'precipitacion_mm']:
        rules[c] = d[c] < 0
    for c, mask in rules.items():
        mask = mask & d[c].notna()
        audit['fisicamente_invalidos'][c] = int(mask.sum())
        d.loc[mask, c] = np.nan
    audit['precios_negativos_conservados'] = int((d.precio_kwh < 0).sum())
    hour = d.timestamp.dt.hour
    d['hora'] = hour
    d['dia_semana'] = d.timestamp.dt.dayofweek
    d['fin_semana'] = (d.dia_semana >= 5).astype(int)
    d['mes'] = d.timestamp.dt.month
    # Es redundante con hora: se documenta/corrige, pero no se duplica en X.
    period = np.select([hour < 6, hour < 12, hour < 18], ['madrugada', 'mañana', 'tarde'], default='noche')
    audit['periodos_corregidos'] = int((d['periodo_dia'].astype(str) != period).sum()) if 'periodo_dia' in d else 0
    d['periodo_dia'] = period
    d = d.set_index('timestamp')
    full = pd.date_range(d.index.min(), d.index.max(), freq='h', name='timestamp')
    audit['horas_ausentes_insertadas_sin_rellenar'] = len(full) - len(d)
    d = d.reindex(full)
    # y(k) es la demanda observada en k. Verificamos la etiqueta publicada en k-1.
    if 'demanda_objetivo' not in d:
        raise ValueError('El CSV de entrenamiento requiere demanda_objetivo para auditar etiquetas.')
    observed = d.demanda_mw
    declared = d.demanda_objetivo.shift(1)
    comparable = observed.notna() & declared.notna()
    agree = np.isclose(observed, declared, rtol=0, atol=1e-6)
    audit['etiquetas_comparables'] = int(comparable.sum())
    audit['etiquetas_discrepantes_excluidas'] = int((comparable & ~agree).sum())
    d['target_observado'] = observed.where(comparable & agree)
    audit['filas_horarias'] = len(d)
    audit['inicio'] = str(d.index.min()); audit['fin'] = str(d.index.max())
    return d, audit


def features(d):
    """Transformaciones deterministas; sin ajuste y sin variables objetivo."""
    x = d[NUM + ['festivo']].copy()
    t = pd.DatetimeIndex(d.index)
    x['fin_semana'] = (t.dayofweek >= 5).astype(float)
    for name, value, cycle in [('hora', t.hour, 24), ('semana', t.dayofweek, 7), ('mes', t.month - 1, 12)]:
        x[name + '_sin'] = np.sin(2 * np.pi * value / cycle)
        x[name + '_cos'] = np.cos(2 * np.pi * value / cycle)
    return x[FEATURES].to_numpy(dtype=np.float64)


def prepare(raw, max_window=48):
    d, audit = clean(raw)
    x = features(d); y = d.target_observado.to_numpy()
    n = len(d); cut1 = int(n * .70); cut2 = int(n * .85)
    valid_row = np.isfinite(x).all(axis=1)
    # Mismos instantes objetivo para las tres ventanas: comparación justa.
    ids = np.array([k for k in range(max_window, n)
                    if valid_row[k-max_window:k].all() and np.isfinite(y[k])])
    groups = {'train': ids[ids < cut1], 'val': ids[(ids >= cut1) & (ids < cut2)], 'test': ids[ids >= cut2]}
    if any(len(v) < 20 for v in groups.values()):
        raise ValueError('Muy pocas ventanas completas en algún conjunto.')
    # Solo filas que realmente participan como entradas del entrenamiento.
    used = np.zeros(n, dtype=bool)
    for k in groups['train']:
        used[k-max_window:k] = True
    mean = x[used].mean(axis=0); std = x[used].std(axis=0)
    std[std == 0] = 1
    ymean = float(y[groups['train']].mean()); ystd = float(y[groups['train']].std())
    if ystd == 0: ystd = 1.
    scaler = {'features': FEATURES, 'mean': mean.tolist(), 'std': std.tolist(), 'y_mean': ymean, 'y_std': ystd}
    audit['cortes_por_timestamp_objetivo'] = {'inicio_validacion': str(d.index[cut1]), 'inicio_prueba': str(d.index[cut2])}
    audit['muestras_comunes'] = {g: len(v) for g, v in groups.items()}
    audit['candidatos_desde_hora_48'] = n - max_window
    audit['cobertura_pct'] = float(len(ids) / (n-max_window) * 100)
    audit['filas_ajuste_scaler_X'] = int(used.sum())
    audit['max_fecha_ajuste_X'] = str(d.index[used][-1])
    audit['max_fecha_ajuste_y'] = str(d.index[groups['train'][-1]])
    return d, ((x-mean)/std).astype('float32'), ((y-ymean)/ystd).astype('float32'), groups, scaler, audit


def windows(x, ids, w):
    return np.stack([x[k-w:k] for k in ids]).astype('float32')


def validate_input(raw, window):
    missing = set(REQUIRED) - set(raw.columns)
    if missing: raise ValueError(f'Faltan columnas: {sorted(missing)}')
    if len(raw) != window: raise ValueError(f'Ingresa exactamente {window} filas.')
    d = raw[REQUIRED].copy()
    t = pd.to_datetime(d.pop('timestamp'), errors='coerce')
    if t.isna().any() or not t.is_monotonic_increasing or t.duplicated().any():
        raise ValueError('Fechas inválidas, repetidas o desordenadas.')
    if (t != t.dt.floor('h')).any() or not t.diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError('Se requieren horas completas y consecutivas, sin saltos.')
    for c in NUM + ['festivo']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    if not np.isfinite(d.to_numpy(dtype=float)).all(): raise ValueError('Hay valores faltantes o no numéricos.')
    if not d.humedad_pct.between(0,100).all(): raise ValueError('Humedad fuera de 0 a 100 %.')
    if not d.festivo.isin([0,1]).all(): raise ValueError('festivo debe ser 0 o 1.')
    for c in ['demanda_mw','viento_kmh','radiacion_wm2','precipitacion_mm']:
        if (d[c] < 0).any(): raise ValueError(f'{c} no puede ser negativo.')
    d.index = pd.DatetimeIndex(t)
    return d


def inference(raw, model, config):
    d = validate_input(raw, config['window'])
    s = config['scaler']; x = features(d)
    if s['features'] != FEATURES: raise ValueError('Orden de variables incompatible.')
    z = ((x-np.array(s['mean']))/np.array(s['std'])).astype('float32')[None, ...]
    scaled = np.asarray(model(z, training=False)).reshape(-1)[0]
    pred = float(scaled*s['y_std']+s['y_mean'])
    if not np.isfinite(pred): raise ValueError('Predicción no finita.')
    return pred, d.index[-1]+pd.Timedelta(hours=1)
