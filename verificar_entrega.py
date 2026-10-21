"""Audita los artefactos existentes sin volver a entrenar.

Ejecutar con --app para comprobar ademas el modelo y la interfaz Streamlit.
No sustituye ni sobrescribe el registro de la ejecucion original.
"""
import argparse
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from pipeline import ROOT, RAW, REQUIRED, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'results'
    audit = json.loads((out / 'auditoria.json').read_text(encoding='utf-8'))
    cfg = json.loads((ROOT / 'models/config.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256(RAW.read_bytes()).hexdigest()
    assert digest == audit['dataset_sha256'] == cfg['dataset_sha256'], 'Dataset distinto al entrenado'
    supplied = ROOT / RAW.name
    # El archivo entregado puede diferir en BOM o salto de linea final.
    pd.testing.assert_frame_equal(pd.read_csv(supplied), pd.read_csv(RAW))
    d, x, y, groups, scaler, current = prepare(pd.read_csv(RAW))
    assert current['muestras_comunes'] == audit['muestras_comunes']
    for key in ['mean', 'std', 'y_mean', 'y_std']:
        np.testing.assert_allclose(scaler[key], cfg['scaler'][key], rtol=1e-12)
    table = pd.read_csv(out / 'resultados_modelos.csv')
    assert len(table) == 9 and not table.duplicated(['Modelo', 'Ventana']).any()
    assert set(zip(table.Modelo, table.Ventana)) == {
        (m, w) for m in ['base', 'profunda', 'propuesta'] for w in [12, 24, 48]}
    selected = table.loc[table.Seleccionado]
    assert len(selected) == 1
    winner = table.loc[table.Val_RMSE.idxmin()]
    assert winner.Modelo == cfg['model'] and int(winner.Ventana) == cfg['window']
    assert selected.index[0] == winner.name
    expected_time = d.index[groups['test']]
    expected_y = d.demanda_mw.iloc[groups['test']].to_numpy()
    for row in table.itertuples():
        tag = f'{row.Modelo}_v{row.Ventana}'
        pred = pd.read_csv(out / f'predicciones_{tag}.csv')
        assert pd.DatetimeIndex(pd.to_datetime(pred.timestamp)).equals(expected_time)
        np.testing.assert_allclose(pred.real_mw, expected_y, atol=0.0002)
        real = pred.real_mw.to_numpy()
        err = pred.predicha_mw.to_numpy() - real
        calculated = dict(MAE=np.abs(err).mean(), MSE=np.square(err).mean(),
                          RMSE=np.sqrt(np.square(err).mean()),
                          MAPE=(np.abs(err[real != 0] / real[real != 0])).mean() * 100,
                          R2=1-np.square(err).sum()/np.square(real-real.mean()).sum())
        for name, value in calculated.items():
            np.testing.assert_allclose(value, getattr(row, name), rtol=3e-6, atol=3e-5)
        history = pd.read_csv(out / f'historial_{tag}.csv')
        assert len(history) == row.Epocas
        assert int(history.loc[history.val_loss.idxmin(), 'epoca']) == row.Mejor_epoca
        for filename in [f'models/{tag}.keras', f'loss_{tag}.png', f'prediccion_error_{tag}.png']:
            assert (out / filename).stat().st_size > 0
    suite = unittest.defaultTestLoader.discover(str(ROOT), pattern='test_pipeline.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful()
    report = {'dataset_sha256': digest, 'experimentos_verificados': len(table),
              'metricas_recalculadas_desde_predicciones': True,
              'pruebas_temporales_aprobadas': result.testsRun,
              'seleccionado': f'{cfg["model"]}_v{cfg["window"]}',
              'app_verificada_en_esta_ejecucion': False}
    if args.app:
        from tensorflow import keras
        from pipeline import inference
        from streamlit.testing.v1 import AppTest
        model = keras.models.load_model(ROOT / 'models/mejor_modelo.keras', compile=False)
        example = pd.read_csv(ROOT / 'data/ejemplo_entrada.csv')
        value, timestamp = inference(example, model, cfg)
        at = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=90)
        assert not at.exception, str(at.exception)
        button = next(b for b in at.button if b.label == 'Predecir siguiente hora')
        button.click().run(timeout=90)
        assert not at.exception and not at.error, str(at.exception)
        assert len(at.metric) == 2
        assert at.metric[0].value == f'{value:,.2f} MW'
        invalid = example.copy()
        invalid.loc[0, 'humedad_pct'] = 101
        try:
            inference(invalid, model, cfg)
        except ValueError:
            pass
        else:
            raise AssertionError('Se acepto humedad invalida')
        report.update(app_verificada_en_esta_ejecucion=True,
                      demanda_ejemplo_mw=value, hora_objetivo=str(timestamp))
    (out / 'revision_entrega.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
