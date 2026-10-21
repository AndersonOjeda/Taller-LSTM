"""Pruebas de riesgos temporales y coherencia de inferencia."""
import unittest
import numpy as np
import pandas as pd
from pipeline import REQUIRED, prepare, windows, validate_input, features


def fixture():
    n=1000; t=pd.date_range('2025-01-01',periods=n,freq='h')
    demand=500+np.arange(n)*.1
    return pd.DataFrame({'timestamp':t,'demanda_mw':demand,'temperatura_c':20.,'humedad_pct':60.,
        'viento_kmh':10.,'radiacion_wm2':0.,'precipitacion_mm':0.,'precio_kwh':25.,'festivo':0,
        'periodo_dia':'TARDE','demanda_objetivo':np.r_[demand[1:],np.nan]})

class TemporalTests(unittest.TestCase):
    def test_next_hour_and_disjoint_targets(self):
        d,x,y,g,s,a=prepare(fixture())
        for name,ids in g.items():
            for k in ids:
                self.assertEqual(d.index[k]-d.index[k-1],pd.Timedelta(hours=1))
            self.assertTrue(np.allclose(windows(x,ids,12)[0],x[ids[0]-12:ids[0]]))
        self.assertLess(max(g['train']),min(g['val']))
        self.assertLess(max(g['val']),min(g['test']))
        self.assertTrue(np.allclose(y[g['test']]*s['y_std']+s['y_mean'],d.demanda_mw.iloc[g['test']],atol=.001))
    def test_future_change_cannot_change_training(self):
        raw=fixture(); a=prepare(raw)
        raw.loc[850:,'temperatura_c']=1000
        raw.loc[850:,'demanda_mw']+=100
        raw.loc[849:998,'demanda_objetivo']+=100
        b=prepare(raw)
        self.assertEqual(a[4],b[4])
        np.testing.assert_array_equal(a[1][:700],b[1][:700])
        np.testing.assert_array_equal(a[2][:700],b[2][:700])
    def test_gaps_are_not_bridged(self):
        raw=fixture().drop(index=100)
        d,x,y,g,s,a=prepare(raw)
        ids=np.concatenate(list(g.values()))
        self.assertEqual(a['horas_ausentes_insertadas_sin_rellenar'],1)
        self.assertFalse(any(k in ids for k in range(101,149)))
    def test_no_imputation_and_conflicting_labels_excluded(self):
        raw=fixture(); raw.loc[100,'humedad_pct']=120; raw.loc[200,'demanda_objetivo']=1
        d,x,y,g,s,a=prepare(raw); ids=np.concatenate(list(g.values()))
        self.assertTrue(np.isnan(d.humedad_pct.iloc[100]))
        self.assertNotIn(201,ids)
        self.assertFalse(any(k in ids for k in range(101,149)))
    def test_input_requires_sequence(self):
        raw=fixture()[REQUIRED].head(24)
        validate_input(raw,24)
        with self.assertRaises(ValueError): validate_input(raw.iloc[::-1],24)
        with self.assertRaises(ValueError): validate_input(raw.head(1),24)
        raw.loc[3,'humedad_pct']=101
        with self.assertRaises(ValueError): validate_input(raw,24)
    def test_training_inference_feature_parity(self):
        raw=fixture(); d,x,y,g,s,a=prepare(raw)
        sample=raw[REQUIRED].iloc[50:74]
        z=validate_input(sample,24)
        np.testing.assert_allclose(features(z),features(d.iloc[50:74]))
    def test_duplicate_conflict_is_not_arbitrarily_resolved(self):
        raw=fixture(); other=raw.iloc[[10]].copy(); other['demanda_mw']=900
        with self.assertRaises(ValueError): prepare(pd.concat([raw,other]))

if __name__=='__main__': unittest.main(verbosity=2)
