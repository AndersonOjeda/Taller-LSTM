"""Ejecutar: python train.py --epochs 40 --patience 7."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import argparse, json, time, hashlib, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipeline import ROOT, RAW, REQUIRED, prepare, windows

MODELS = ['base','profunda','propuesta']
WINDOWS = [12,24,48]


def build(name,w,f):
    layers=[keras.Input((w,f))]
    if name=='base':
        layers += [keras.layers.LSTM(32)]
    elif name=='profunda':
        layers += [keras.layers.LSTM(64,return_sequences=True),keras.layers.Dropout(.2),keras.layers.LSTM(32),keras.layers.Dropout(.2)]
    elif name=='propuesta':
        layers += [keras.layers.Conv1D(32,3,padding='causal',activation='relu'),keras.layers.LSTM(32),keras.layers.Dropout(.2),keras.layers.Dense(16,activation='relu')]
    else: raise ValueError(name)
    model=keras.Sequential(layers+[keras.layers.Dense(1)],name=name)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=.001),loss='mse')
    return model


def dataset(x,y=None):
    ds=tf.data.Dataset.from_tensor_slices(x if y is None else (x,y)).batch(128)
    opt=tf.data.Options(); opt.threading.private_threadpool_size=1
    return ds.with_options(opt).prefetch(1)


def predict(model,x):
    return model.predict(dataset(x),verbose=0).ravel()


def metrics(y,p):
    nonzero=np.abs(y)>1e-8
    mse=float(mean_squared_error(y,p))
    return {'MAE':float(mean_absolute_error(y,p)), 'MSE':mse,'RMSE':float(np.sqrt(mse)),
            'MAPE':float(np.mean(np.abs((y[nonzero]-p[nonzero])/y[nonzero]))*100) if nonzero.any() else np.nan,
            'R2':float(r2_score(y,p)), 'n':len(y)}


def savejson(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--epochs',type=int,default=40); parser.add_argument('--patience',type=int,default=7)
    parser.add_argument('--out',default=str(ROOT/'results')); args=parser.parse_args()
    if args.epochs<1 or args.patience<1: raise ValueError('epochs y patience deben ser positivos.')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    modeldir=out/'models'; modeldir.mkdir(exist_ok=True)
    d,x,y,groups,scaler,audit=prepare(pd.read_csv(RAW))
    audit['dataset_sha256']=hashlib.sha256(RAW.read_bytes()).hexdigest()
    audit['entorno']={'python':sys.version.split()[0], 'tensorflow':tf.__version__, 'keras':keras.__version__, 'numpy':np.__version__, 'pandas':pd.__version__}
    audit['configuracion']={'epochs_max':args.epochs,'patience':args.patience,'batch_size':128,'seed':42,'learning_rate':.001,'shuffle':False}
    savejson(out/'auditoria.json',audit)
    d.to_csv(out/'datos_auditados.csv')
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)
    rows=[]; histories={}
    # Fase 1: seleccionar exclusivamente con validación; test no se predice todavía.
    for w in WINDOWS:
        xt=windows(x,groups['train'],w); xv=windows(x,groups['val'],w)
        yt=y[groups['train']]; yv=y[groups['val']]
        for name in MODELS:
            keras.backend.clear_session(); keras.utils.set_random_seed(42)
            model=build(name,w,x.shape[1]); tag=f'{name}_v{w}'
            start=time.perf_counter()
            cb=keras.callbacks.EarlyStopping(monitor='val_loss',patience=args.patience,restore_best_weights=True)
            class Progress(keras.callbacks.Callback):
                def on_epoch_end(self,epoch,logs=None):
                    if epoch==0 or (epoch+1)%5==0: print(f'{tag}: época {epoch+1}, loss={logs["loss"]:.5f}, val={logs["val_loss"]:.5f}',flush=True)
            h=model.fit(dataset(xt,yt),validation_data=dataset(xv,yv),epochs=args.epochs,callbacks=[cb,Progress()],shuffle=False,verbose=0)
            hist=pd.DataFrame(h.history); hist.index=np.arange(1,len(hist)+1); hist.index.name='epoca'
            hist.to_csv(out/f'historial_{tag}.csv'); histories[tag]=hist
            # Las métricas se calculan en MW, con los mejores pesos restaurados.
            pm=predict(model,xv)*scaler['y_std']+scaler['y_mean']; true=yv*scaler['y_std']+scaler['y_mean']
            row={'Modelo':name,'Ventana':w,'Epocas':len(hist),'Mejor_epoca':int(hist.val_loss.idxmin()),'Parametros':model.count_params(),
                 'Segundos':time.perf_counter()-start, **{'Val_'+k:v for k,v in metrics(true,pm).items()}}
            pt=predict(model,xt)*scaler['y_std']+scaler['y_mean']
            row['Train_RMSE']=metrics(yt*scaler['y_std']+scaler['y_mean'],pt)['RMSE']
            rows.append(row); model.save(modeldir/f'{tag}.keras')
            pd.DataFrame(rows).to_csv(out/'seleccion_validacion.csv',index=False)
            fig,ax=plt.subplots(figsize=(7,3.4)); ax.plot(hist.index,hist.loss,label='Entrenamiento'); ax.plot(hist.index,hist.val_loss,label='Validación'); ax.set(xlabel='Época',ylabel='MSE estandarizado',title=tag); ax.legend(); fig.tight_layout(); fig.savefig(out/f'loss_{tag}.png',dpi=150); plt.close(fig)
            print(f'FINAL {tag}: RMSE validación={row["Val_RMSE"]:.3f}',flush=True)
    selected=min(rows,key=lambda r:r['Val_RMSE'])
    cfg={'model':selected['Modelo'],'window':selected['Ventana'],'selection_metric':'Val_RMSE','scaler':scaler,'dataset_sha256':audit['dataset_sha256']}
    savejson(out/'seleccion_fijada_antes_de_test.json',cfg)
    selected_tag=f'{cfg["model"]}_v{cfg["window"]}'
    # Fase 2: test final, sin modificar la selección tras consultar estas métricas.
    testrows=[]
    for row in rows:
        w=row['Ventana']; tag=f'{row["Modelo"]}_v{w}'
        model=keras.models.load_model(modeldir/f'{tag}.keras',compile=False)
        ids=groups['test']; true=y[ids]*scaler['y_std']+scaler['y_mean']
        p=predict(model,windows(x,ids,w))*scaler['y_std']+scaler['y_mean']
        testrows.append({**row,**metrics(true,p),'Seleccionado':tag==selected_tag})
        pred=pd.DataFrame({'timestamp':d.index[ids],'real_mw':true,'predicha_mw':p,'error_pred_menos_real':p-true})
        pred.to_csv(out/f'predicciones_{tag}.csv',index=False)
        fig,axes=plt.subplots(2,1,figsize=(9,6)); axes[0].plot(pred.timestamp.iloc[:168],true[:168],label='Real'); axes[0].plot(pred.timestamp.iloc[:168],p[:168],label='Predicción',alpha=.8); axes[0].set(title=tag+' · primeras 168 predicciones válidas de test',ylabel='MW'); axes[0].legend()
        axes[1].plot(pred.timestamp,p-true,lw=.7); axes[1].axhline(0,color='black',lw=.8); axes[1].set(xlabel='Hora objetivo',ylabel='Predicha - real (MW)',title='Error en todo test'); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(out/f'prediccion_error_{tag}.png',dpi=150); plt.close(fig)
    table=pd.DataFrame(testrows); table.to_csv(out/'resultados_modelos.csv',index=False)
    ids=groups['test']; yy=d.demanda_mw.to_numpy(); true=yy[ids]
    baseline=[]
    for label,p in [('Persistencia_1h',yy[ids-1]),('Estacional_24h',yy[ids-24])]:
        baseline.append({'Modelo':label,**metrics(true,p)})
    pd.DataFrame(baseline).to_csv(out/'baselines.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,4)); pos=np.arange(len(table)); ax.bar(pos-.18,table.Val_RMSE,.36,label='Validación'); ax.bar(pos+.18,table.RMSE,.36,label='Prueba'); ax.axhline(baseline[0]['RMSE'],color='black',ls='--',label='Persistencia (test)'); ax.set_xticks(pos,[f'{r.Modelo}\n{r.Ventana}h' for r in table.itertuples()]); ax.set(ylabel='RMSE (MW)',title='Comparación: mismos instantes objetivo'); ax.legend(); fig.tight_layout(); fig.savefig(out/'comparacion.png',dpi=160); plt.close(fig)
    # Importancia por perturbación: validación, bloques de 24 muestras; asociación, no causalidad.
    model=keras.models.load_model(modeldir/f'{selected_tag}.keras',compile=False)
    xv=windows(x,groups['val'],cfg['window']); true=y[groups['val']]*scaler['y_std']+scaler['y_mean']
    base=metrics(true,predict(model,xv)*scaler['y_std']+scaler['y_mean'])['RMSE']; importance=[]
    from pipeline import FEATURES
    rng=np.random.default_rng(42)
    for c,col in enumerate(FEATURES):
        vals=[]
        for repeat in range(3):
            z=xv.copy(); blocks=[np.arange(i,min(i+24,len(z))) for i in range(0,len(z),24)]
            order=np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
            z[:,:,c]=xv[order,:,c]
            vals.append(metrics(true,predict(model,z)*scaler['y_std']+scaler['y_mean'])['RMSE']-base)
        importance.append({'variable':col,'aumento_RMSE_medio':float(np.mean(vals)),'desviacion':float(np.std(vals))})
    pd.DataFrame(importance).sort_values('aumento_RMSE_medio',ascending=False).to_csv(out/'importancia_validacion.csv',index=False)
    deploy=ROOT/'models'; deploy.mkdir(exist_ok=True)
    shutil.copy2(modeldir/f'{selected_tag}.keras',deploy/'mejor_modelo.keras'); savejson(deploy/'config.json',cfg)
    # Ejemplo real y completo del final del histórico; no se repite una sola fila.
    valid=np.isfinite(x).all(axis=1)
    end=next(k for k in range(len(d),cfg['window']-1,-1) if valid[k-cfg['window']:k].all())
    d.iloc[end-cfg['window']:end].reset_index()[REQUIRED].to_csv(ROOT/'data/ejemplo_entrada.csv',index=False)
    savejson(out/'finalizado.json',{'experimentos':len(rows),'seleccionado':selected_tag,'estado':'Entrenamiento y evaluación reales completados'})
    print(table[['Modelo','Ventana','MAE','RMSE','R2','Epocas','Seleccionado']].to_string(index=False),flush=True)
    print('SELECCIONADO POR VALIDACIÓN:',selected_tag,flush=True)

if __name__=='__main__': main()
