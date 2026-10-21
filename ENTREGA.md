# Revisión de la entrega

Los resultados de entrenamiento ya estaban incluidos en el proyecto al iniciar
esta revisión. Se contrastaron con el dataset aportado y se recalcularon las
métricas a partir de las predicciones, sin atribuirlos a un nuevo entrenamiento.

| Requisito del enunciado | Evidencia |
| --- | --- |
| Orden temporal, limpieza justificada y ausencia de relleno inventado | `pipeline.py`, `results/auditoria.json`, informe |
| Partición cronológica y escalado con entrenamiento | `pipeline.py`, `test_pipeline.py` |
| Base, profunda y propuesta; ventanas de 12, 24 y 48 h | `train.py`, nueve modelos en `results/models/` |
| MAE, MSE, RMSE, MAPE, R² y épocas | `results/resultados_modelos.csv` |
| Loss de entrenamiento y validación | Nueve imágenes `results/loss_*.png` |
| Demanda real, predicha y error | Nueve imágenes `results/prediccion_error_*.png` |
| Comparación de modelos | `results/comparacion.png` |
| Sobreajuste, generalización y nueve respuestas | `Informe_Taller_LSTM_Anderson.pdf` |
| Ingreso de X y visualización de y | `app.py`, `data/ejemplo_entrada.csv` |
| Comprobación reproducible de artefactos e interfaz | `verificar_entrega.py --app` |
| Publicación web | https://taller-lstm-anderson.onrender.com; predicción pública y captura verificadas |

La ventana del modelo seleccionado es de 48 horas. Esas filas deben ser
consecutivas; la salida corresponde a la hora inmediatamente posterior.
El modelo seleccionado por validación es la LSTM profunda: RMSE de prueba
28,06 MW y R² de 0,880, sobre 1.182 instantes válidos. La cobertura de los
instantes candidatos es 42,28 %; las métricas no describen las horas excluidas.

Referencia reconocida: https://github.com/samumafu/Ia-Taller-ltsm.
La aplicación del compañero no es la publicación de esta entrega.

Para ejecutar en este equipo, abre `iniciar_app.cmd`. Para revisar todos los
artefactos: `.venv\Scripts\python.exe verificar_entrega.py --app`.
Para crear el ZIP de entrega: `python empacar_entrega.py`.
