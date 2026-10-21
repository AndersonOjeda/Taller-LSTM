# Taller LSTM · Demanda energética

**Estudiante:** Anderson David Ojeda Zambrano · Ingeniería de Software · UCC Pasto  
**Asignatura:** Inteligencia Artificial · Profesor Cristian Camilo Ordoñez Quintero

Implementación propia apoyada en la estructura y el dataset del repositorio de referencia:
https://github.com/samumafu/Ia-Taller-ltsm  
Revisión consultada: `085b97d225c8e835e27b4ec4ed94418c80dfc3ad`.
Se reconoce la referencia de samumafu; no se presentan sus modelos ni métricas como propios.

## Empezar en Windows

Usa Python **3.12**, abre una terminal en la carpeta extraída y ejecuta:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

No necesitas activar el entorno ni reentrenar para abrir la app: el paquete entregado contiene el modelo seleccionado, su configuración y un ejemplo real de entrada. Abre la dirección local que muestre Streamlit.

En Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

## Reproducir el experimento

Para comprobar el dataset, recalcular las métricas de los nueve CSV de predicciones
y probar la interfaz sin reentrenar:

```powershell
.\.venv\Scripts\python.exe verificar_entrega.py --app
```

El resultado de esta revisión se guarda en `results/revision_entrega.json`.
También puedes abrir `iniciar_app.cmd` con doble clic cuando el entorno esté instalado.

Para Render, usa Python 3.12 (fijado en `.python-version`), comando de construcción
`pip install -r requirements-app.txt` y comando de inicio `python render_start.py`.
El arranque toma el puerto de `PORT` y escucha en `0.0.0.0`.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s . -p test_pipeline.py -v
.\.venv\Scripts\python.exe train.py --epochs 40 --patience 7
```

El entrenamiento puede demorar según el computador. Usa TensorFlow CPU; no exige GPU.
Se regeneran nueve modelos y la selección de la app. El informe PDF incluido corresponde a la ejecución entregada; si reentrenas, vuelve a generarlo mediante `python generar_informe.py` (requiere `pip install reportlab`).
Para revisar por etapas, abre `Taller_LSTM_Anderson.ipynb` en Jupyter o VS Code con extensión de notebooks. El notebook usa los módulos de esta carpeta: no subas solamente el `.ipynb` a Colab. En Colab, carga y extrae primero el ZIP completo, entra a la carpeta y ejecuta las celdas. No es necesario instalar dependencias si ya son compatibles; el notebook muestra el comando cuando hagan falta.

## Contenido

- `pipeline.py`: limpieza auditable, calendario, división temporal, escalado y validación de inferencia.
- `train.py`: las nueve combinaciones, selección por validación, prueba final, baselines, gráficos e importancia exploratoria.
- `app.py`: interfaz web con CSV editable y predicción en MW.
- `models/`: mejor modelo y configuración de escalado para la aplicación.
- `results/`: métricas reales, predicciones, curvas de aprendizaje, auditoría y los nueve modelos.
- `data/`: dataset fuente y ejemplo para la aplicación.
- `test_pipeline.py`: pruebas de temporalidad, datos inválidos y paridad de preprocesamiento.
- `Informe_Taller_LSTM_Anderson.pdf`: metodología, resultados, respuestas y limitaciones.
- `generar_informe.py`: reproduce el PDF a partir de los resultados guardados.

## Decisiones metodológicas

1. `X(k) = [k-n, ..., k-1]`, `y(k) = demanda_mw(k)`. El objetivo se acepta solo si coincide con `demanda_objetivo(k-1)`. No se emplea `demanda_objetivo` como predictor. Los 40 conflictos se excluyen como etiquetas; no se elige arbitrariamente cuál fuente es correcta.
2. Se eliminan duplicados exactos, se ordena el tiempo y se mantiene una rejilla horaria. Humedad fuera de 0–100 % y magnitudes físicamente negativas se marcan como faltantes. No se rellenan con interpolación, medias, ceros ni valores futuros.
3. Precios negativos y extremos estadísticos se conservan sin una regla de dominio que demuestre su invalidez. La magnitud de temperatura, viento y demanda debe investigarse con la fuente; no se recorta por percentiles.
4. Se derivan hora, día, mes y fin de semana desde timestamp. `periodo_dia` se normaliza determinísticamente pero no se incluye en X por redundancia con hora. Hora, día y mes se codifican con seno/coseno. Festivo se conserva; no se presupone un país para el calendario del dataset.
5. División 70/15/15 por el timestamp del valor a predecir. Se permite contexto anterior en validación y prueba porque ya estaría disponible al emitir la predicción. Esto es evaluación progresiva a una hora, no un pronóstico simultáneo de todo el bloque de prueba.
6. Los escaladores se calculan únicamente con entrenamiento. Los nueve modelos se comparan sobre los mismos instantes válidos, exigiendo las 48 horas históricas completas incluso para 12 y 24 horas. Esto aísla el efecto de la ventana, pero reduce la cobertura al 42,28 % y la evaluación no representa automáticamente las horas con datos faltantes.
7. Base: LSTM(32). Profunda: LSTM(64) + Dropout(0,2) + LSTM(32) + Dropout(0,2). Propuesta: Conv1D causal(32, kernel=3) + LSTM(32) + Dropout(0,2) + Dense(16, ReLU). Las tres terminan en Dense(1) lineal.
8. Adam, MSE, lote 128, máximo 40 épocas, EarlyStopping con paciencia 7 y restauración de los mejores pesos. La entrada no se baraja.
9. Selección fijada por menor RMSE de validación antes de consultar prueba. Todos los modelos se evalúan una vez en prueba para cumplir la tabla comparativa; no se cambia el ganador por esos resultados.
10. MAE, MSE, RMSE, MAPE (%) y R² se calculan al volver a MW. MAPE excluye objetivos cero; `n` informa el tamaño evaluado. Se comparan persistencia de 1 hora y referencia estacional de 24 horas.

## Despliegue público (paso pendiente)

La app está implementada y comprobada localmente. Este paquete no proporciona una URL pública desplegada.

Para publicar en Streamlit Community Cloud:

1. Crea un repositorio tuyo y sube `app.py`, `pipeline.py`, `requirements.txt`, `models/mejor_modelo.keras`, `models/config.json`, `data/ejemplo_entrada.csv` y opcionalmente `results/resultados_modelos.csv`.
2. En https://share.streamlit.io/ conecta tu GitHub y elige **Create app**.
3. Selecciona tu repositorio, la rama y `app.py`. En la configuración avanzada selecciona Python 3.12, si está disponible.
4. Pulsa **Deploy**. Comprueba carga del ejemplo, predicción, carga de CSV e ingreso inválido. Conserva el enlace y una captura para la entrega del taller.

No publiques claves ni archivos de entorno. La app no necesita credenciales para inferir.
Documentación oficial: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Alcance y honestidad de los resultados

El CSV contiene fechas hasta diciembre de 2026. Se utiliza como dataset académico suministrado; no se ha verificado que sean mediciones observadas, ni su jurisdicción o generador. Una sola semilla y una sola partición no prueban superioridad estadística ni calidad en producción. Requiere validación temporal adicional, manejo operativo de datos faltantes, monitoreo y revisión del origen antes de uso real.
