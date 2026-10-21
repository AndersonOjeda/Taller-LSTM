"""Genera el informe desde los resultados reales guardados (pip install reportlab)."""
from pathlib import Path
import json
from xml.sax.saxutils import escape
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4

R=Path(__file__).resolve().parent; O=R/'results'
t=pd.read_csv(O/'resultados_modelos.csv'); a=json.loads((O/'auditoria.json').read_text()); c=json.loads((R/'models/config.json').read_text())
best=t[t.Seleccionado].iloc[0]; tag=f'{best.Modelo}_v{int(best.Ventana)}'
bas=pd.read_csv(O/'baselines.csv'); imp=pd.read_csv(O/'importancia_validacion.csv')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCustom',fontName='Helvetica-Bold',fontSize=25,leading=29,textColor=colors.HexColor('#132B46'),spaceAfter=18))
styles.add(ParagraphStyle(name='Kicker',fontSize=9,leading=12,textColor=colors.HexColor('#007D83'),spaceAfter=8))
styles['BodyText'].fontSize=10;styles['BodyText'].leading=14;styles['BodyText'].spaceAfter=8
styles['Heading1'].fontSize=17;styles['Heading1'].leading=21;styles['Heading1'].textColor=colors.HexColor('#132B46');styles['Heading1'].spaceAfter=13
styles['Heading2'].fontSize=11;styles['Heading2'].leading=15;styles['Heading2'].textColor=colors.HexColor('#007D83');styles['Heading2'].spaceBefore=8
styles.add(ParagraphStyle(name='SmallCustom',fontSize=8,leading=11,spaceAfter=7))
story=[]
def p(s,style='BodyText'): story.append(Paragraph(s,styles[style]))
def h(s):p(s,'Heading1')
def sub(s):p(s,'Heading2')
def page(): story.append(PageBreak())
def table(rows,widths=None):
    rr=[[Paragraph(str(x),styles['SmallCustom']) for x in row] for row in rows]
    tb=Table(rr,colWidths=widths,repeatRows=1,hAlign='LEFT'); tb.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DDEEF0')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F3F6FA')]),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),4),
        ('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#9CB1BC'))]));story.append(tb);story.append(Spacer(1,9))
def pic(name,width):
    im=Image(str(O/name)); im.drawHeight=im.imageHeight*width/im.imageWidth;im.drawWidth=width;story.append(im)
def link(label,url):p(f'{escape(label)}: <link href="{url}" color="#007D83">{escape(url)}</link>','SmallCustom')

p('INTELIGENCIA ARTIFICIAL / TALLER PRÁCTICO','Kicker')
p('Predicción de demanda<br/>energética con LSTM','TitleCustom')
p('<b>Anderson David Ojeda Zambrano</b><br/>Ingeniería de Software · Universidad Cooperativa de Colombia, Campus Pasto<br/>Profesor: Cristian Camilo Ordoñez Quintero<br/>21 de octubre de 2026')
p('Diseño, implementación y comparación de tres arquitecturas con ventanas de 12, 24 y 48 horas. Resultados obtenidos mediante una ejecución propia sobre el dataset del repositorio de referencia.')
sub('1. Objetivo y formulación')
p('Predecir la demanda eléctrica de la siguiente hora usando exclusivamente el histórico multivariado. Para una hora objetivo k, X contiene las horas k-n hasta k-1 y la salida y es la demanda observada en k. La salida tiene una sola neurona lineal y se expresa en megavatios (MW).')
sub('2. Dataset y procedencia')
p(f'Archivo: dataset_demanda_energia_LSTM_2025_2026.csv. Contiene {a["filas_originales"]:,} filas y 15 columnas, desde {a["inicio"]} hasta {a["fin"]}. Después de eliminar {a["duplicados_exactos"]} duplicados exactos quedan {a["filas_horarias"]:,} horas. Se obtuvo del repositorio de samumafu, revisión 085b97d225c8e835e27b4ec4ed94418c80dfc3ad.')
p('Se utiliza como dataset académico. No se ha verificado su origen observacional, jurisdicción o método de generación. Contiene fechas posteriores a la fecha de este informe; por ello no se presenta como evidencia de consumo real ya observado.')
sub('3. Diagnóstico y reglas de limpieza')
table([['Hallazgo','Tratamiento justificado'],['280 faltantes tras quitar duplicados','Se conservan como NaN; no se imputan valores.'],['9 humedades inválidas y 8 vientos negativos','Se marcan como NaN por violar límites físicos.'],['80 textos de periodo_dia corregidos','Se derivan del timestamp, igual que las variables de calendario.'],['40 desacuerdos entre etiqueta y medición siguiente','Se excluyen como objetivos; no se inventa cuál valor es correcto.'],['3 precios negativos y otros extremos','Se conservan sin evidencia suficiente para declararlos errores.']],[205,290])
p('El valor objetivo solo se admite cuando demanda_mw(k) coincide con demanda_objetivo(k-1). Esta comprobación se realiza fuera de las entradas X. No se elimina una hora para unir artificialmente las horas anterior y posterior.','SmallCustom')

page();h('Método experimental')
sub('4. Separación temporal y escalado')
table([['Conjunto','Intervalo por hora objetivo','Muestras válidas'],['Entrenamiento','01/01/2025 a 26/05/2026',a['muestras_comunes']['train']],['Validación','27/05/2026 a 13/09/2026 11:00',a['muestras_comunes']['val']],['Prueba','13/09/2026 12:00 a 31/12/2026',a['muestras_comunes']['test']]],[95,295,105])
p('Los intervalos corresponden al reparto 70/15/15 de la rejilla, antes de descartar ventanas inválidas. Las primeras 48 horas no tienen suficiente contexto para la comparación común. El escalador X se ajustó con 9.255 filas que participan en ventanas de entrenamiento; el escalador y, con sus 5.110 objetivos. Validación y prueba solo usan esos parámetros.')
p('Para comparar las tres ventanas de manera justa se utilizan los mismos instantes objetivo, que disponen de 48 horas históricas completas. Se conservan 7.387 de 17.472 instantes candidatos (42,28 %). Es una decisión conservadora: evita inventar valores, pero reduce la cobertura y puede introducir sesgo hacia periodos con mejor calidad de datos.')
p('La evaluación es progresiva a una hora: cada predicción puede utilizar mediciones anteriores del bloque de prueba. No se utiliza el valor de la hora objetivo ni información posterior. No equivale a pronosticar todo el trimestre desde una única fecha inicial.')
sub('5. Variables de entrada')
p('Se usan 15 variables: demanda, temperatura, humedad, viento, radiación, precipitación, precio, festivo, fin de semana y codificaciones seno/coseno para hora, día de semana y mes. Las codificaciones representan la continuidad de los ciclos. periodo_dia se normaliza, pero se omite del modelo por redundancia con hora. Timestamp y demanda_objetivo no son entradas.')
sub('6. Arquitecturas')
table([['Modelo','Capas principales','Parámetros'],['A. Base','LSTM(32) → Dense(1)','6.177'],['B. Profunda','LSTM(64, secuencias) → Dropout(0,2) → LSTM(32) → Dropout(0,2) → Dense(1)','32.929'],['C. Propuesta','Conv1D(32, kernel 3, causal, ReLU) → LSTM(32) → Dropout(0,2) → Dense(16, ReLU) → Dense(1)','10.337']],[85,340,70])
p('La base permite medir si una estructura sencilla es suficiente. La profunda añade capacidad para aprender representaciones temporales sucesivas. La propuesta combina detección de patrones locales mediante convolución con memoria LSTM; su padding causal evita mirar hacia delante en la convolución.')
sub('7. Entrenamiento y selección')
p('Configuración común: Adam (0,001), pérdida MSE, lote 128, semilla 42, máximo 40 épocas y EarlyStopping con paciencia 7, restaurando los mejores pesos. Los datos no se barajan. La elección se fija por el menor RMSE de validación y se guarda antes de calcular las métricas de prueba. No se cambia el ganador después de mirar test.')

page();h('Resultados de los nueve experimentos')
p('Métricas de prueba en los mismos 1.182 instantes objetivo. MAE y RMSE están en MW; MSE, en MW²; MAPE, en porcentaje; R² es adimensional. Se calculan tras deshacer el escalado. Un R² de 0,88 no significa 88 % de exactitud.')
rows=[['Modelo','n (h)','MAE','MSE','RMSE','MAPE %','R²','Épocas']]
for r in t.itertuples(): rows.append([r.Modelo+('*' if r.Seleccionado else ''),int(r.Ventana),f'{r.MAE:.2f}',f'{r.MSE:.2f}',f'{r.RMSE:.2f}',f'{r.MAPE:.2f}',f'{r.R2:.3f}',int(r.Epocas)])
table(rows,[84,39,53,66,57,64,59,73])
p('* Seleccionado previamente por validación. Épocas indica el total ejecutado; la profunda de 48 horas ejecutó 27 y restauró los pesos de la época 20.','SmallCustom')
pic('comparacion.png',495)
sub('Comparación con referencias simples')
table([['Referencia / modelo','MAE (MW)','RMSE (MW)','R²'],*[[r.Modelo,f'{r.MAE:.2f}',f'{r.RMSE:.2f}',f'{r.R2:.3f}'] for r in bas.itertuples()],['LSTM profunda de 48 h',f'{best.MAE:.2f}',f'{best.RMSE:.2f}',f'{best.R2:.3f}']],[240,85,90,80])
improve=(1-best.RMSE/bas.iloc[0].RMSE)*100
p(f'La seleccionada reduce el RMSE un {improve:.1f} % frente a persistir la última demanda. La base de 12 horas obtiene RMSE 28,42 MW con 6.177 parámetros; su diferencia con la profunda de 48 horas es pequeña. Una sola ejecución no demuestra que esa diferencia sea estadísticamente significativa.')

page();h('Gráficas del modelo seleccionado')
p('LSTM profunda · ventana de 48 horas. Se incluyen también las gráficas individuales de los otros ocho experimentos en la carpeta results.')
pic(f'loss_{tag}.png',470)
p('La pérdida está calculada sobre el objetivo estandarizado. Dropout está activo durante entrenamiento y desactivado al validar, lo que puede hacer que la loss de validación sea menor sin indicar una fuga de información.','SmallCustom')
pic(f'prediccion_error_{tag}.png',495)
p('La gráfica superior muestra las primeras 168 predicciones válidas del bloque de prueba, que pueden abarcar más de una semana por las exclusiones. La inferior contiene los 1.182 errores: un signo positivo representa sobreestimación.','SmallCustom')

page();h('Respuestas de análisis · 1 a 6')
sub('1. ¿Por qué LSTM es apropiada para este problema?')
p('Porque la demanda de una hora depende del consumo anterior y de patrones horarios y semanales. Una LSTM procesa secuencias y utiliza un estado de memoria y compuertas para conservar o descartar información relevante. En este trabajo combina ese contexto temporal con clima, precio y calendario. Su idoneidad se respalda comparándola con referencias simples; no se supone que sea la mejor familia posible.')
sub('2. ¿Qué efecto tiene cambiar la ventana temporal?')
p('Una ventana de 12 horas aporta contexto reciente; 24 cubre un ciclo diario y 48 incorpora dos días. Más historia aumenta el costo y puede aportar redundancia. En la profunda, los RMSE de prueba fueron 28,41; 28,17 y 28,06 MW para 12, 24 y 48 horas. En la base, 12 horas fue mejor que 24 y 48. Por tanto, ampliar la ventana no garantiza mejorar: depende de la arquitectura y de los datos.')
sub('3. ¿Existe sobreajuste?')
p(f'Hay fluctuaciones y mejoras de validación que dejan de sostenerse, por lo que se aplicó parada temprana. Con los pesos restaurados, la seleccionada obtuvo RMSE {best.Train_RMSE:.2f} en entrenamiento, {best.Val_RMSE:.2f} en validación y {best.RMSE:.2f} MW en prueba. La brecha es moderada y no evidencia sobreajuste severo en esta partición. No basta para descartarlo definitivamente: se necesita repetir semillas y validación temporal en varios periodos.')
sub('4. ¿Qué es y qué efecto tiene Dropout?')
p('Durante entrenamiento, Dropout anula aleatoriamente una fracción de activaciones para reducir la dependencia excesiva de unas pocas unidades. Con tasa 0,2 se desactiva aproximadamente el 20 % en cada aplicación. Durante inferencia no se desactivan. Puede reducir sobreajuste, pero demasiado Dropout puede impedir aprender. Este experimento no aísla su efecto causal, porque las arquitecturas también difieren en otros componentes.')
sub('5. ¿Qué es y qué efecto tiene Early Stopping?')
p('Detiene el entrenamiento cuando la pérdida de validación deja de mejorar durante un número de épocas. Aquí la paciencia es 7 y se restauran los mejores pesos. La profunda de 48 horas se detuvo tras 27 épocas y volvió a los pesos de la época 20. Evita continuar ajustando cuando ya no mejora la validación y reduce tiempo; no garantiza por sí solo buena generalización.')
sub('6. ¿Más neuronas implican mejores resultados?')
p('No. Más neuronas aumentan capacidad, parámetros y costo, pero pueden aprender ruido o necesitar más datos. La base de 12 horas logra RMSE 28,42 MW con 6.177 parámetros; la profunda de 12 horas, 28,41 MW con 32.929. En ese caso, multiplicar la capacidad aporta una mejora mínima. También deben evaluarse latencia, estabilidad y costo de mantenimiento.')

page();h('Respuestas de análisis · 7 a 9')
sub('7. ¿Qué modelo recomendaría para producción y por qué?')
p(f'Como candidato para un piloto, seleccionaría la LSTM profunda de 48 horas: obtuvo el menor RMSE de validación ({best.Val_RMSE:.2f} MW) y mantiene un RMSE de prueba de {best.RMSE:.2f} MW. La elección se realizó antes de consultar prueba. La base de 12 horas sería una alternativa eficiente: tiene menos parámetros, requiere menos histórico y su error es cercano. La comparación de cobertura operativa entre ventanas queda pendiente, porque aquí todas se evaluaron sobre las mismas horas completas.')
p('No afirmaría que está lista para una red eléctrica real. Antes de producción verificaría origen y disponibilidad de variables, compararía con modelos adicionales, repetiría semillas, aplicaría validación temporal móvil y mediría latencia, deriva y cobertura con datos faltantes. La app entregada es una demostración académica.')
sub('8. ¿Qué variables pueden influir más en la demanda?')
p('Por conocimiento del problema, son plausibles la demanda previa, la hora, el tipo de día y la temperatura. Para esta ejecución se estimó sensibilidad en validación perturbando cada variable por bloques de 24 muestras, tres veces. Los primeros resultados fueron:')
table([['Variable','Aumento medio del RMSE (MW)'],*[[r.variable,f'{r.aumento_RMSE_medio:.2f}'] for r in imp.head(5).itertuples()]],[285,210])
p('Estos resultados describen la sensibilidad del modelo, no una relación causal. La hora aparece como dos componentes cíclicos; la correlación entre calendario, clima y demanda puede repartir o distorsionar la importancia. Los bloques son de muestras válidas y no necesariamente 24 horas continuas. La perturbación puede crear combinaciones poco realistas; se usa como análisis exploratorio.')
sub('9. ¿Qué limitaciones tiene el modelo?')
p('Solo se usan dos años de datos académicos, una partición y una semilla. La procedencia y geografía no están confirmadas. Los 40 conflictos de etiquetas quedan sin resolver con la fuente. Se preservan extremos sin prueba de error, que pueden afectar las entradas. El filtrado de ventanas reduce la cobertura a 42,28 % y puede sesgar las métricas. No hay intervalos de incertidumbre, ni verificación ante cambios estructurales, eventos extremos o fallos de sensores.')
p('El horizonte es exclusivamente la siguiente hora. La app exige el histórico completo y las variables contemporáneas de cada hora; no se ha comprobado su latencia de disponibilidad en un sistema real. Si las últimas horas tienen faltantes, el sistema rechaza la entrada en vez de inventar valores. Una estrategia causal de imputación tendría que validarse en un experimento adicional.')

page();h('Aplicación web, verificación y entrega')
sub('10. Aplicación implementada')
p('La app Streamlit carga el modelo seleccionado y los parámetros de escalado del entrenamiento. Permite cargar un CSV o editar una tabla con 48 horas históricas consecutivas. Calcula las variables de calendario y devuelve la demanda prevista en MW y su timestamp. No reentrena ni ajusta el escalador con los datos ingresados.')
p('Se incluyen un ejemplo real de entrada y controles para filas insuficientes, desorden, duplicados, huecos, valores no numéricos y límites físicos. Se verificó la carga del modelo guardado y la predicción mediante la aplicación. Los resultados y el notebook permiten revisar el experimento sin repetir entrenamiento.')
sub('11. Qué está comprobado y qué falta')
table([['Elemento','Estado'],['Nueve entrenamientos, métricas y gráficas','Completados con ejecución real.'],['Siete pruebas de preparación temporal','Aprobadas. Incluyen ausencia de fuga por cambios futuros.'],['Carga del modelo e interacción de predicción','Verificadas mediante prueba automatizada de Streamlit.'],['Despliegue en una URL pública','Pendiente: requiere publicar la app en una cuenta de alojamiento.']],[285,210])
p('Para uso local: crear un entorno Python 3.12, instalar requirements.txt y ejecutar <b>python -m streamlit run app.py</b>. Para alojamiento público, subir los archivos de la app, el modelo y el ejemplo a un repositorio propio; después seleccionar repositorio, rama y app.py en Streamlit Community Cloud y desplegar. El README contiene los pasos exactos. Guardar la URL y una captura como evidencia final del requisito de despliegue.')
sub('12. Reproducibilidad')
p('Entorno de ejecución: Python '+a['entorno']['python']+', TensorFlow '+a['entorno']['tensorflow']+', Keras '+a['entorno']['keras']+', NumPy '+a['entorno']['numpy']+' y pandas '+a['entorno']['pandas']+'. Se guardan configuración, semillas, historial, pesos de cada experimento, predicciones y hash del dataset. Puede existir variación numérica entre plataformas.','SmallCustom')
p('SHA-256 del CSV usado: '+a['dataset_sha256'],'SmallCustom')
sub('Fuentes y reconocimiento')
p('Enunciado: “Taller_lstm2026_2 (1).pdf”, aportado por el estudiante. Referencia externa reconocida: estructura de trabajo y dataset del repositorio de samumafu. El código, entrenamiento y análisis de esta entrega se prepararon de forma independiente.','SmallCustom')
link('Repositorio de referencia','https://github.com/samumafu/Ia-Taller-ltsm')
link('TensorFlow: LSTM','https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM')
link('TensorFlow: EarlyStopping','https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping')
link('Streamlit: despliegue','https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app')

def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#D6E0E8'));canvas.line(50,40,545,40)
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#617080'))
    canvas.drawString(50,27,'ANDERSON OJEDA · TALLER LSTM · UCC PASTO');canvas.drawRightString(545,27,str(doc.page))
path=R/'Informe_Taller_LSTM_Anderson.pdf'
doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=40,bottomMargin=53,title='Taller LSTM - Anderson Ojeda',author='Anderson David Ojeda Zambrano')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(path)
