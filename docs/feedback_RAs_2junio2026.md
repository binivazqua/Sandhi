# Feedback Sandhi Interface — Junio 2026
**Para:** Interface Builders (RAs)  
**De:** Biniza Vázquez (PI)  
**Fecha:** 2 de junio de 2026  
**Script revisado:** `Sandhi_Interface_demo.py` (entregado 1 junio 2026)

---

Hicieron un buen trabajo construyendo la estructura base de los tres bloques de tarea y la integración serial con los dispositivos físicos :)) . Lo siguiente es lo que necesita corrección antes de que podamos correr sesiones formales con EEG.

---

## Lo que está perfecto:

- Tres bloques de tarea (emociones, botones, palanca) funcionando
  - Me agrada la idea del slider con emociones, ¿cómo se les ocurrió? ¿es una prueba? ¿cómo piensan integrarlo?
- Integración serial con ESP32 en COM3 y COM5
- Marcadores LSL en formato string (correcto para protocolo v1, prueba local entre nosotros).
- Estructura de trials con CSV y TrialHandler

---

## Lo que hay que revisar

> **Nota de iteración:** Los puntos 1–6 del feedback anterior están resueltos en V2 :)) — buen trabajo. Lo que sigue son los pendientes para la siguiente iteración, basados en el protocolo Sandhi Alpha 01 v1.

---

### 1. Verificación del stream de Muse

**Problema:** El script nunca checa si el Muse 2 está conectado y enviando EEG. Si BlueMuse o muselsl no están corriendo, el experimento corre igual. En la práctica, por la falla constante en el stream, podríamos olvidar conectar o incluso perder datos. 

**Fix Sugerido:** Al inicio del script (Before Experiment), agregar:
```python
from eeg_lsl_bridge import verify_eeg_stream, SandhiMarkerOutlet, MARKERS
verify_eeg_stream()          # aborta si no hay stream EEG
marker_outlet = SandhiMarkerOutlet()
```

El archivo `eeg_lsl_bridge.py` ya está en `software/`. Está hecho con base en toda la arquitectura, hay que probar si funciona :DD.

---

### 2. Añadir nombres acordados para markers

**Problema:** En el protocolo, en la sección de Trial 01 acordamos las strings que debe ir en cada evento (sólo como buena práctica). 

| Actual | Fix |
|---|---|
| `'experiment_start'` | `MARKERS.BLOCK_START` |
| `'experiment_end'` | `MARKERS.BLOCK_END` |
| `'emociones_trial_start'` | — (no existe en Fase 01, revisar protocolo §3.3) |
| `'botones_trial_start'` | — (ídem) |

Usar siempre la clase `MARKERS` del archivo `eeg_lsl_bridge.py`. Están como cosntantes al inicio :) 

---

### 3. Marker de respuesta

**Problema:** Cuando el participante presiona el botón, no se emite ningún marcador. Esto es el dato más importante para la sincronización del EEG, para alinear la señal cerebral con la respuesta.

**Fix:** En el momento exacto en que se detecta el press (dentro del loop de serial), agregar:
```python
marker_outlet.push(MARKERS.RESP_BUTTON)
```

Lo mismo para la palanca:
```python
marker_outlet.push(MARKERS.RESP_LEVER_L)   # o RESP_LEVER_R según dirección
```

---

### 4. Markers al momento de flip.

**Posible problema:** Ahora los marcadores van en `Begin Routine`, antes de que la pantalla realmente cambie. Eso introduce error de timing de hasta approx 16 ms. El protocolo requiere < 10 ms.

**Fix:** Usar `callOnFlip` (está en docs, pero hay que checar si sí funciona) en el frame donde aparece el estímulo:
```python
win.callOnFlip(marker_outlet.push, MARKERS.STIM_GO)
```

En PsychoPy Builder, esto va en la pestaña **Each Frame**, en el primer frame del componente de estímulo (cuando `frameN == 0`).

---

### 5. Pausa entre trials (ISI)

**Problema:** Los trials se suceden inmediatamente. El protocolo requiere una pausa aleatoria de 1.0 a 3.5 s entre trials para control experimental.

**Fix:** Al final de cada trial (End Routine), agregar una blank screen con duración:
```python
import random
isi_duration = random.uniform(1.0, 3.5)
```

En Builder: agregar un componente `Polygon` (negro, tamaño de pantalla) o un componente `Text` vacío con esa duración.

---

### 6. Bug: `esp32.close()` se llama dos veces al cierre (creo que es un typo, pero puede crashear)

**Ubicación:** Líneas 3015–3017 del demo.

**Problema:** Cerrar el mismo puerto serial dos veces puede hacer crash en windows. 

**Fix:** Dejar solo una llamada a `esp32.close()` y una a `esp32_1.close()`.

---

### 7. Orden de trials 

**Problema:** `TrialHandler2` usa `seed=None`, así que el orden cambia en cada sesión. Si necesitamos reproducir una sesión de nuevo, no sería del todo correcto.

**Fix (no urgente):** Cambiar a:
```python
seed=int(expInfo['participant'])
```

Así cada participante siempre recibe el mismo orden.

---

## Cómo probar que los fixes funcionan

1. Instalar dependencias: `pip install pylsl muselsl`
2. Correr el self-test del bridge: `python software/eeg_lsl_bridge.py`
   - Debe encontrar el stream EEG del Muse y enviar 4 marcadores de prueba
3. Abrir LabRecorder, verificar que aparecen dos streams: `type=EEG` y `type=Markers`
4. Correr el experimento con `python run_experiment.py`
5. Al terminar, abrir el .xdf con pyxdf y verificar que los marcadores tienen los nombres correctos y aparecen alineados con el EEG

---

## Prioridad de correcciones

| # | Fix | Prioridad |
|---|---|---|
| 1 | `verify_eeg_stream()` al inicio | Alta — sin esto no hay datos EEG garantizados |
| 2 | `RESP_BUTTON` / `RESP_LEVER` markers | Alta — dato central del experimento |
| 3 | Nombres de marcadores → clase `MARKERS` | Alta — protocolo lo requiere exacto |
| 4 | `callOnFlip` para marcadores de estímulo | Media — afecta precisión de timing |
| 5 | ISI 1.0–3.5 s | Media — afecta validez del control experimental |
| 6 | Bug `esp32.close()` duplicado | Media — crash en Windows |
| 7 | `seed=int(participant)` | Baja — decisión pendiente de PI |

---

Cualquier duda sobre la clase `MARKERS` o el archivo `eeg_lsl_bridge.py`, está todo documentado en `software/eeg_lsl_bridge.py` con comentarios inline.

---

## Pendientes para V3 — basados en Protocolo Sandhi Alpha 01 v1

### 8. La tarea de emociones no tiene marcador de onset de estímulo

**Protocolo §3.3:** Cada estímulo visual que queramos epochar en el EEG necesita su marcador en el momento exacto del flip. La tabla de marcadores de Fase 01 incluye `STIM_GO` como bandera del onset visual — sin él no podemos alinear el EEG con la presentación de la imagen.

**Problema concreto:** En la tarea de emociones, la imagen aparece en pantalla pero no se emite ningún marcador. Si en Fase 02 queremos ver qué pasa en AF7/AF8 cuando aparece un estímulo emocional, no tenemos el timestamp con qué epochar.

**Fix:** Agregar `callOnFlip` en el primer frame de la imagen de emoción, igual que lo hicieron para la tarea de botones:
```python
if imagen_emocion.status == STARTED and not marker_enviado:
    win.callOnFlip(marker_outlet.push, MARKERS.STIM_GO)
    marker_enviado = True
```

---

### 9. La lógica FTI mide tiempo desde inicio del trial, no desde onset del estímulo

**Protocolo §4.3:** La ventana FTI es de **200 ms exactos post-onset del estímulo No-Go** (Gavenas et al., 2025). Cualquier presión dentro de esa ventana es un impulso balístico que ya pasó el punto de no retorno.

**Problema concreto:** En V2, `feedbackStart = t` captura el tiempo local del frame de respuesta, y se compara directamente con `0.200`. Esto mide el tiempo desde el inicio del trial completo, no desde el onset del estímulo. Si hay instrucciones o cuenta regresiva antes del estímulo, la ventana queda mal calibrada.

**Fix:** Guardar el timestamp del onset del estímulo y calcular la latencia de respuesta desde ahí:
```python
# Al onset del estímulo (callOnFlip o primer frame):
stim_onset_t = t

# Al recibir respuesta:
resp_latency = t - stim_onset_t
if resp_latency < 0.200:
    marker_outlet.push(MARKERS.FTI_BALLISTIC_ERROR)
else:
    marker_outlet.push(MARKERS.CONTROLLED_RESPONSE)
```
Esto es crítico — la señal "oro" del experimento (FTI) depende de que esta ventana esté bien definida. Ver protocolo §4.3 y la referencia Gavenas et al. (2025).

---

### 10. Ratio Go/No-Go no está definido en los CSVs (GAP-03)

**Protocolo §4.5 / GAP-03:** La condición No-Go debe ser **minoritaria** para que la respuesta Go sea prepotente — típicamente 70/30 o 80/20. Si hay demasiados No-Go, el sujeto deja de tener un sesgo Go y el control inhibitorio pierde validez como medida.

**Problema concreto:** No sé cuántos trials Go vs. No-Go tienen `Trial_1.csv` y `Trial_2.csv`. Hay que revisarlo y justificar la proporción elegida — esta decisión bloquea Fase 02.

**Fix (coordinación conmigo):** Antes de definir el número de trials, calculen cuántos FTI esperados hay con el ratio elegido. Con 20–40 trials totales y 30% No-Go, esperamos ~6–12 No-Go, de los cuales solo una fracción serán FTI. Puede ser estadísticamente insuficiente (GAP-04). Revisen y me reportan la proporción actual.

---

### 11. La tarea de emociones puede ser una variable Pavloviana (§6.2)

**Protocolo §6.2 — Control de ruido cognitivo:** Las imágenes de caras y expresiones emocionales son "pistas sociales" que el protocolo identifica explícitamente como fuente de fluctuación en banda alfa (8–14 Hz), confundible con el veto voluntario.

**Problema de diseño:** Si registramos EEG durante la tarea de emociones y la usamos en el análisis, las caras pueden "secuestrar" la señal y falsear el pre-impulso. El protocolo requiere entorno visualmente neutro y estímulos Pavlovianos fuera de la línea de visión.

**No es un fix de código — es una decisión de diseño experimental que coordinaremos.** Por ahora documéntenlo como limitación conocida en el script con un comentario:
```python
# NOTA PROTOCOLO §6.2: La tarea de emociones usa imágenes de caras.
# Estas son variables Pavlovianas (fluctuación alfa). Registrar EEG
# en este bloque requiere análisis separado y no debe mezclarse con
# los epochs de la tarea Go/No-Go.
```

---

### 12. Tarea de palanca — solo emite STIM_GO, falta lógica direccional

**Protocolo §4.2:** La tarea de palanca (Sistema 2) tiene su propia lógica de selección de acción: el sujeto usa la mano indicada y orienta la palanca hacia la dirección mostrada. La lateralización importa — `RESP_LEVER_R` debería correlacionar con drop de potencia beta mayor en AF7 (corteza contralateral, Lui et al., 2021).

**Problema concreto:** El marcador `STIM_GO` se emite para cualquier onset de instrucción de palanca, pero no hay diferenciación entre Go y No-Go en esta tarea, ni entre dirección izquierda y derecha al nivel del estímulo. Solo se diferencian en la respuesta (`RESP_LEVER_L` vs `RESP_LEVER_R`).

**Fix (a coordinar):** Definir si la tarea de palanca tiene condición No-Go. Si la tiene, agregar `STIM_NOGO` para los trials donde el estímulo indica "no mover". Si no la tiene en Fase 01, documentarlo explícitamente y reservarlo para Fase 02.

---

### 13. seed=None — el orden de trials sigue sin ser reproducible

Esto ya estaba en el feedback anterior. Mientras no definamos el protocolo de contrabalanceo, por favor cambien a:
```python
seed=int(expInfo['participant'])
```
Así podemos reproducir exactamente la sesión de cualquier participante si necesitamos debuggear. Sin seed fijo, cualquier bug de orden de trials es irrepetible.

---

## Resumen de prioridades V3

| # | Pendiente | Prioridad | Bloquea |
|---|---|---|---|
| 9 | FTI — latencia desde onset del estímulo | **Alta** | Señal central del experimento |
| 8 | Marcador onset tarea emociones | **Alta** | Epoching Fase 02 |
| 13 | seed reproducible | **Media** | Debuggeo y replicabilidad |
| 10 | Ratio Go/No-Go en CSVs | **Media** | Diseño Fase 02 (GAP-03) |
| 12 | Lógica direccional palanca | **Media** | Lateralización Fase 02 |
| 11 | Emociones como variable Pavloviana | **Baja** | Decisión de diseño (PI) |


