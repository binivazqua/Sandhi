# Feedback Sandhi Interface — Junio 2026

**Para:** Interface Builders (RAs)  
**De:** Biniza Vázquez (PI)  
**Fecha:** 2 de junio de 2026  
**Script revisado:** `Sandhi_Interface_demo.py` (entregado 1 junio 2026)

---

Todas las implementaciones sugeridas fueron iteradas en la V2!

---

## Lo que está perfecto:

- `verify_eeg_stream()` perfectamente implementado desde el módulo `eeg_lsl_bridge.py`.
- todos los markers importados correctamente desde `eeg_lsl_bridge.py`.
- Fix de timing en markers de respuesta (`RESP_BUTTON`).
- Implementación de markers `RESP_LEVER_L` / `RESP_LEVER_R` con timing perfecto.
- ISI (randomization) entre trials.
- fix del bug `esp32.close()`

---

## Lo que hay que revisar

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

| Actual                    | Fix                                              |
| ------------------------- | ------------------------------------------------ |
| `'experiment_start'`      | `MARKERS.BLOCK_START`                            |
| `'experiment_end'`        | `MARKERS.BLOCK_END`                              |
| `'emociones_trial_start'` | — (no existe en Fase 01, revisar protocolo §3.3) |
| `'botones_trial_start'`   | — (ídem)                                         |

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

| #   | Fix                                      | Prioridad                                       |
| --- | ---------------------------------------- | ----------------------------------------------- |
| 1   | `verify_eeg_stream()` al inicio          | Alta — sin esto no hay datos EEG garantizados   |
| 2   | `RESP_BUTTON` / `RESP_LEVER` markers     | Alta — dato central del experimento             |
| 3   | Nombres de marcadores → clase `MARKERS`  | Alta — protocolo lo requiere exacto             |
| 4   | `callOnFlip` para marcadores de estímulo | Media — afecta precisión de timing              |
| 5   | ISI 1.0–3.5 s                            | Media — afecta validez del control experimental |
| 6   | Bug `esp32.close()` duplicado            | Media — crash en Windows                        |
| 7   | `seed=int(participant)`                  | Baja — decisión pendiente de PI                 |

---

Cualquier duda sobre la clase `MARKERS` o el archivo `eeg_lsl_bridge.py`, está todo documentado en `software/eeg_lsl_bridge.py` con comentarios inline.
