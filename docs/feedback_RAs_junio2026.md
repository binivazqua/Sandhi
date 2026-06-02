# Feedback Sandhi Interface — Junio 2026
**Para:** Interface Builders (RAs)  
**De:** Biniza Vázquez (PI)  
**Fecha:** 2 de junio de 2026  
**Script revisado:** `Sandhi_Interface_demo.py` (entregado 1 junio 2026)

---

Hicieron un buen trabajo construyendo la estructura base de los tres bloques de tarea y la integración serial con los dispositivos físicos. Lo siguiente es lo que necesita corrección antes de que podamos correr sesiones formales con EEG.

---

## Lo que está bien ✓

- Tres bloques de tarea (emociones, botones, palanca) funcionando
- Integración serial con ESP32 en COM3 y COM5
- Marcadores LSL en formato string (correcto para protocolo v1)
- Estructura de trials con CSV y TrialHandler

---

## Lo que hay que corregir

### 1. El experimento no verifica que el Muse esté transmitiendo

**Problema:** El script nunca checa si el Muse 2 está conectado y enviando EEG. Si BlueMuse o muselsl no están corriendo, el experimento corre igual — sin EEG, sin advertencia.

**Fix:** Al inicio del script (Before Experiment), agregar:
```python
from eeg_lsl_bridge import verify_eeg_stream, SandhiMarkerOutlet, MARKERS
verify_eeg_stream()          # aborta si no hay stream EEG
marker_outlet = SandhiMarkerOutlet()
```

El archivo `eeg_lsl_bridge.py` ya está en `software/`. Solo hay que importarlo.

---

### 2. Los nombres de los marcadores no coinciden con el protocolo

**Problema:** El protocolo dice exactamente qué string debe ir en cada evento. Lo que tienen ahora es diferente:

| Lo que tienen | Lo que debe ser |
|---|---|
| `'experiment_start'` | `MARKERS.BLOCK_START` |
| `'experiment_end'` | `MARKERS.BLOCK_END` |
| `'emociones_trial_start'` | — (no existe en Fase 01, revisar protocolo §3.3) |
| `'botones_trial_start'` | — (ídem) |

Usar siempre la clase `MARKERS` del archivo `eeg_lsl_bridge.py`. No escribir strings manualmente.

---

### 3. Falta el marcador de respuesta del participante

**Problema:** Cuando el participante presiona el botón, no se emite ningún marcador. Esto es el dato más importante para la sincronización EEG — sin él, no podemos alinear la señal cerebral con la respuesta.

**Fix:** En el momento exacto en que se detecta el press (dentro del loop de serial), agregar:
```python
marker_outlet.push(MARKERS.RESP_BUTTON)
```

Lo mismo para la palanca:
```python
marker_outlet.push(MARKERS.RESP_LEVER_L)   # o RESP_LEVER_R según dirección
```

---

### 4. Los marcadores de estímulo deben emitirse al flip de pantalla

**Problema:** Ahora los marcadores van en `Begin Routine`, antes de que la pantalla realmente cambie. Eso introduce error de timing de hasta ~16 ms. El protocolo requiere < 10 ms.

**Fix:** Usar `callOnFlip` en el frame donde aparece el estímulo:
```python
win.callOnFlip(marker_outlet.push, MARKERS.STIM_GO)
```

En PsychoPy Builder, esto va en la pestaña **Each Frame**, en el primer frame del componente de estímulo (cuando `frameN == 0`).

---

### 5. No hay pausa entre trials (ISI)

**Problema:** Los trials se suceden inmediatamente. El protocolo §3.4 requiere una pausa aleatoria de 1.0 a 3.5 s entre trials para control experimental.

**Fix:** Al final de cada trial (End Routine), agregar un componente de blank screen con duración:
```python
import random
isi_duration = random.uniform(1.0, 3.5)
```

En Builder: agregar un componente `Polygon` (negro, tamaño de pantalla) o un componente `Text` vacío con esa duración.

---

### 6. Bug: `esp32.close()` se llama dos veces al cierre

**Ubicación:** Líneas 3015–3017 del demo.

**Problema:** Cerrar el mismo puerto serial dos veces causa un crash en Windows. Es un error de copy-paste.

**Fix:** Dejar solo una llamada a `esp32.close()` y una a `esp32_1.close()`.

---

### 7. El orden de trials no es reproducible

**Problema:** `TrialHandler2` usa `seed=None`, así que el orden cambia en cada sesión. Si necesitamos contrabalancear o reproducir una sesión, no podemos.

**Fix (cuando PI confirme):** Cambiar a:
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

*Revisión PI — 2 junio 2026*
