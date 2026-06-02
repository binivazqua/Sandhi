# Sandhi Alpha 01 — Revisión de Código: Historial de Análisis
**PI:** Biniza Vázquez  
**Fecha de revisión:** 2 de junio de 2026  
**Scope:** `software/` — Interfaz PsychoPy + pipeline EEG/LSL  
**Protocolo de referencia:** Sandhi Alpha 01 Protocol v1  

---

## 1. Contexto de la revisión

Esta revisión cubre el estado del código entregado por los RAs al cierre de la semana del 28 de mayo – 1 de junio de 2026. Se analizaron dos versiones del script de interfaz PsychoPy y la ausencia de integración con el pipeline EEG.

Archivos analizados:

| Archivo | Fecha | Estado |
|---|---|---|
| `Sandi_Interface_All_Trials_lastrun.py` | 28 mayo 2026 | En repo (versión anterior) |
| `Sandhi_Interface_demo.py` | 1 junio 2026 | Entregado por RAs, no en repo |
| `eeg_lsl_bridge.py` | 2 junio 2026 | Creado por PI, en repo |
| `run_experiment.py` | 2 junio 2026 | Creado por PI, en repo |

---

## 2. Hallazgos por área

### 2.1 Conectividad Muse 2 / LSL

**Hallazgo crítico:** Ninguna de las dos versiones del script contiene código para conectar con el Muse 2. No existe ninguna llamada a `muselsl`, `BlueMuse`, ni `resolve_stream()`.

El script solo crea un `StreamOutlet` de marcadores (líneas 39–50 del demo). La adquisición EEG es completamente externa y asumida, no verificada. Si el operador no inicia BlueMuse o `muselsl stream` antes de correr PsychoPy, el experimento corre sin EEG y no hay ninguna advertencia en el código.

**Implicación para el protocolo:** El criterio de éxito de Fase 01 §3.2 —*"verify_stream() confirma EEG activo antes de abrir la ventana"*— no se cumple en ninguna de las dos versiones entregadas.

**Acción tomada por PI:** Se creó `eeg_lsl_bridge.py` con `verify_eeg_stream()` (aborta si no hay stream EEG) y `start_muse_stream()` (lanza muselsl en daemon thread). Se creó `run_experiment.py` como launcher que enforcea este chequeo antes de abrir PsychoPy.

---

### 2.2 Marcadores LSL

**Versión anterior (`_lastrun.py`):**
- Tipo: `string` ✓
- Nombres: `experiment_start`, `emociones_trial_start`, `botones_trial_start`, etc.
- No coinciden con los códigos del protocolo §3.3

**Demo (`Sandhi_Interface_demo.py`):**
- Tipo: `string` ✓
- `push_sample(['experiment_start'])` en línea 421 — nombre incorrecto, debería ser `BLOCK_START`
- Ausencia total de `RESP_BUTTON` al momento del press — crítico para la sincronización EEG-comportamiento
- Ausencia total de `RESP_LEVER_L` / `RESP_LEVER_R` al momento del movimiento de palanca
- `STIM_NOGO` nunca emitido (tarea de Fase 02 no implementada aún, pero el stream ya existe)

**Acción tomada por PI:** Se creó la clase `MARKERS` en `eeg_lsl_bridge.py` con los 13 códigos string del protocolo §3.3/§4.4/§5.3, carácter por carácter.

---

### 2.3 Timing de marcadores

El momento correcto para emitir un marcador de estímulo visual es al flip de pantalla, usando:
```python
win.callOnFlip(marker_outlet.push, MARKERS.STIM_GO)
```

En ambas versiones, los marcadores se emiten **antes** del flip, en el bloque `Begin Routine`. Esto introduce un jitter sistemático de entre 1 y 16 ms dependiendo del frame rate. El protocolo §3.2 fija el criterio de jitter en < 10 ms.

**Estado:** No corregido en el código de los RAs. Pendiente de instrucción específica para corregir en PsychoPy Builder.

---

### 2.4 ISI (Inter-Stimulus Interval)

El protocolo §3.4 requiere un ISI aleatorio uniforme de 1.0–3.5 s entre trials para control Pavloviano.

**Hallazgo:** Ninguna de las dos versiones implementa ISI. Los trials se suceden de forma inmediata tras el fin del anterior.

**Estado:** GAP documentado. Pendiente de implementación en Builder.

---

### 2.5 Hardware serial (botones y palanca)

**Demo `Sandhi_Interface_demo.py`:**
- `COM3` (ESP32 botones): `serial.Serial('COM3', 115200)` — línea 415
- `COM5` (joystick/palanca): `serial.Serial('COM5', 115200)` — línea 418
- Bug crítico: `esp32.close()` se llama dos veces al cierre (líneas 3015 y 3017) — crash en Windows

**Versión anterior `_lastrun.py`:**
- Solo `COM3`. No hay COM5 (palanca no integrada en esa versión).

**Estado:** El bug de `esp32.close()` duplicado está documentado para los RAs. No corregido aún en el repo porque el demo no ha sido incorporado oficialmente.

---

### 2.6 Reproducibilidad de trials

`TrialHandler2` se llama con `seed=None` en los tres bloques de tarea. El orden de trials no es reproducible entre sesiones. Para contrabalanceo o análisis de orden, se debe pasar:
```python
seed=int(expInfo['participant'])
```

**Estado:** GAP documentado. Decisión de diseño experimental pendiente de PI.

---

### 2.7 Modo piloto

Todos los datos recolectados hasta la fecha se generaron con `PILOTING = True` (auto-set por PsychoPy Builder cuando se corre desde la GUI). Los archivos de datos se guardan en la carpeta `pilot/` y no en `data/`. Para sesiones formales, usar el launcher:
```bash
python run_experiment.py
```

---

## 3. GAPs abiertos por responsable

| ID | Descripción | Responsable | Estado |
|---|---|---|---|
| GAP-01 | Unificar banda beta a 13–20 Hz en análisis EEG | PI | Pendiente |
| GAP-02 | Verificar orden de canales TP9/AF7/AF8/TP10 empíricamente | Interface Builders | Pendiente |
| GAP-03 | `win.callOnFlip()` para marcadores de estímulo | Interface Builders | Pendiente |
| GAP-04 | ISI 1.0–3.5 s entre trials | Interface Builders | Pendiente |
| GAP-05 | `RESP_BUTTON` al momento del press | Interface Builders | Pendiente |
| GAP-06 | `RESP_LEVER_L` / `RESP_LEVER_R` al momento del movimiento | Interface Builders | Pendiente |
| GAP-07 | `seed=int(expInfo['participant'])` para reproducibilidad | PI (decisión) | Pendiente |
| GAP-08 | Bug `esp32.close()` duplicado | Interface Builders | Pendiente |
| GAP-09 | Nombres de marcadores migrados a clase `MARKERS` | Interface Builders | Pendiente |

---

## 4. Archivos creados por PI (en repo)

### `software/eeg_lsl_bridge.py`
Módulo de integración EEG/LSL. Provee:
- `MARKERS`: clase con los 13 códigos string del protocolo
- `verify_eeg_stream(timeout)`: verifica que el stream Muse está activo
- `start_muse_stream(address)`: lanza muselsl en daemon thread (macOS/Linux)
- `SandhiMarkerOutlet`: StreamOutlet preconfigrado para el experimento
- Self-test ejecutable: `python eeg_lsl_bridge.py`

### `software/Sandhi_Demo/run_experiment.py`
Launcher de pre-vuelo. Reemplaza la ejecución directa de `_lastrun.py`. Pasos:
1. Verifica stream EEG (o lanza muselsl con `--start-muselsl`)
2. Pide confirmación de que LabRecorder está grabando
3. Lanza el script PsychoPy via subprocess

Flags:
```
--skip-eeg-check       (debug sin Muse conectado)
--start-muselsl        (lanza muselsl automáticamente)
--muse-address XX:XX   (MAC address del Muse)
```

### `software/muse`, `software/labrecorder`, `software/psychopy`
Archivos de setup e instrucciones operacionales para el equipo.

---

## 5. Resumen ejecutivo

El equipo de RAs construyó una interfaz PsychoPy funcional para los tres bloques de tarea (emociones, botones, palanca), con integración serial para el hardware físico. Sin embargo, la integración con la adquisición EEG es inexistente en el código: el Muse 2 se asume como externo y no verificado. Los marcadores LSL existen pero no coinciden con el protocolo v1 ni se emiten en el momento correcto del ciclo de pantalla. Los GAPs críticos para Fase 01 (verify_stream, callOnFlip, RESP_BUTTON/LEVER, ISI) deben cerrarse antes de la primera sesión formal de datos.

---

*Documento generado por revisión de código PI — sesión del 2 junio 2026*
