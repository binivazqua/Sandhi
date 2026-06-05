# Feedback Sandhi Interface — Junio 2026

**Para:** Interface Builders (RAs)  
**De:** Biniza Vázquez (PI)  
**Fecha:** 2 de junio de 2026  
**Script revisado:** `Sandhi_Interface_demo.py` (entregado 1 junio 2026)

---

Todas las implementaciones sugeridas fueron iteradas en la V2!

---

## Lo que está perfecto:

- `verify_eeg_stream()` bien importado desde el módulo `eeg_lsl_bridge.py`.
- todos los markers importados correctamente desde `eeg_lsl_bridge.py`.
- Fix de timing en markers de respuesta (`RESP_BUTTON`).
- Implementación de markers `RESP_LEVER_L` / `RESP_LEVER_R` con timing perfecto.
- ISI (randomization) entre trials.
- fix del bug `esp32.close()`

---

## Lo que hay que revisar — V3

> Scope estrictamente Fase 01: fidelidad de la señal y condiciones del sujeto. Nada de Fase 02 todavía.

---

### 1. El contador visible en pantalla genera CNV — hay que quitarlo

El protocolo es explícito en esto: *"No debe contar los segundos mentalmente. El conteo activaría procesos cognitivos adicionales que generarían una señal de interferencia (CNV) en lugar del RP puro."*

`Cuenta_tarea1` y `Cuenta_tarea2` muestran un número contando en pantalla durante el trial. Si el sujeto ve eso, cuenta aunque no quiera. La CNV (Contingent Negative Variation) es justamente la señal que contamina el RP que queremos capturar.

**Fix:** Eliminar los componentes `Cuenta_tarea1` y `Cuenta_tarea2` de las rutinas de trial. Si necesitan un timer para el operador, que sea en la consola (`print()`), no en la ventana del sujeto.

---

### 2. El ISI puede ser menor al mínimo del protocolo

El protocolo dice: *"Debe esperar un mínimo de 3 segundos entre cada pulsación para que la actividad motora previa se disipe y la señal regrese a su línea base."*

El ISI actual es `random.uniform(1.0, 3.5)` — puede salir de 1 segundo. Con 1–2 s entre trials, el EEG del trial anterior todavía no se limpió cuando empieza el siguiente, y las épocas se solapan.

**Fix:**
```python
isi_duration = random.uniform(3.0, 5.0)
```
El mínimo 3 s es hard, no negociable. El máximo de 5 s da suficiente variabilidad para que el sujeto no pueda predecir el onset del siguiente estímulo (control Pavloviano).

---

### 3. No hay cruz de fijación durante los trials

El protocolo pide: *"Fijación Visual: mantener la mirada fija en la cruz de la pantalla para evitar movimientos oculares innecesarios."*

No existe ninguna cruz en las rutinas de trial de botones ni de palanca. Los movimientos de ojos generan artefactos EOG que caen directo en AF7/AF8 — los electrodos frontales que usamos para medir la desincronización beta. Sin cruz, no podemos controlar eso.

**Fix:** Agregar un `TextStim` con `'+'` centrado, color blanco, visible durante todo el trial (desde ISI hasta respuesta). Es una línea en Builder.

---

### 4. Los botones amarillo y rojo están siempre en pantalla — son pistas Pavlovianas de color

Los colores son variables Pavlovianas que causan desincronización beta (12–30 Hz) antes de que haya cualquier acción. Si el sujeto ve el botón amarillo y el rojo todo el tiempo, el cerebro ya está procesando el color aunque no haya llegado el estímulo — eso introduce ruido beta que no corresponde al momento de la acción.

En V2, `Boton_Amarillo` y `Boton_Rojo` están dibujados permanentemente durante el trial, no solo cuando aparece el color estímulo. El sujeto ve ambos colores a la vez, siempre.

**Fix:** Mostrar las imágenes de botones solo al onset del estímulo (mismo frame que `callOnFlip` del marcador), y ocultarlas durante el ISI. Antes del estímulo: pantalla negra + cruz de fijación solamente.

---

### 5. Las instrucciones no comunican espontaneidad ni postura correcta

Las instrucciones actuales dicen *"presiona el botón del color que aparece en pantalla, tienes 1.5s"*. Eso describe una tarea de reacción rápida, no una tarea de acción espontánea. El sujeto va a prepararse mentalmente para reaccionar, lo cual genera exactamente el tipo de señal de esfuerzo deliberado que queremos evitar.

El protocolo requiere: espontaneidad absoluta, sin planeación previa, dedo relajado en contacto, sin cerrar los ojos.

**Fix — wording sugerido:**

```
• Mantén el dedo apoyado suavemente sobre el botón todo el tiempo.
• Cuando aparezca el color en pantalla, presiona solo si coincide con tu botón.
  Si el color no coincide, no hagas nada.
• Hazlo en el momento exacto en que sientas el impulso — sin anticipar ni contar.
• Mantén la mirada en la cruz del centro de la pantalla.
• No cierres los ojos durante la tarea.
```

El cambio de tono importa: de "tienes X segundos para hacer Y" a "espera el impulso y reacciona".

---

## Prioridad de correcciones V3

| # | Fix | Por qué importa en Fase 01 |
|---|---|---|
| 1 | Quitar countdown de pantalla | CNV contamina el RP — es la señal que queremos capturar |
| 2 | ISI mínimo 3 s | Solapamiento de épocas EEG entre trials |
| 3 | Cruz de fijación | Artefactos EOG en AF7/AF8 |
| 4 | Ocultar botones durante ISI | Pista Pavloviana de color genera ruido beta falso |
| 5 | Rewording instrucciones | El sujeto no entiende que debe ser espontáneo |

---

Cualquier duda, está todo documentado en `software/eeg_lsl_bridge.py` y en el protocolo Sandhi Alpha 01 v1.
