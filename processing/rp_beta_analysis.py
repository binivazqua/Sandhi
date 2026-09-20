
"""
Readiness Potential / Beta-ERD analysis pipeline for Sandhi Alpha 01 Phase!
=======================================================================
Hardware:  Muse 2 (AF7, AF8, TP9, TP10 @ 256 Hz) ->  RIGHT NOW!
           Unicorn Hybrid Black — channel-agnostic, detected at runtime -> TO COME!
Markers:   ON THIS PHASE: STIM_GO, STIM_NOGO, RESP_BUTTON, RESP_LEVER_L, RESP_LEVER_R
           BLOCK_START, BLOCK_END
"""

import argparse
import os

import numpy as np
import pandas as pd
import mne
import pyxdf
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from rich.console import Console
from rich.table import Table

import pyxdf
import numpy as np
import matplotlib.pyplot as plt

XDF = 'data/sandhi_beta/sub_Debbie_run021_eeg.xdf' # replace WITH FILE PATH TO ANALYZE

# FIRST: LOAD ALL XDF STREAMS BY PSYCHOPY ---
# pyxdf devuelve una LISTA de streams + un header global (que ignoramos con _)
streams, _ = pyxdf.load_xdf(XDF)

# THEN: DIVIDE EEG AND PSYCHOPY LSL MARKERS BY CONTENT, NOT POS ---

eeg = next(s for s in streams if s['info']['type'][0].lower() == 'eeg')
mrk = next(s for s in streams if s['info']['type'][0].lower() == 'markers')

# NEXT: NO HARD CODED CHANNELS, BUT ACTUAL PARSING ---
ch_names = [ch['label'][0] for ch in eeg['info']['desc'][0]['channels'][0]['channel']]
# → ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']

# OBTAIN EEG MATRI USING PYXDF ---
# pyxdf la da como (n_samples, n_channels); la transponemos con .T
# para tener (n_channels, n_samples), que es como MNE la espera
data  = np.array(eeg['time_series']).T
ts    = np.array(eeg['time_stamps'])          # timestamp LSL de cada muestra
sfreq = float(eeg['info']['nominal_srate'][0]) # donde nominal_srate es la sample rate del device (256 hz para muse 2) 
# el [0] es para acceder al único elemento de la lista nominal_srate, que es un string (por eso float())

# FINALLY: TARGET JUST AF7 AND AF8 (PHENOMENIC CONSCIOUS INTENTION!!!) ---
keep = ['AF7', 'AF8']
idx  = [ch_names.index(name) for name in keep]   # → [1, 2]
data_keep = data[idx, :]                         # solo esos 2 canales

# bug 1 FIX: ch_names y data_keep deben coincidir.
# ch_names = keep
# aun más anti todo:
ch_names = [ch_names[i] for i in idx] # esto es redundante pero asegura que ch_names solo tenga los canales que estamos usando


# BLOQUE 2: FILTERING AS PREPROCESSING TO ISOLATE LOW BETA.

def preprocess(data, sfreq, ch_names=None, verbose=True):
    """
    Filtrado de la señal EEG.
    Decisiones CLAVE:
      1. Notch 60 Hz --> red eléctrica de México (NO 50 Hz europeo)
      2. Pasa-banda 12-20 Hz --> low-beta pero basado en (Gavenas et al. 2025)
    Orden: notch primero, luego pasa-banda.
    Ambos filtros son de FASE CERO (filtfilt / sosfiltfilt).
    Se cancela el phase delay para preservar la temporalidad lograda por RAs.

    log: 
        Si verbose=True, imprime el % de energía retenida por canal
        como criterio de calidad del filtrado (esperado: bajo, ~5-15%,
        porque la mayor parte de la energía EEG vive fuera de low-beta).
    """

    # S1: Notch 60 Hz (ruido de red eléctrica) ---
    # w0=60: frecuencia a eliminar | Q=30: qué tan angosto el notch
    b_notch, a_notch = iirnotch(w0=60, Q=30, fs=sfreq)
    data_notch = filtfilt(b_notch, a_notch, data, axis=-1)

    # S2: Pasa-banda 12-20 Hz (low-beta) ---
    # N=4: orden | Wn=[12,20]: banda | output='sos': estable
    sos_bp = butter(N=4, Wn=[12, 20], btype='band', fs=sfreq, output='sos')
    data_filt = sosfiltfilt(sos_bp, data_notch, axis=-1)

    # LOG DE ANÁLISIS:

    if verbose:
        if ch_names is None:
            ch_names = [f"ch{i}" for i in range(data.shape[0])]
        print(f"[preprocess] notch 60 Hz + pasa-banda 13-20 Hz | fase cero")
        for i, ch in enumerate(ch_names):
            rms_raw  = np.sqrt(np.mean(data[i]**2))
            rms_filt = np.sqrt(np.mean(data_filt[i]**2))
            pct = 100 * rms_filt / rms_raw if rms_raw > 0 else 0
            print(f"[preprocess]   {ch}: {pct:5.1f}% energía retenida "
                  f"(RMS {rms_raw:.1f} → {rms_filt:.1f} µV)")
    return data_filt

# debug 1: 

# print("data tiene", data_keep.shape[0], "canales")
# print("ch_names tiene", len(ch_names), "nombres:", ch_names)

# CALL PREPROCESSING --- (vamos en orden, pero esto se puede optimizar luego)
data_preprocessed = preprocess(data_keep, sfreq, ch_names=ch_names)

# BLOCK 3: EPOCHING 
# Checklist: 
# 1. Ya cargamos xdf,
# 2. Ya filtramos -> data_preprocessed = (2, 84096)
# Por hacer: convertir a obj MNE para hacer el epoching.abs

# FIRST: CREATE MNE INFO OBJECT ---
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
raw  = mne.io.RawArray(data_preprocessed * 1e-6, info) # converto from picoVolts to Volts (MNE espera V)

# THEN: EXTRACT ONLY TARGETED MARKERS AND CREATE MNE EVENTS ---
# WE WORK WITH MARKERS' ARRAY "mrk"
labels = [m[0] for m in mrk['time_series']]
mrk_ts = np.array(mrk['time_stamps'])

button_ts     = mrk_ts[[i for i, l in enumerate(labels) if l == 'RESP_BUTTON']]
onsets_rel    = button_ts - ts[0]                  # removemos el offset del timestamp LSL para tener onsets relativos al inicio de la grabación
onset_samples = (onsets_rel * sfreq).astype(int)    # multiplicamos por la sampling rate para convertir de segundos a muestras

# MNE exige eventos en formato (n, 3): [muestra, valor_previo, id_evento].
# Solo la col 1 (muestra) lleva información real; col 2 = relleno, por mutuo acuerdo es (0),
# col 3 = id del tipo de evento (1 = RESP_BUTTON, definido en event_id).
events   = np.column_stack([onset_samples,
                            np.zeros(len(onset_samples), int),
                            np.ones(len(onset_samples), int)])
event_id = {'RESP_BUTTON': 1}

# FINALLY: EPOCHING CON MNE --- de -2s a + 0.5s alrededor del marker.abs
epochs = mne.Epochs(raw, events, event_id=event_id,
                    tmin=-2.0, tmax=0.5,
                    baseline=None,        # normalización a mano en bloque 4
                    preload=True, verbose=False)

print(f"[epoch] {len(epochs)} épocas | shape {epochs.get_data().shape}")
# nueva shape, significa: (epochas, canales, muestras_por_época) 

# BLOQUE 4: ANÁLISIS DE RPs Y BETA-ERD

# Épocas del Bloque 3 (ya en memoria como `epochs`)
data  = epochs.get_data()    # (8, 2, 641) -> épocas, canales, muestras
times = epochs.times         # eje temporal: -2.0 a +0.5 s
sfreq = epochs.info['sfreq']
ch_names = epochs.ch_names

# PASO 1 — Potencia instantánea PARA poder ver una curva y no sólo un escalar.
# ============================================================
power = data ** 2            # cuadrar cada muestra a V², todo positivo

# Suavizar con ventana móvil de 200 ms (la potencia cruda es muy ruidosa)
win = int(0.2 * sfreq)       # 200 ms -> no. de muestras
kernel = np.ones(win) / win  # ventana de promedio simple
power_smooth = np.empty_like(power)
for ep in range(power.shape[0]):
    for ch in range(power.shape[1]):
        power_smooth[ep, ch] = np.convolve(power[ep, ch], kernel, mode='same')

# PASO 2 — Normalización ERD/ERS vs baseline temprano (-2.0 a -1.5 s)
# ============================================================
bl_mask = (times >= -2.0) & (times <= -1.5)
baseline_power = power_smooth[:, :, bl_mask].mean(axis=2, keepdims=True)

# % de cambio: (P - P_baseline) / P_baseline * 100
erd = (power_smooth - baseline_power) / baseline_power * 100 # sin unidades, es un % de cambio respecto al baseline


# PASO 3 — Promediar las 8 épocas y graficar
# ============================================================
erd_mean = erd.mean(axis=0)  # promedio sobre épocas -> (2 canales, 641)

fig, ax = plt.subplots(figsize=(9, 5))
colors = {'AF7': '#7B6FC4', 'AF8': '#A89FD8'} # aesthetic
for ch_idx, ch in enumerate(ch_names):
    ax.plot(times, erd_mean[ch_idx], label=ch, color=colors.get(ch, 'gray'), lw=2)

ax.axvline(0, color='#2C2836', ls='--', lw=1, label='RESP_BUTTON (movimiento)')
ax.axhline(0, color='#9490A8', ls='-', lw=0.5)
ax.axvspan(-2.0, -1.5, color='#C9C2E8', alpha=0.3, label='baseline')
ax.set_xlabel('Tiempo relativo al movimiento (s)')
ax.set_ylabel('Cambio de potencia beta (% vs baseline)')
ax.set_title('Desincronización beta peri-movimiento — Sandhi Alpha 01')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('beta_erd.png', dpi=130)