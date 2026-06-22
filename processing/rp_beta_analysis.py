
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
      2. Pasa-banda 13-20 Hz --> low-beta (Gavenas et al. 2025)
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