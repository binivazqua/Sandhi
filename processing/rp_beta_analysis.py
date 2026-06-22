
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

XDF = 'sub_Debbie_run021_eeg.xdf' # replace WITH FILE PATH TO ANALYZE

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
sfreq = float(eeg['info']['nominal_srate'][0])

# FINALLY: TARGET JUST AF7 AND AF8 (PHENOMENIC CONSCIOUS INTENTION!!!) ---
keep = ['AF7', 'AF8']
idx  = [ch_names.index(name) for name in keep]   # → [1, 2]
data_keep = data[idx, :]                         # solo esos 2 canales