#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eeg_lsl_bridge.py  —  Sandhi Interface EEG/LSL integration layer
=================================================================
Link between the PsychoPy UX/UI logic and the EEG acquisition
pipeline for Sandhi Alpha 01 protocol.

Full data pipeline:
    Muse 2
      └─► BlueMuse (Windows) / muselsl (macOS/Linux) (As PI, i'll craft the macOS/Linux instructions, but the Windows workflow is handled by BlueMuse's GUI — just start the stream there before running PsychoPy.)
            └─► LSL EEG stream  ──┐
                                   ├─► LabRecorder ──► .xdf file
            LSL Marker stream  ──┘
              ▲
              │  push_sample() calls from PsychoPy routines
              │
         THIS MODULE  (SandhiMarkerOutlet + verify_eeg_stream +
                       check_signal_quality + SignalQualityMonitor)

Usage in PsychoPy script  (replace the existing "EEG_Start_Code" block):
    from eeg_lsl_bridge import verify_eeg_stream, SandhiMarkerOutlet, MARKERS
    verify_eeg_stream()                     # abort if Muse not streaming
    marker_outlet = SandhiMarkerOutlet()    # creates the LSL marker stream
    marker_outlet.push(MARKERS.BLOCK_START)

Pre-flight signal quality check (call before opening the PsychoPy window):
    from eeg_lsl_bridge import check_signal_quality
    quality = check_signal_quality()        # prints HSI readout, returns dict
    # quality = {
    #   'TP9':  {'score': 1, 'label': 'Good',   'amplitude': 45.2},
    #   'AF7':  {'score': 1, 'label': 'Good',   'amplitude': 38.7},
    #   'AF8':  {'score': 2, 'label': 'Medium',  'amplitude': 420.1},
    #   'TP10': {'score': 4, 'label': 'Bad',     'amplitude': 1.3},
    #   'forehead_contact': True
    # }
    # RAs: map score 1→green, 2→orange, 4→red in your UI widget.

On Windows:    start BlueMuse and press "Start LSL" before running PsychoPy.
On macOS/Linux: call start_muse_stream() below — it launches muselsl in a
               background thread so it does not block PsychoPy's main loop.

NOTE: LabRecorder must be open and recording BEFORE the experiment starts.
      LabRecorder listens for all active LSL streams and writes them to .xdf.
      There is no need to interact with LabRecorder from Python for basic use and practicity.
"""

import threading
import time
import sys

import numpy as np
from scipy.signal import iirnotch, butter, filtfilt
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream


# ---------------------------------------------------------------------------
# String marker codes — Sandhi Alpha 01 Protocol v1


# Why strings (not int32):
#   - Protocol v1 explicitly requires string type for legibility in analysis.
#   - MNE/pyxdf Annotations display the event name directly when loading .xdf,
#     requiring no lookup table to interpret epochs.
#   - LabRecorder records whatever type is declared ( both work, but string
#     is the agreed standard for this lab version).
# ---------------------------------------------------------------------------
class MARKERS:
    # --- Fase 01: Pilot Trial Nivel 0, just PIPELINE VALIDATION ---
    BLOCK_START      = 'BLOCK_START'     # start of recording session
    BLOCK_END        = 'BLOCK_END'       # end of recording session
    STIM_GO          = 'STIM_GO'         # Go stimulus: color matches button
    RESP_BUTTON      = 'RESP_BUTTON'     # participant pressed a button
    RESP_LEVER_L     = 'RESP_LEVER_L'    # lever movement: left shoulder rotation (17.5°)
    RESP_LEVER_R     = 'RESP_LEVER_R'    # lever movement: right shoulder rotation (17.5°)

    # --- Fase 02: MVI — Go/No-Go action selection ---
    STIM_NOGO            = 'STIM_NOGO'            # No-Go stimulus: color does NOT match button
    FTI_BALLISTIC_ERROR  = 'FTI_BALLISTIC_ERROR'  # No-Go response < 200 ms (ballistic impulse)
    CONTROLLED_RESPONSE  = 'CONTROLLED_RESPONSE'  # response > 200 ms (successful controlled action)
    CORRECT_INHIBITION   = 'CORRECT_INHIBITION'   # No-Go trial: no response at all (clean inhibition)

    # --- Fase 03: Probes + metacognition ---
    PROBE_AUDIO      = 'PROBE_AUDIO'      # brief auditory probe tone (500 Hz, 0.1 s)
    PROBE_REPORT_YES = 'PROBE_REPORT_YES' # participant reports: was preparing to move
    PROBE_REPORT_NO  = 'PROBE_REPORT_NO'  # participant reports: was NOT preparing to move

    # --- Signal quality (cross-phase) ---
    # Emitted by SignalQualityMonitor when a channel transitions from Good/Medium → Bad
    # mid-experiment. Payload format: 'SIGNAL_QUALITY_DROP:TP9' etc.
    # Annotated in .xdf so epochs near the drop can be flagged during analysis.
    SIGNAL_QUALITY_DROP = 'SIGNAL_QUALITY_DROP'


# ---------------------------------------------------------------------------
# Internal DSP helpers
# Applied before any amplitude-based quality estimate to remove 60 Hz line
# noise and high-frequency muscle artifact, matching the filtering pipeline
# validated in Sandhi Codigo_4.1 (Blinks & Jaw dataset tool).
#
# fs = 256 Hz (Muse 2 nominal rate).
# Notch Q=30 is narrow enough to avoid distorting adjacent EEG bands.
# Butterworth order 4 at 40 Hz removes EMG without touching gamma (30–40 Hz
# is still passed — quality check only needs flatline/saturation detection,
# not spectral purity).
# ---------------------------------------------------------------------------
_FS = 256.0

def _build_filters():
    b_notch, a_notch = iirnotch(60.0, 30.0, _FS)
    nyq = 0.5 * _FS
    b_lp, a_lp = butter(4, 40.0 / nyq, btype='low', analog=False)
    return (b_notch, a_notch), (b_lp, a_lp)

_NOTCH, _LP = _build_filters()


def _filter_channel(raw: list) -> np.ndarray:
    """Apply notch + lowpass to a single channel buffer. Returns ndarray."""
    data = np.array(raw, dtype=float)
    if len(data) < 50:
        return data
    try:
        data = filtfilt(*_NOTCH, data)
        data = filtfilt(*_LP, data)
    except Exception:
        pass
    return data


# ---------------------------------------------------------------------------
# Signal quality evaluation
#
# Amplitude thresholds (μV, post-filter):
#   score 1 — Good   :  10 ≤ amplitude ≤ 400   (clean EEG range for Muse 2)
#   score 2 — Medium :  400 < amplitude ≤ 1500  (elevated noise, borderline)
#   score 4 — Bad    :  amplitude < 10           (flatline / no contact)
#                       amplitude > 1500          (saturated / strong artifact)
#
# These map to the hardware Horseshoe Indicator values BlueMuse reports
# (1 = green, 2 = yellow, 4 = red), so the same color logic applies in the UI.
#
# Score 4 is deliberately non-contiguous (skips 3) to match the HSI encoding
# used by Interaxon's SDK and BlueMuse, keeping analysis code consistent if
# a hardware HSI stream is added later.
# ---------------------------------------------------------------------------
_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']

def _evaluate_channel(filtered_data: np.ndarray) -> dict:
    if len(filtered_data) < 50:
        return {'score': 4, 'label': 'Calculating', 'amplitude': 0.0}
    amplitude = float(np.max(filtered_data) - np.min(filtered_data))
    if amplitude < 10 or amplitude > 1500:
        return {'score': 4, 'label': 'Bad', 'amplitude': amplitude}
    elif amplitude > 400:
        return {'score': 2, 'label': 'Medium', 'amplitude': amplitude}
    else:
        return {'score': 1, 'label': 'Good', 'amplitude': amplitude}


# ---------------------------------------------------------------------------
# EEG stream verification
# Call this once before the PsychoPy experiment window opens.
# If the Muse is not streaming, the experiment should not start.
# ---------------------------------------------------------------------------
def verify_eeg_stream(timeout: float = 10.0) -> bool:
    """
    Resolve the Muse EEG LSL stream and confirm it is active.

    Parameters
    ----------
    timeout : float
        Seconds to wait for a stream to appear (default 10 s).

    Returns
    -------
    StreamInlet  open inlet to the verified EEG stream.

    Raises
    ------
    RuntimeError if no EEG stream appears within `timeout`.
    """
    print(f"[Sandhi] Searching for Muse EEG stream (timeout={timeout}s)...")
    # resolve_stream() returns a list of StreamInfo objects matching the query.
    # by default, Muse 2 is an EEG type stream.
    #  It would accept an OpenBCI, a g.tec, a test stream from a laptop, pretty much anything.
    #   If another EEG device happened to be on the same local network, it would pass silently...abs

    streams = resolve_stream('type', 'EEG', timeout=timeout)

    if not streams:
        raise RuntimeError(
            "\n[Sandhi] ERROR: No EEG stream found.\n"
            "  - Windows: Open BlueMuse and press 'Start LSL' for your Muse 2.\n"
            "  - macOS/Linux: Run  muselsl stream  in a terminal first,\n"
            "    or call  start_muse_stream()  before verify_eeg_stream().\n"
            "  The experiment will NOT start without a confirmed EEG stream."
        )

    # filter to Muse-specific streams if the latter becomes an issue.
    # muse_streams = [
    #     s for s in streams
    #     if s.channel_count() in (4, 5)
    #     and abs(s.nominal_srate() - 256.0) < 1.0
    #     and ('muse' in s.name().lower() or 'interaxon' in s.manufacturer().lower())
    # ]

    info = streams[0]
    print(
        f"[Sandhi] EEG stream found: '{info.name()}' "
        f"({info.channel_count()} ch @ {info.nominal_srate()} Hz)"
    )

    print("[Sandhi] NOTE (GAP-02): confirm channel order in stream matches TP9/AF7/AF8/TP10.")
    return StreamInlet(streams[0])


# ---------------------------------------------------------------------------
# Pre-flight signal quality check
#
# Collects 2 s of EEG data (~512 samples at 256 Hz), applies the DSP pipeline,
# evaluates per-channel quality, and prints a color-coded terminal readout.
# Returns a plain dict so RAs can directly drive a UI widget from the scores.
#
# UI wiring for RAs (example — adapt to your widget toolkit):
#   quality = check_signal_quality()
#   for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
#       color = {1: 'green', 2: 'orange', 4: 'red'}.get(quality[ch]['score'], 'gray')
#       # set your widget's background/text color to `color`
# ---------------------------------------------------------------------------
def check_signal_quality(inlet: StreamInlet = None,
                          n_samples: int = 512) -> dict:
    """
    Collect a short EEG buffer and estimate per-channel contact quality.

    Parameters
    ----------
    inlet : StreamInlet, optional
        An already-open EEG inlet. If None, verify_eeg_stream() is called
        internally (adds ~10 s timeout on first use).
    n_samples : int
        Number of samples to collect before evaluating (default 512 = 2 s).
        Minimum 50. More samples = more stable estimate.

    Returns
    -------
    dict with keys 'TP9', 'AF7', 'AF8', 'TP10', 'forehead_contact'.
    Each channel value: {'score': int, 'label': str, 'amplitude': float}
    score encoding: 1=Good, 2=Medium, 4=Bad  (matches BlueMuse HSI values)
    forehead_contact: True when both AF7 and AF8 score != 4.

    Notes
    -----
    Quality is estimated from filtered EEG amplitude, not from a hardware HSI
    stream. This is portable (works with both BlueMuse and muselsl) but requires
    the participant to be still during the 2 s collection window.
    """
    owned_inlet = False
    if inlet is None:
        inlet = verify_eeg_stream()
        owned_inlet = True

    buffers = {ch: [] for ch in _CHANNELS}
    collected = 0
    print(f"[Sandhi] Collecting {n_samples} samples for signal quality check "
          f"(~{n_samples/_FS:.1f}s — participant should be still) ...")

    while collected < n_samples:
        sample, _ = inlet.pull_sample(timeout=1.0)
        if sample and len(sample) >= 4:
            for i, ch in enumerate(_CHANNELS):
                buffers[ch].append(sample[i])
            collected += 1

    results = {}
    for ch in _CHANNELS:
        filtered = _filter_channel(buffers[ch])
        results[ch] = _evaluate_channel(filtered)

    results['forehead_contact'] = (
        results['AF7']['score'] != 4 and results['AF8']['score'] != 4
    )

    _print_quality_report(results)

    if owned_inlet:
        inlet.close_stream()

    return results


def _print_quality_report(results: dict):
    """Print a human-readable HSI table to the terminal."""
    _COLOR = {1: '\033[92m', 2: '\033[93m', 4: '\033[91m'}  # green / yellow / red
    _RESET = '\033[0m'
    _GRAY  = '\033[90m'

    print("\n[Sandhi] ── Signal Quality (HSI) ─────────────────────────")
    for ch in _CHANNELS:
        r = results[ch]
        c = _COLOR.get(r['score'], _GRAY)
        bar = '█' * {1: 3, 2: 2, 4: 1}.get(r['score'], 0)
        print(f"         {ch:>4}  {c}{bar:<3} {r['label']:<11}{_RESET}  "
              f"amp={r['amplitude']:6.1f} μV")
    forehead = results['forehead_contact']
    fc_color = '\033[94m' if forehead else '\033[91m'  # blue / red
    print(f"  Forehead contact: {fc_color}{'YES' if forehead else 'NO'}{_RESET}")
    print("[Sandhi] ──────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Signal quality monitor  (background thread, runs during the experiment)
#
# Watches the live EEG stream in a daemon thread. Every `poll_interval` seconds
# it re-evaluates quality. When a channel transitions from Good or Medium to Bad
# it pushes a SIGNAL_QUALITY_DROP marker onto the LSL marker stream so the drop
# is time-stamped in the .xdf file alongside the EEG and behavioral data.
#
# Payload format: 'SIGNAL_QUALITY_DROP:TP9'  (channel name appended after colon)
# This lets analysis code filter epochs by both event type and affected channel.
#
# Usage:
#   monitor = SignalQualityMonitor(marker_outlet, inlet)
#   monitor.start()
#   # ... run experiment ...
#   monitor.stop()
# ---------------------------------------------------------------------------
class SignalQualityMonitor:
    """
    Background thread that watches EEG signal quality and emits LSL markers
    when a channel drops to Bad (score=4) mid-experiment.

    Parameters
    ----------
    marker_outlet : SandhiMarkerOutlet
        The active marker outlet to push drop events onto.
    inlet : StreamInlet
        Open EEG inlet (returned by verify_eeg_stream()).
    poll_interval : float
        Seconds between quality evaluations (default 5.0).
    buffer_seconds : float
        Seconds of EEG to evaluate per poll (default 2.0).
    """

    def __init__(self, marker_outlet: 'SandhiMarkerOutlet',
                 inlet: StreamInlet,
                 poll_interval: float = 5.0,
                 buffer_seconds: float = 2.0):
        self._outlet = marker_outlet
        self._inlet = inlet
        self._poll_interval = poll_interval
        self._n_samples = int(buffer_seconds * _FS)
        self._prev_scores = {ch: 1 for ch in _CHANNELS}  # assume Good at start
        self._running = False
        self._thread = threading.Thread(
            target=self._run, daemon=True, name='SandhiQualityMonitor'
        )

    def start(self):
        self._running = True
        self._thread.start()
        print(f"[Sandhi] SignalQualityMonitor started "
              f"(poll every {self._poll_interval}s)")

    def stop(self):
        self._running = False

    def _run(self):
        while self._running:
            time.sleep(self._poll_interval)
            if not self._running:
                break
            try:
                self._poll()
            except Exception as e:
                print(f"[Sandhi] SignalQualityMonitor error: {e}")

    def _poll(self):
        buffers = {ch: [] for ch in _CHANNELS}
        collected = 0
        while collected < self._n_samples:
            sample, _ = self._inlet.pull_sample(timeout=0.0)
            if sample and len(sample) >= 4:
                for i, ch in enumerate(_CHANNELS):
                    buffers[ch].append(sample[i])
                collected += 1
            else:
                time.sleep(0.005)
            if not self._running:
                return

        for ch in _CHANNELS:
            filtered = _filter_channel(buffers[ch])
            result = _evaluate_channel(filtered)
            new_score = result['score']
            # Fire only on transition to Bad — not on every Bad poll — to avoid
            # flooding the marker stream during a sustained dropout.
            if new_score == 4 and self._prev_scores[ch] != 4:
                payload = f"{MARKERS.SIGNAL_QUALITY_DROP}:{ch}"
                self._outlet.push(payload)
                print(f"[Sandhi] ⚠ Signal quality drop: {ch} "
                      f"(amp={result['amplitude']:.1f} μV)")
            self._prev_scores[ch] = new_score


# ---------------------------------------------------------------------------
# muselsl thread launcher  (macOS / Linux only)
# On Windows, BlueMuse handles this; calling this function there will fail
# w an ImportError message.
#
# muselsl.stream() is blocking, so it MUST run in a daemon (independent) thread — otherwise
# it would freeze the PsychoPy main loop. daemon=True ensures the thread is
# killed automatically when the main process exits.
# ---------------------------------------------------------------------------
def start_muse_stream(address: str = None) -> threading.Thread:
    """
    Start a muselsl EEG stream in a background daemon thread.

    Parameters
    ----------
    address : str, optional
        Bluetooth MAC address of the Muse 2. If None, the first available
        device is used (requires Bluetooth scan, adds a 5s ish startup time).

    Returns
    -------
    threading.Thread  (already started)
    """
    try:
        from muselsl import stream, list_muses
    except ImportError:
        print(
            "[Sandhi] muselsl not installed. On Windows use BlueMuse instead.\n"
            "         pip install muselsl"
        )
        return None

    def _run():
        target = address
        if target is None:
            print("[Sandhi] Scanning for Muse devices...")
            muses = list_muses()
            if not muses:
                raise RuntimeError("[Sandhi] No Muse device found via Bluetooth.")
            target = muses[0]['address']
            print(f"[Sandhi] Connecting to Muse at {target}")
        stream(target)  # blocking — runs until thread is killed

    t = threading.Thread(target=_run, daemon=True, name="MuseLSL-stream")
    t.start()
    # Give the stream a moment to negotiate BT and register with LSL
    time.sleep(3.0)
    return t


# ---------------------------------------------------------------------------
# Marker outlet
# The side of data production. From Psychopy to LabRecorder via LSL.
# Marker Life-Cycle:
# 1. win.callOnFlip(marker_outlet.push, 'STIM_GO')    
# 2. GPU flip ocurre → PsychoPy calls push()
# 3. push_sample(['STIM_GO']) → LSL internal timestamp assigned (e.g. t=12.345s) ← same clock as EEG stream
# 4. LabRecorder has an  StreamInlet → receives marker with timestamp t=12.345s and writes to .xdf
# 5. Inside .xdf: event 'STIM_GO' in t=12.345s
#    Inside EEG:  marker in t=12.345s  ← alignment.
# ---------------------------------------------------------------------------
class SandhiMarkerOutlet:
    """
    LSL marker outlet tailored to the Sandhi Alpha 01 experiment.

    Replaces the inline StreamInfo/StreamOutlet block in EEG_Start_Code
    inside Sandi_Interface_All_Trials_lastrun.py.

    Example implementation in PsychoPy script:
    -------------------------------------------
    marker_outlet = SandhiMarkerOutlet()
    marker_outlet.push(MARKERS.BLOCK_START) # use constants.
    ...
    marker_outlet.push(MARKERS.STIM_GO)
    marker_outlet.push(MARKERS.RESP_BUTTON)
    ...
    marker_outlet.push(MARKERS.BLOCK_END)
    """

    def __init__(self):
        # Define the marker stream info and create the outlet.
        info = StreamInfo(
            name='SandhiMarkers',
            type='Markers',
            channel_count=1,
            nominal_srate=0,        # irregular rate —> event-driven
            channel_format='string',
            source_id='sandhi_psychopy_markers'
        )
        self._outlet = StreamOutlet(info)
        # Brief pause so LabRecorder detects the new stream before any
        # markers are pushed. Without this, the first marker may be missed.

        # LabRecorder receives as a StreamInlet.
        time.sleep(0.5)
        print("[Sandhi] Marker outlet created: 'SandhiMarkers' (string)")

    def push(self, marker: str, verbose: bool = True):
        """Push a single string marker onto the LSL stream."""
        self._outlet.push_sample([str(marker)])
        if verbose:
            print(f"[Sandhi] Marker sent: '{marker}'")


# ---------------------------------------------------------------------------
# Quick self-test  (run as  python eeg_lsl_bridge.py  to verify setup)
# Sends the minimal Fase 01 marker sequence: BLOCK_START → STIM_GO →
# RESP_BUTTON.
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=== Sandhi LSL Bridge self-test (Protocol v1 — Fase 01 markers) ===")

    print("\nStep 1 — verifying Muse EEG stream (start BlueMuse or muselsl first) ...")
    try:
        eeg_inlet = verify_eeg_stream(timeout=10)
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    print("\nStep 2 — signal quality check (sit still for ~2 s) ...")
    quality = check_signal_quality(inlet=eeg_inlet)
    bad_channels = [ch for ch in _CHANNELS if quality[ch]['score'] == 4]
    if bad_channels:
        print(f"[Sandhi] WARNING: poor contact on {bad_channels}. "
              "Adjust headset before starting the experiment.")
    else:
        print("[Sandhi] All channels within acceptable range.")

    print("\nStep 3 — creating marker outlet ...")
    outlet = SandhiMarkerOutlet()

    print("\nStep 4 — sending Fase 01 test sequence ...")
    for marker in [
        MARKERS.BLOCK_START,
        MARKERS.STIM_GO,
        MARKERS.RESP_BUTTON,
        MARKERS.BLOCK_END,
    ]:
        outlet.push(marker)
        time.sleep(0.5)

    print("\n[Sandhi] Self-test complete. Check LabRecorder for 4 string markers.")
    print("         Expected in .xdf: BLOCK_START, STIM_GO, RESP_BUTTON, BLOCK_END")

    eeg_inlet.close_stream()
