#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eeg_lsl_bridge.py  —  Sandhi Interface EEG/LSL integration layer
=================================================================
This module is the missing link between the PsychoPy UX/UI logic
(Sandi_Interface_All_Trials_lastrun.py) and the EEG acquisition
pipeline described in the Sandhi Alpha 01 architecture document.

Full data pipeline:
    Muse 2
      └─► BlueMuse (Windows) / muselsl (macOS/Linux)
            └─► LSL EEG stream  ──┐
                                   ├─► LabRecorder ──► .xdf file
            LSL Marker stream  ──┘
              ▲
              │  push_sample() calls from PsychoPy routines
              │
         THIS MODULE  (SandhiMarkerOutlet + verify_eeg_stream)

Usage in PsychoPy script  (replace the existing "EEG_Start_Code" block):
    from eeg_lsl_bridge import verify_eeg_stream, SandhiMarkerOutlet, MARKERS
    verify_eeg_stream()                     # abort if Muse not streaming
    marker_outlet = SandhiMarkerOutlet()    # creates the LSL marker stream
    marker_outlet.push(MARKERS.EXPERIMENT_START)

On Windows:    start BlueMuse and press "Start LSL" before running PsychoPy.
On macOS/Linux: call start_muse_stream() below — it launches muselsl in a
               background thread so it does not block PsychoPy's main loop.

NOTE: LabRecorder must be open and recording BEFORE the experiment starts.
      LabRecorder listens for all active LSL streams and writes them to .xdf.
      There is no need to interact with LabRecorder from Python for basic use.
"""

import threading
import time
import sys

from pylsl import StreamInfo, StreamOutlet, resolve_stream


# ---------------------------------------------------------------------------
# Integer marker codes
# Using int32 codes (not strings) is best practice:
#   - smaller payload, lower LSL latency
#   - directly usable as event codes in EEGLab / MNE without string parsing
#   - unambiguous across analysis scripts
# ---------------------------------------------------------------------------
class MARKERS:
    # Experiment-level
    EXPERIMENT_START = 10
    EXPERIMENT_END   = 99

    # Emotions block  (Task 1 — slider, 3 s exposure)
    EMOCIONES_BLOCK_START = 20
    EMOCIONES_TRIAL_START = 21
    EMOCIONES_TRIAL_END   = 22
    EMOCIONES_BLOCK_END   = 23

    # Buttons block   (Task 2 — ROJO/AMARILLO/AZUL/VERDE)
    BOTONES_BLOCK_START   = 30
    BOTONES_TRIAL_START   = 31
    BOTONES_CORRECT       = 32   # correct response registered
    BOTONES_INCORRECT     = 33   # incorrect or no response
    BOTONES_BLOCK_END     = 34

    # Arrows/joystick block  (Task 3 — ARRIBA/ABAJO/IZQ/DER)
    PALANCAS_BLOCK_START  = 40
    PALANCAS_TRIAL_START  = 41
    PALANCAS_CORRECT      = 42
    PALANCAS_INCORRECT    = 43
    PALANCAS_BLOCK_END    = 44


# ---------------------------------------------------------------------------
# EEG stream verification
# Call this once before the PsychoPy experiment window opens.
# If the Muse is not streaming, the experiment should not start — there is
# no point collecting behavioral data without the EEG it is meant to annotate.
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
    bool  True if stream found.

    Raises
    ------
    RuntimeError if no EEG stream appears within `timeout`.
    """
    print(f"[Sandhi] Searching for Muse EEG stream (timeout={timeout}s)...")
    streams = resolve_stream('type', 'EEG', timeout=timeout)
    if not streams:
        raise RuntimeError(
            "\n[Sandhi] ERROR: No EEG stream found.\n"
            "  - Windows: Open BlueMuse and press 'Start LSL' for your Muse 2.\n"
            "  - macOS/Linux: Run  muselsl stream  in a terminal first,\n"
            "    or call  start_muse_stream()  before verify_eeg_stream().\n"
            "  The experiment will NOT start without a confirmed EEG stream."
        )
    info = streams[0]
    print(
        f"[Sandhi] EEG stream found: '{info.name()}' "
        f"({info.channel_count()} ch @ {info.nominal_srate()} Hz)"
    )
    return True


# ---------------------------------------------------------------------------
# muselsl thread launcher  (macOS / Linux only)
# On Windows, BlueMuse handles this; calling this function there will fail
# gracefully with an ImportError message.
#
# muselsl.stream() is blocking, so it MUST run in a daemon thread — otherwise
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
        device is used (requires Bluetooth scan, adds ~5 s startup time).

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
# Wraps StreamOutlet with integer codes (int32) so LabRecorder saves them
# as numeric event markers in the XDF file — directly readable by EEGLab
# and MNE without any string-to-int conversion step.
#
# The existing PsychoPy code uses channel_format='string'. That works, but
# switching to int32 is strongly preferred for downstream analysis toolchains.
# ---------------------------------------------------------------------------
class SandhiMarkerOutlet:
    """
    LSL marker outlet tailored to the Sandhi experiment.

    Replaces the inline StreamInfo/StreamOutlet block in EEG_Start_Code
    inside Sandi_Interface_All_Trials_lastrun.py.

    Example
    -------
    marker_outlet = SandhiMarkerOutlet()
    marker_outlet.push(MARKERS.EXPERIMENT_START)
    ...
    marker_outlet.push(MARKERS.EMOCIONES_TRIAL_START)
    """

    def __init__(self):
        info = StreamInfo(
            name='SandhiMarkers',
            type='Markers',
            channel_count=1,
            nominal_srate=0,          # irregular rate — event-driven
            channel_format='int32',   # int32 preferred over 'string'
            source_id='sandhi_psychopy_markers'
        )
        self._outlet = StreamOutlet(info)
        # Brief pause so LabRecorder detects the new stream before any
        # markers are pushed. Without this, the first marker may be missed.
        time.sleep(0.5)
        print("[Sandhi] Marker outlet created: 'SandhiMarkers' (int32)")

    def push(self, code: int, verbose: bool = True):
        """Push a single integer marker code onto the LSL stream."""
        self._outlet.push_sample([int(code)])
        if verbose:
            label = _code_to_label(code)
            print(f"[Sandhi] Marker sent: {code}  ({label})")


def _code_to_label(code: int) -> str:
    """Reverse-lookup a MARKERS constant name from its integer value."""
    for name, val in vars(MARKERS).items():
        if not name.startswith('_') and val == code:
            return name
    return 'UNKNOWN'


# ---------------------------------------------------------------------------
# Quick self-test  (run as  python eeg_lsl_bridge.py  to verify setup)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=== Sandhi LSL Bridge self-test ===")
    print("Step 1 — verifying EEG stream (start BlueMuse/muselsl first) ...")
    try:
        verify_eeg_stream(timeout=10)
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    print("\nStep 2 — creating marker outlet ...")
    outlet = SandhiMarkerOutlet()

    print("\nStep 3 — sending test markers ...")
    for code in [MARKERS.EXPERIMENT_START,
                 MARKERS.EMOCIONES_TRIAL_START,
                 MARKERS.EMOCIONES_TRIAL_END,
                 MARKERS.EXPERIMENT_END]:
        outlet.push(code)
        time.sleep(0.5)

    print("\n[Sandhi] Self-test complete. Check LabRecorder for the 4 markers.")
