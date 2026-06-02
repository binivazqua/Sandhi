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
         THIS MODULE  (SandhiMarkerOutlet + verify_eeg_stream)

Usage in PsychoPy script  (replace the existing "EEG_Start_Code" block):
    from eeg_lsl_bridge import verify_eeg_stream, SandhiMarkerOutlet, MARKERS
    verify_eeg_stream()                     # abort if Muse not streaming
    marker_outlet = SandhiMarkerOutlet()    # creates the LSL marker stream
    marker_outlet.push(MARKERS.BLOCK_START)

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

from pylsl import StreamInfo, StreamOutlet, resolve_stream


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
    bool  True if stream found.

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
    return True


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
    print("Verifying EEG stream (start BlueMuse or muselsl first) ...")
    try:
        verify_eeg_stream(timeout=10)
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    print("\nCreating marker outlet ...")
    outlet = SandhiMarkerOutlet()

    print("\nSending Fase 01 test sequence ...")
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
