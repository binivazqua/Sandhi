#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_experiment.py  —  Sandhi experiment launcher
=================================================
Run THIS script instead of Sandi_Interface_All_Trials_lastrun.py directly.

It performs all pre-flight checks for the Muse 2 / LSL / LabRecorder pipeline
before handing control to PsychoPy.

Pre-flight order:
    1. (macOS/Linux only) optionally start muselsl in a background thread
    2. Verify the Muse EEG stream is visible on LSL
    3. Remind operator to start LabRecorder and begin recording
    4. Wait for operator confirmation (Enter key) then launch PsychoPy

Usage:
    python run_experiment.py                    # interactive
    python run_experiment.py --skip-eeg-check   # skip EEG check (dry-run / testing)
    python run_experiment.py --start-muselsl    # auto-start muselsl thread (macOS/Linux)
"""

import sys
import os
import argparse
import subprocess

# Ensure imports resolve whether run from repo root or from Sandhi_Demo/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOFTWARE_DIR = os.path.dirname(_THIS_DIR)
if _SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, _SOFTWARE_DIR)

from eeg_lsl_bridge import verify_eeg_stream, start_muse_stream, SandhiMarkerOutlet, MARKERS


PSYCHOPY_SCRIPT = os.path.join(_THIS_DIR, 'Sandi_Interface_All_Trials_lastrun.py')


def parse_args():
    p = argparse.ArgumentParser(description='Sandhi experiment launcher')
    p.add_argument('--skip-eeg-check', action='store_true',
                   help='Skip Muse EEG stream verification (for dry-run testing)')
    p.add_argument('--start-muselsl', action='store_true',
                   help='Auto-start muselsl in background thread (macOS/Linux)')
    p.add_argument('--muse-address', default=None,
                   help='Bluetooth MAC address of Muse 2 (optional, auto-scans if omitted)')
    return p.parse_args()


def preflight(args):
    print("=" * 60)
    print("  Sandhi Interface — Pre-flight checklist")
    print("=" * 60)

    # Step 1 — optionally launch muselsl
    if args.start_muselsl:
        print("\n[1/3] Starting muselsl stream thread...")
        start_muse_stream(address=args.muse_address)
        print("      muselsl thread started. Waiting 3 s for stream to stabilise...")
    else:
        print(
            "\n[1/3] EEG stream — manual setup required:\n"
            "      Windows : Open BlueMuse, pair Muse 2, press 'Start LSL'.\n"
            "      macOS/Linux: Run  muselsl stream  in a separate terminal,\n"
            "                   or re-run with  --start-muselsl  flag."
        )

    # Step 2 — verify EEG stream
    if not args.skip_eeg_check:
        print("\n[2/3] Verifying Muse EEG stream on LSL...")
        try:
            verify_eeg_stream(timeout=15)
        except RuntimeError as e:
            print(e)
            sys.exit(1)
    else:
        print("\n[2/3] EEG stream check SKIPPED (--skip-eeg-check flag set).")

    # Step 3 — LabRecorder reminder
    # LabRecorder does not have a stable Python remote-control API in all versions,
    # so we prompt the operator manually. In future this could use the
    # LabRecorder remote-control protocol (TCP port 22345) via socket.
    print(
        "\n[3/3] LabRecorder setup:\n"
        "      - Open LabRecorder.\n"
        "      - You should see 'type=EEG' AND 'type=Markers' in the stream list.\n"
        "        (Markers stream appears after PsychoPy starts, that is fine.)\n"
        "      - Set the output file path and filename.\n"
        "      - Press 'Start' in LabRecorder to begin recording.\n"
    )
    input("      Press ENTER when LabRecorder is recording and Muse is on participant...")

    print("\n[Sandhi] Pre-flight complete. Launching PsychoPy experiment...\n")


def main():
    args = parse_args()
    preflight(args)

    # Hand off to PsychoPy — subprocess keeps the launcher process alive
    # so LabRecorder continues to see the marker stream throughout the session.
    result = subprocess.run([sys.executable, PSYCHOPY_SCRIPT], cwd=_THIS_DIR)
    if result.returncode != 0:
        print(f"\n[Sandhi] PsychoPy exited with code {result.returncode}.")
    else:
        print("\n[Sandhi] Experiment finished. Stop LabRecorder and save the .xdf file.")


if __name__ == '__main__':
    main()
