#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rp_beta_analysis.py — Readiness Potential / Beta-ERD analysis pipeline
=======================================================================
Hardware:  Muse 2 (AF7, AF8, TP9, TP10 @ 256 Hz)
           Unicorn Hybrid Black — channel-agnostic, detected at runtime
Markers:   STIM_GO, STIM_NOGO, FTI_BALLISTIC_ERROR,
           CONTROLLED_RESPONSE, CORRECT_INHIBITION, BLOCK_START
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

# ---------------------------------------------------------------------------
# Global console — all terminal output routes through here, no bare print()
# ---------------------------------------------------------------------------
console = Console()

# ---------------------------------------------------------------------------
# Marker vocabulary — add a new marker here and the rest of the pipeline
# picks it up automatically (1-line change per new condition).
# ---------------------------------------------------------------------------
MARKER_IDS = {
    "FTI_BALLISTIC_ERROR":  1,
    "CONTROLLED_RESPONSE":  2,
    "CORRECT_INHIBITION":   3,
    "STIM_GO":              4,
    "STIM_NOGO":            5,
    "BLOCK_START":          6,
}

# Minimum trial count below which a condition is skipped rather than
# producing unreliable statistics.
MIN_TRIALS_FOR_ANALYSIS = 5


# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------

def load_xdf(filepath: str) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    """Load EEG and marker streams from an .xdf file recorded via LabRecorder.

    Searches all streams in the file for one typed 'EEG' and one typed
    'Markers'.  Channel names and sampling rate are read from the stream
    header so the function is hardware-agnostic: Muse 2 and Unicorn both
    work without code changes.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the .xdf recording.

    Returns
    -------
    eeg_data : np.ndarray, shape (n_channels, n_samples)
        Raw voltage samples in microvolts.
    timestamps : np.ndarray, shape (n_samples,)
        LSL timestamps (seconds, monotonic clock) for each EEG sample.
    marker_labels : list[str]
        Marker string values in arrival order.
    marker_times : np.ndarray
        LSL timestamps corresponding to each marker.
    sfreq : float
        Nominal sampling rate reported by the EEG stream header.
    ch_names : list[str]
        Channel labels as declared in the stream header (e.g. ['AF7','AF8',...]).
    """
    # TODO: call pyxdf to parse the file; returns list of stream dicts + file header
    streams, _ = pyxdf.load_xdf(filepath)

    eeg_stream = None
    marker_stream = None

    for stream in streams:
        # TODO: inspect 'type' key in stream['info'] to identify stream role
        stream_type = stream["info"]["type"][0].lower()
        if stream_type == "eeg":
            eeg_stream = stream
        elif stream_type == "markers":
            marker_stream = stream

    if eeg_stream is None:
        raise RuntimeError("No EEG stream found in .xdf file.")
    if marker_stream is None:
        raise RuntimeError("No Marker stream found in .xdf file.")

    # TODO: extract sample matrix; pyxdf stores it under key 'time_series'
    #       shape is (n_samples, n_channels) — transpose to (n_channels, n_samples)
    eeg_data = eeg_stream["time_series"].T

    # TODO: read per-sample timestamps from 'time_stamps' key
    timestamps = eeg_stream["time_stamps"]

    # TODO: read nominal_srate from stream header (stored as a string, cast to float)
    sfreq = float(eeg_stream["info"]["nominal_srate"][0])

    # TODO: walk the nested XML-like header to collect channel labels
    ch_names = [
        ch["label"][0]
        for ch in eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
    ]

    # TODO: marker time series is shape (n_markers, 1); flatten to 1-D list of strings
    marker_labels = [m[0] for m in marker_stream["time_series"]]
    marker_times = marker_stream["time_stamps"]

    console.log(
        f"[green]Loaded[/green] {eeg_data.shape[1]} samples × "
        f"{eeg_data.shape[0]} channels @ {sfreq} Hz | "
        f"{len(marker_labels)} markers"
    )
    return eeg_data, timestamps, marker_labels, marker_times, sfreq, ch_names


# ---------------------------------------------------------------------------
# 2. MNE RAW OBJECT
# ---------------------------------------------------------------------------

def build_raw(
    eeg_data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
) -> mne.io.RawArray:
    """Construct an MNE RawArray from raw voltage data.

    Sets channel types to 'eeg' and attempts to apply a standard 10-20
    montage when all channel names are recognised (e.g. AF7, AF8 are valid
    10-20 locations).  Falls back silently when non-standard labels are
    present (e.g. Unicorn custom labels).

    Parameters
    ----------
    eeg_data : np.ndarray, shape (n_channels, n_samples)
        Voltage in microvolts.
    sfreq : float
        Sampling frequency in Hz.
    ch_names : list[str]
        Labels matching the columns of eeg_data.

    Returns
    -------
    raw : mne.io.RawArray
        Unfilitered, unannotated MNE Raw object.
    """
    # TODO: create an MNE Info object — the metadata container required by all MNE objects
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types="eeg",  # TODO: all channels typed as EEG; extend here for EOG/EMG channels
    )

    # TODO: MNE expects volts internally; Muse 2 / Unicorn deliver microvolts — scale down
    eeg_volts = eeg_data * 1e-6

    # TODO: wrap numpy array in RawArray; verbose=False suppresses MNE's default log spam
    raw = mne.io.RawArray(eeg_volts, info, verbose=False)

    # TODO: attempt standard 10-20 montage assignment for source localisation compatibility
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        # set_montage raises ValueError when any ch_name is not in the montage
        raw.set_montage(montage, on_missing="raise", verbose=False)
        console.log("[green]Montage[/green] standard_1020 applied.")
    except ValueError:
        # TODO: non-10-20 labels (custom Unicorn config) — montage skipped, analysis continues
        console.log("[yellow]Montage[/yellow] not applied — channel names not in standard_1020.")

    return raw


# ---------------------------------------------------------------------------
# 3. PREPROCESSING
# ---------------------------------------------------------------------------

def preprocess(raw: mne.io.RawArray) -> mne.io.RawArray:
    """Apply bandpass and notch filters to the continuous Raw signal.

    Filter order:
        1. Bandpass 1–30 Hz  (IIR Butterworth, order 4, zero-phase via sosfiltfilt)
        2. Notch   50 Hz     (IIR notch, Q=30, zero-phase via filtfilt)

    A 1 Hz high-pass removes slow DC drift without distorting the RP's
    slow negative buildup.  30 Hz low-pass removes high-frequency EMG.
    50 Hz notch removes European mains interference.

    Parameters
    ----------
    raw : mne.io.RawArray
        Unfiltered signal, modified in-place.

    Returns
    -------
    raw : mne.io.RawArray
        Same object, now filtered.
    """
    sfreq = raw.info["sfreq"]

    # --- Bandpass 1–30 Hz (Butterworth IIR, order 4) ---
    # butter() args:
    #   N=4         : filter order — higher = steeper roll-off, more phase delay
    #   Wn=[1, 30]  : cutoff frequencies in Hz (normalised internally by sfreq/2)
    #   btype='band': bandpass (passes frequencies between Wn[0] and Wn[1])
    #   fs=sfreq    : sampling rate used to normalise Wn
    #   output='sos': second-order sections — numerically stable vs. 'ba' for order > 2
    # TODO: increase order to 6 if 1 Hz rolloff bleeds into DC; check with plot_psd()
    sos_bp = butter(N=4, Wn=[1, 30], btype="band", fs=sfreq, output="sos")

    # TODO: get raw data as (n_channels, n_samples) numpy array; picks='eeg' excludes non-EEG
    data, times = raw.get_data(picks="eeg", return_times=True)

    # TODO: sosfiltfilt applies the filter twice (forward + backward) for zero phase distortion
    data_bp = sosfiltfilt(sos_bp, data, axis=-1)

    # --- Notch 50 Hz ---
    # iirnotch() args:
    #   w0=50      : centre frequency to reject (Hz)
    #   Q=30       : quality factor — higher Q = narrower notch (30 ≈ ±1.7 Hz bandwidth)
    #   fs=sfreq   : sampling rate
    # TODO: change w0 to 60 for North American recordings (60 Hz mains)
    b_notch, a_notch = iirnotch(w0=50, Q=30, fs=sfreq)

    # TODO: filtfilt for zero-phase; axis=-1 filters along time axis
    data_notch = filtfilt(b_notch, a_notch, data_bp, axis=-1)

    # TODO: write filtered data back into the Raw object in-place
    raw._data[mne.pick_types(raw.info, eeg=True), :] = data_notch

    console.log("[green]Preprocess[/green] bandpass 1–30 Hz + notch 50 Hz applied.")
    return raw


# ---------------------------------------------------------------------------
# 4. MARKER INJECTION
# ---------------------------------------------------------------------------

def inject_markers(
    raw: mne.io.RawArray,
    marker_labels: list[str],
    marker_times: np.ndarray,
) -> mne.io.RawArray:
    """Convert LSL marker timestamps into MNE Annotations and attach to Raw.

    LSL timestamps are on an absolute monotonic clock; MNE Annotations use
    time relative to raw.first_time (the timestamp of sample 0).  The offset
    is computed by subtracting the first EEG timestamp from each marker time.

    Parameters
    ----------
    raw : mne.io.RawArray
        Filtered Raw object without annotations.
    marker_labels : list[str]
        String marker values in arrival order.
    marker_times : np.ndarray
        Absolute LSL timestamps for each marker.

    Returns
    -------
    raw : mne.io.RawArray
        Same object with Annotations attached.
    """
    # TODO: raw.annotations.orig_time or raw.first_time gives the t=0 reference
    #       for MNE; here we use the timestamp of the first EEG sample as origin
    t0 = raw.times[0]  # relative offset — RawArray always starts at 0.0

    # TODO: for RawArray built from pyxdf data we need the absolute LSL t0
    #       store it on raw.info as a custom key so inject_markers can retrieve it
    lsl_t0 = raw.info.get("lsl_t0", 0.0)

    # TODO: convert absolute LSL marker timestamps to seconds-since-raw-start
    onset_seconds = marker_times - lsl_t0

    # TODO: filter out markers that land before sample 0 or after last sample
    valid_mask = (onset_seconds >= 0) & (onset_seconds <= raw.times[-1])
    onset_seconds = onset_seconds[valid_mask]
    labels_valid = [marker_labels[i] for i, v in enumerate(valid_mask) if v]

    # TODO: MNE Annotations need onset (s), duration (s), description (str)
    #       duration=0.0 for instantaneous event markers
    annotations = mne.Annotations(
        onset=onset_seconds,
        duration=np.zeros(len(onset_seconds)),   # TODO: set non-zero for block/epoch boundaries
        description=labels_valid,
    )

    # TODO: set_annotations replaces any existing annotations on the Raw object
    raw.set_annotations(annotations)

    console.log(f"[green]Markers[/green] {len(labels_valid)} annotations injected.")
    return raw


# ---------------------------------------------------------------------------
# 5. EPOCHING
# ---------------------------------------------------------------------------

def epoch(
    raw: mne.io.RawArray,
    event_ids: dict,
    tmin: float = -1.5,
    tmax: float = 0.5,
) -> mne.Epochs:
    """Extract stimulus-locked epochs from annotated continuous data.

    Baseline correction uses the −1500 to −1000 ms window to avoid
    contamination by the RP buildup (which begins ~−1000 ms pre-movement).

    Epochs exceeding 150 µV peak-to-peak on any channel are dropped as
    artefacts (blinks, jaw clench, cable movement).

    Parameters
    ----------
    raw : mne.io.RawArray
        Annotated, filtered Raw object.
    event_ids : dict
        Mapping of condition name → integer event ID (from MARKER_IDS).
    tmin : float
        Epoch start relative to marker (seconds).  −1.5 captures full RP.
    tmax : float
        Epoch end relative to marker (seconds).

    Returns
    -------
    epochs : mne.Epochs
        Cleaned, baseline-corrected epoch object.
    """
    # TODO: mne.events_from_annotations converts string Annotations to integer events array
    #       returns (events array [n_events × 3], mapping dict {description: int_id})
    events, _ = mne.events_from_annotations(raw, event_id=event_ids, verbose=False)

    if len(events) == 0:
        raise RuntimeError("No events found — check marker labels match MARKER_IDS keys.")

    # TODO: mne.Epochs args:
    #   raw        : source signal
    #   events     : (n_events, 3) array [sample_idx, 0, event_id]
    #   event_id   : dict mapping condition name to integer
    #   tmin/tmax  : epoch window relative to event onset
    #   baseline   : interval used for mean subtraction; tuple in seconds
    #   reject     : peak-to-peak threshold dict; 150e-6 V = 150 µV
    #   preload    : load all epochs into RAM now (required for reject)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_ids,
        tmin=tmin,
        tmax=tmax,
        baseline=(-1.5, -1.0),    # TODO: adjust if RP onset differs across participants
        reject={"eeg": 150e-6},   # TODO: consider auto-reject (autoreject library) for data-driven thresholds
        preload=True,
        verbose=False,
    )

    n_dropped = len(events) - len(epochs)
    console.log(
        f"[green]Epochs[/green] {len(epochs)} kept / {n_dropped} dropped "
        f"(reject threshold 150 µV p-p)"
    )
    return epochs


# ---------------------------------------------------------------------------
# 6. BETA-ERD
# ---------------------------------------------------------------------------

def compute_beta_erd(
    epochs: mne.Epochs,
    fmin: float = 13,
    fmax: float = 20,
) -> dict:
    """Compute Welch power spectral density in the low-beta band per condition.

    Event-Related Desynchronisation (ERD) is quantified as relative power
    decrease during movement preparation vs. a pre-movement baseline.
    This function returns raw PSD arrays; the ERD ratio is computed in main().

    Parameters
    ----------
    epochs : mne.Epochs
        Cleaned epoch object containing at least one condition.
    fmin : float
        Lower bound of beta band (Hz).  Default 13 Hz.
    fmax : float
        Upper bound of beta band (Hz).  Default 20 Hz.

    Returns
    -------
    results : dict
        Nested dict: {condition_name: {channel_name: mean_beta_power (µV²/Hz)}}
    """
    results = {}

    for condition in epochs.event_id:
        # TODO: select epochs for this condition only
        cond_epochs = epochs[condition]

        # TODO: guard — skip condition if trial count is below minimum
        if len(cond_epochs) < MIN_TRIALS_FOR_ANALYSIS:
            console.log(
                f"[yellow]WARNING[/yellow] '{condition}' has only "
                f"{len(cond_epochs)} trials (< {MIN_TRIALS_FOR_ANALYSIS}). "
                "Skipping beta ERD for this condition."
            )
            continue

        # TODO: compute_psd() returns a Spectrum object; method='welch' uses
        #       Welch's overlapping segments (reduces variance vs. single FFT)
        #       fmin/fmax restricts output to the beta band of interest
        spectrum = cond_epochs.compute_psd(
            method="welch",
            fmin=fmin,
            fmax=fmax,
            picks="eeg",
            verbose=False,
        )

        # TODO: .get_data() returns (n_epochs, n_channels, n_freqs); average over
        #       epochs (axis=0) then over frequencies (axis=-1) for a scalar per channel
        psd_data = spectrum.get_data()          # shape: (n_epochs, n_channels, n_freqs)
        mean_psd = psd_data.mean(axis=0)        # shape: (n_channels, n_freqs)
        mean_beta = mean_psd.mean(axis=-1)      # shape: (n_channels,)

        # TODO: convert from V²/Hz (MNE internal) to µV²/Hz for human-readable values
        mean_beta_uv2 = mean_beta * 1e12

        # TODO: zip channel names with their corresponding mean beta power values
        results[condition] = {
            ch: float(pwr)
            for ch, pwr in zip(cond_epochs.ch_names, mean_beta_uv2)
        }

    return results


# ---------------------------------------------------------------------------
# 7. ERP PLOT
# ---------------------------------------------------------------------------

def plot_erp(epochs: mne.Epochs) -> None:
    """Plot grand average ERP waveform per condition.

    Highlights the −1500 to 0 ms Readiness Potential window with a shaded
    region.  Each condition is overlaid on the same axes for direct
    comparison of FTI_BALLISTIC_ERROR vs CONTROLLED_RESPONSE morphology.

    Parameters
    ----------
    epochs : mne.Epochs
        Cleaned epoch object.  All EEG channels are averaged.
    """
    import matplotlib.pyplot as plt  # TODO: move to top-level import if plots are used frequently

    fig, ax = plt.subplots(figsize=(10, 4))

    for condition in epochs.event_id:
        # TODO: .average() computes the mean across all epochs for one condition
        evoked = epochs[condition].average()

        # TODO: .get_data() shape (n_channels, n_times); mean over channels for a global field
        gfp = evoked.get_data().mean(axis=0) * 1e6  # convert V → µV

        # TODO: epochs.times gives the time axis in seconds; convert to ms for readability
        times_ms = evoked.times * 1000

        ax.plot(times_ms, gfp, label=condition)

    # TODO: shade the RP window (−1500 to 0 ms) to guide the reader's eye
    ax.axvspan(-1500, 0, alpha=0.08, color="blue", label="RP window")

    # TODO: vertical line at t=0 marks the movement/response onset
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title("Grand Average ERP — Readiness Potential")
    ax.legend()
    plt.tight_layout()

    # TODO: save to outdir rather than plt.show() for headless / server environments
    plt.show()


# ---------------------------------------------------------------------------
# 8. CSV EXPORT
# ---------------------------------------------------------------------------

def export_csv(epochs: mne.Epochs, outpath: str) -> None:
    """Export epoch data to a tidy CSV for downstream R analysis with lme4.

    Output columns:
        epoch_idx, condition, channel, time_ms, amplitude_uv

    This long format is directly ingestible by tidyr/lme4 in R without
    further reshaping.

    Parameters
    ----------
    epochs : mne.Epochs
        Cleaned epoch object.
    outpath : str
        Full path to the output .csv file.
    """
    rows = []

    for condition in epochs.event_id:
        # TODO: select condition subset; .get_data() shape (n_epochs, n_ch, n_times)
        cond_data = epochs[condition].get_data()    # V
        times_ms = epochs.times * 1000              # seconds → milliseconds

        for epoch_idx in range(cond_data.shape[0]):
            for ch_idx, ch_name in enumerate(epochs.ch_names):
                for t_idx, t_ms in enumerate(times_ms):
                    rows.append({
                        "epoch_idx":  epoch_idx,
                        "condition":  condition,
                        "channel":    ch_name,
                        "time_ms":    round(t_ms, 2),
                        # TODO: convert V → µV for interpretable scale in R models
                        "amplitude_uv": cond_data[epoch_idx, ch_idx, t_idx] * 1e6,
                    })

    # TODO: pd.DataFrame from list of dicts; .to_csv() with index=False omits row numbers
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_csv(outpath, index=False)
    console.log(f"[green]Export[/green] {len(df)} rows → {outpath}")


# ---------------------------------------------------------------------------
# 9. MAIN / CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entrypoint for the full analysis pipeline.

    Flags
    -----
    --file    : (required) path to the .xdf recording
    --outdir  : (optional, default ./output) directory for CSV and plots

    Pipeline order:
        load_xdf → build_raw → preprocess → inject_markers →
        epoch → compute_beta_erd → plot_erp → export_csv

    Prints a rich summary table on completion.
    """
    # TODO: argparse wires CLI flags to Python variables without manual sys.argv parsing
    parser = argparse.ArgumentParser(
        description="Readiness Potential / Beta-ERD analysis — Sandhi project"
    )
    # TODO: --file is required; no default forces the user to specify a recording
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the .xdf recording file",
    )
    # TODO: --outdir has a sensible default so the user can omit it in quick runs
    parser.add_argument(
        "--outdir",
        default="./output",
        help="Directory for CSV output and plots (default: ./output)",
    )
    args = parser.parse_args()

    console.rule("[bold cyan]Sandhi EEG Analysis Pipeline[/bold cyan]")

    # Step 1 — load
    eeg_data, timestamps, marker_labels, marker_times, sfreq, ch_names = load_xdf(args.file)

    # Step 2 — build Raw
    raw = build_raw(eeg_data, sfreq, ch_names)

    # TODO: store the LSL t0 in raw.info so inject_markers can compute relative onsets
    raw.info["lsl_t0"] = timestamps[0]

    # Step 3 — filter
    raw = preprocess(raw)

    # Step 4 — markers
    raw = inject_markers(raw, marker_labels, marker_times)

    # Step 5 — epoch; use only the two primary contrast conditions
    # TODO: add STIM_GO / STIM_NOGO here to expand the contrast set
    event_ids = {
        k: v for k, v in MARKER_IDS.items()
        if k in {"FTI_BALLISTIC_ERROR", "CONTROLLED_RESPONSE"}
    }
    epochs = epoch(raw, event_ids)

    # Step 6 — beta ERD
    beta_results = compute_beta_erd(epochs)

    # Step 7 — plot
    plot_erp(epochs)

    # Step 8 — export
    csv_path = os.path.join(args.outdir, "epochs.csv")
    export_csv(epochs, csv_path)

    # --- Summary table ---
    table = Table(title="Pipeline Summary", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    for condition in epochs.event_id:
        n = len(epochs[condition])
        table.add_row(f"n_epochs  [{condition}]", str(n))

    # TODO: count dropped epochs by comparing events found vs epochs kept
    n_total_events = sum(
        len(mne.events_from_annotations(raw, event_id={k: v}, verbose=False)[0])
        for k, v in event_ids.items()
    )
    table.add_row("n_rejected", str(n_total_events - len(epochs)))

    for condition, ch_powers in beta_results.items():
        for ch, pwr in ch_powers.items():
            table.add_row(f"mean_beta_power  [{condition}] [{ch}]", f"{pwr:.3f} µV²/Hz")

    console.print(table)
    console.rule("[bold green]Done[/bold green]")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
