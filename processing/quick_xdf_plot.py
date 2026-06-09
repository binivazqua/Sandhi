"""
quick_xdf_plot.py (mainly) for poster-quality EEG + marker visualiser for .xdf files after trials!!
==========================================================================
Loads any .xdf recorded with LabRecorder (As of Sandhi Beta 01, Muse 2),
stacks EEG channels vertically, and draws colour-coded marker lines.

HOW TO USE!?!?
-----
    python quick_xdf_plot.py --file recording.xdf
    python quick_xdf_plot.py --file recording.xdf --seconds 30 --title "Trial 1"
    python quick_xdf_plot.py --file recording.xdf --outdir ./poster_figs --no-show

Outputs
-------
    <outdir>/<stem>_eeg_markers.png   — high-DPI figure ready for a poster
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless by default; switched to TkAgg/Qt if --show is used
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pyxdf
from rich.console import Console

console = Console()

# ── Marker colours ──────────────────────────────────────────────────────────
# Add any new marker label here; unknown labels fall back to FALLBACK_COLOR.
MARKER_COLORS = {
    "FTI_BALLISTIC_ERROR":  "#e63946",   # red
    "CONTROLLED_RESPONSE":  "#2a9d8f",   # teal
    "CORRECT_INHIBITION":   "#457b9d",   # steel blue
    "STIM_GO":              "#f4a261",   # orange
    "STIM_NOGO":            "#a8dadc",   # light cyan
    "BLOCK_START":          "#6d6875",   # muted purple
}
FALLBACK_COLOR = "#aaaaaa"

# ── Plot style ───────────────────────────────────────────────────────────────
STYLE = {
    "fig_width":    14,       # inches
    "row_height":   1.4,      # inches per EEG channel
    "dpi":          220,      # poster-quality; drop to 100 for quick review
    "linewidth":    0.7,
    "marker_alpha": 0.75,
    "marker_lw":    1.2,
    "font_family":  "sans-serif",
    "font_size":    9,
    "label_size":   8,
    "title_size":   11,
    "bg_color":     "white",
    "grid_alpha":   0.18,
    "ch_offset_sd": 4.0,      # vertical separation between channels in std-dev units
}


# ── 1. Load ──────────────────────────────────────────────────────────────────

def load_xdf(filepath: str):
    """Parse .xdf and return EEG + marker data.

    Identical contract to rp_beta_analysis.load_xdf — reuse or replace.

    Returns
    -------
    eeg   : np.ndarray  (n_channels, n_samples)  µV
    times : np.ndarray  (n_samples,)             seconds, absolute LSL clock
    ch_names : list[str]
    sfreq    : float
    marker_labels : list[str]
    marker_times  : np.ndarray (n_markers,)
    """
    streams, _ = pyxdf.load_xdf(filepath)

    eeg_stream = marker_stream = None
    for s in streams:
        t = s["info"]["type"][0].lower()
        if t == "eeg":
            eeg_stream = s
        elif t == "markers":
            marker_stream = s

    if eeg_stream is None:
        raise RuntimeError("No EEG stream found in .xdf file.")

    eeg   = eeg_stream["time_series"].T            # (n_ch, n_samples)
    times = eeg_stream["time_stamps"]              # absolute LSL seconds
    sfreq = float(eeg_stream["info"]["nominal_srate"][0])
    ch_names = [
        ch["label"][0]
        for ch in eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
    ]

    marker_labels, marker_times = [], np.array([])
    if marker_stream is not None:
        marker_labels = [m[0] for m in marker_stream["time_series"]]
        marker_times  = marker_stream["time_stamps"]

    console.log(
        f"[green]Loaded[/green]  {eeg.shape[0]} ch × {eeg.shape[1]} samples  "
        f"@ {sfreq:.0f} Hz  |  {len(marker_labels)} markers"
    )
    return eeg, times, ch_names, sfreq, marker_labels, marker_times


# ── 2. Plot ──────────────────────────────────────────────────────────────────

def plot_eeg_markers(
    eeg: np.ndarray,
    times: np.ndarray,
    ch_names: list[str],
    sfreq: float,
    marker_labels: list[str],
    marker_times: np.ndarray,
    title: str = "",
    seconds: float | None = None,
    outpath: str | None = None,
    show: bool = True,
) -> Path | None:
    """Draw stacked EEG traces with colour-coded marker lines.

    Each channel is z-scored and offset vertically so traces stay legible
    even at high amplitude (movement artefacts etc.).

    Parameters
    ----------
    eeg           : (n_ch, n_samples) in µV
    times         : absolute LSL timestamps for each sample
    ch_names      : channel labels, same order as eeg rows
    sfreq         : sampling rate
    marker_labels : string labels, one per marker event
    marker_times  : absolute LSL timestamps for each marker
    title         : figure title (appears at top)
    seconds       : how many seconds to show (None = full recording)
    outpath       : if given, saves a PNG here
    show          : if True, opens an interactive window

    Returns
    -------
    Path to saved PNG, or None if not saved.
    """
    n_ch = eeg.shape[0]
    t0   = times[0]                                # anchor to t=0

    rel_times = times - t0                         # seconds from recording start
    win = rel_times[-1] if seconds is None else min(float(seconds), rel_times[-1])
    mask = rel_times <= win
    rel_times = rel_times[mask]
    eeg       = eeg[:, mask]

    # ── figure layout ────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":  STYLE["font_family"],
        "font.size":    STYLE["font_size"],
        "figure.facecolor": STYLE["bg_color"],
        "axes.facecolor":   STYLE["bg_color"],
    })

    fig_h = STYLE["row_height"] * n_ch + 1.2      # extra space for title + x-axis
    fig, ax = plt.subplots(figsize=(STYLE["fig_width"], fig_h))

    # ── channel traces ────────────────────────────────────────────────────────
    # Compute a common scale: median std across channels (robust to outlier channels)
    stds = np.array([np.std(eeg[i]) for i in range(n_ch)])
    stds[stds == 0] = 1.0
    scale = np.median(stds) * STYLE["ch_offset_sd"]   # pixels between channel baselines

    y_ticks, y_labels = [], []
    for i, ch in enumerate(ch_names):
        offset = (n_ch - 1 - i) * scale               # top channel drawn first
        trace  = (eeg[i] - np.mean(eeg[i])) / stds[i] * (scale / STYLE["ch_offset_sd"])
        ax.plot(
            rel_times,
            trace + offset,
            linewidth=STYLE["linewidth"],
            color="#1d3557",
            rasterized=True,                           # rasterize traces for smaller PDF/PNG
        )
        y_ticks.append(offset)
        y_labels.append(ch)

    # ── marker lines ─────────────────────────────────────────────────────────
    seen_labels = {}                                   # track which labels got a legend entry
    for label, mtime in zip(marker_labels, marker_times):
        rel_mt = mtime - t0
        if rel_mt < 0 or rel_mt > win:
            continue
        color = MARKER_COLORS.get(label, FALLBACK_COLOR)
        lw    = STYLE["marker_lw"]
        # Only first occurrence of each label gets added to legend
        if label not in seen_labels:
            ax.axvline(rel_mt, color=color, linewidth=lw,
                       alpha=STYLE["marker_alpha"], label=label)
            seen_labels[label] = True
        else:
            ax.axvline(rel_mt, color=color, linewidth=lw,
                       alpha=STYLE["marker_alpha"])

    # ── axes decoration ───────────────────────────────────────────────────────
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=STYLE["label_size"])
    ax.set_xlabel("Time (s)", fontsize=STYLE["font_size"])
    ax.set_xlim(0, win)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax.grid(axis="x", alpha=STYLE["grid_alpha"], linewidth=0.5)
    ax.spines[["top", "right", "left"]].set_visible(False)

    fig_title = title or Path(outpath).stem if outpath else "EEG + Markers"
    ax.set_title(fig_title, fontsize=STYLE["title_size"], pad=8, fontweight="bold")

    if seen_labels:
        ax.legend(
            loc="upper right",
            fontsize=STYLE["label_size"],
            framealpha=0.85,
            edgecolor="#cccccc",
        )

    # ── info annotation ───────────────────────────────────────────────────────
    ax.annotate(
        f"{sfreq:.0f} Hz  ·  {n_ch} ch  ·  {win:.1f} s shown",
        xy=(0.01, 0.01), xycoords="axes fraction",
        fontsize=7, color="#888888", va="bottom",
    )

    fig.tight_layout(pad=0.8)

    saved = None
    if outpath:
        os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
        fig.savefig(outpath, dpi=STYLE["dpi"], bbox_inches="tight")
        saved = Path(outpath)
        console.log(f"[green]Saved[/green]  {saved}  ({STYLE['dpi']} dpi)")

    if show:
        matplotlib.use("TkAgg")   # switch to interactive backend for display
        plt.show()
    plt.close(fig)
    return saved


# ── 3. CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick EEG + marker poster plot from a .xdf file"
    )
    parser.add_argument("--file",    required=True,        help="Path to .xdf recording")
    parser.add_argument("--outdir",  default="./output",   help="Output directory (default: ./output)")
    parser.add_argument("--seconds", type=float, default=None,
                        help="Seconds of data to show (default: full recording)")
    parser.add_argument("--title",   default="",           help="Figure title")
    parser.add_argument("--no-show", action="store_true",  help="Skip interactive window; save only")
    args = parser.parse_args()

    console.rule("[bold cyan]quick_xdf_plot[/bold cyan]")

    eeg, times, ch_names, sfreq, marker_labels, marker_times = load_xdf(args.file)

    stem    = Path(args.file).stem
    outpath = str(Path(args.outdir) / f"{stem}_eeg_markers.png")

    plot_eeg_markers(
        eeg, times, ch_names, sfreq,
        marker_labels, marker_times,
        title   = args.title or stem,
        seconds = args.seconds,
        outpath = outpath,
        show    = not args.no_show,
    )

    console.rule("[bold green]Done[/bold green]")


if __name__ == "__main__":
    main()
