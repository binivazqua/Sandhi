#!/usr/bin/env python3
"""Visual and statistical comparison for NeuroElectrics `.easy` vs `.edf`."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import welch
from scipy.stats import iqr, linregress, median_abs_deviation, spearmanr

from edf_conv import load_easy_as_raw, load_edf, pick_common_channels, prepare_comparison_arrays


SOURCE_NOTES = [
    {
        "title": "NIC2 User Manual",
        "url": "https://www.neuroelectrics.com/api/downloads/NE_P3_UM004_EN_NIC2.1.0_1.pdf",
        "note": "NIC2 states `.nedf` is 24-bit, `.edf` is 16-bit, and the DC component is filtered in `.edf` exports.",
    },
    {
        "title": "Neuroelectrics EEGLAB Plugin",
        "url": "https://www.neuroelectrics.com/eeglab-plugin",
        "note": "Neuroelectrics exposes `.easy` and `.nedf` as native analysis-ready inputs for their tooling.",
    },
    {
        "title": "Enobio 8 Product Page",
        "url": "https://www.neuroelectrics.com/products/research/enobio/enobio8",
        "note": "Neuroelectrics advertises 24-bit signal resolution, 0.05 uV resolution, and sample-precision data storage.",
    },
]


@dataclass
class AnalysisJob:
    easy: Path
    edf: Path
    info: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera graficas y estadisticos para comparar `.easy` y `.edf`, "
            "con enfasis en preservar informacion util para potenciales lentos."
        )
    )
    parser.add_argument("--easy", type=Path, help="Ruta al archivo .easy")
    parser.add_argument("--edf", type=Path, help="Ruta al archivo .edf")
    parser.add_argument("--info", type=Path, help="Ruta al archivo .info")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Activa modo interactivo para seleccionar archivos sin escribir rutas complejas.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Carpeta base usada por el modo interactivo para listar candidatos.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Numero de pares a procesar en modo interactivo.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("comparison_reports/graphs"),
        help="Carpeta de salida para PNG/CSV/JSON/MD.",
    )
    parser.add_argument(
        "--channels",
        help="Lista separada por comas de canales a graficar. Por defecto usa todos los comunes.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=20.0,
        help="Ventana maxima para comparacion y graficas.",
    )
    parser.add_argument(
        "--plot-seconds",
        type=float,
        default=10.0,
        help="Segundos visibles en las graficas de senal cruda.",
    )
    return parser.parse_args()


def sanitize_stem(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def _prompt_int(message: str, default: int, minimum: int = 1) -> int:
    while True:
        raw = input(f"{message} [{default}]: ").strip()
        if not raw:
            return default
        if raw.isdigit() and int(raw) >= minimum:
            return int(raw)
        print(f"Entrada invalida. Ingresa un entero >= {minimum}.")


def _select_path_interactive(
    label: str,
    candidates: list[Path],
    allow_empty: bool = False,
    default: Path | None = None,
) -> Path | None:
    print(f"\n{label}")
    if default is not None:
        print(f"  Default: {default}")

    if candidates:
        for idx, path in enumerate(candidates, start=1):
            print(f"  [{idx}] {path}")
        print("  Puedes ingresar el numero o pegar una ruta completa.")
    else:
        print("  No hay candidatos listados. Pega una ruta completa.")

    if allow_empty:
        print("  Enter vacio para omitir.")

    while True:
        raw = input("> ").strip()
        if not raw:
            if default is not None:
                return default
            if allow_empty:
                return None
            print("Este valor es obligatorio.")
            continue

        if raw.isdigit() and candidates:
            idx = int(raw)
            if 1 <= idx <= len(candidates):
                return candidates[idx - 1]
            print(f"Indice fuera de rango. Usa 1..{len(candidates)}")
            continue

        chosen = Path(raw).expanduser()
        if chosen.exists() and chosen.is_file():
            return chosen
        print("Ruta invalida o archivo inexistente.")


def _discover_candidates(data_dir: Path, pattern: str) -> list[Path]:
    if not data_dir.exists():
        return []
    return sorted(path for path in data_dir.rglob(pattern) if path.is_file())


def _resolve_jobs(args: argparse.Namespace) -> list[AnalysisJob]:
    explicit_mode = args.easy is not None or args.edf is not None
    if explicit_mode:
        if not (args.easy and args.edf):
            raise ValueError("Si pasas --easy o --edf, debes pasar ambos.")
        return [AnalysisJob(args.easy, args.edf, args.info)]

    data_dir = args.data_dir
    easy_candidates = _discover_candidates(data_dir, "*.easy")
    edf_candidates = _discover_candidates(data_dir, "*.edf")

    print("Modo interactivo de seleccion de rutas")
    print(f"Base de busqueda: {data_dir}")
    print(f"Candidatos .easy: {len(easy_candidates)}")
    print(f"Candidatos .edf: {len(edf_candidates)}")

    jobs_count = _prompt_int("Cuantos pares quieres analizar", default=max(args.jobs, 1), minimum=1)
    jobs: list[AnalysisJob] = []
    for idx in range(1, jobs_count + 1):
        print(f"\n=== Par {idx}/{jobs_count} ===")
        easy_path = _select_path_interactive("Selecciona archivo .easy:", easy_candidates, allow_empty=False)
        assert easy_path is not None
        info_default = easy_path.with_suffix(".info") if easy_path.with_suffix(".info").exists() else None
        edf_path = _select_path_interactive("Selecciona archivo .edf:", edf_candidates, allow_empty=False)
        assert edf_path is not None
        info_path = _select_path_interactive(
            "Selecciona archivo .info (opcional):",
            _discover_candidates(data_dir, "*.info"),
            allow_empty=True,
            default=info_default,
        )
        jobs.append(AnalysisJob(easy=easy_path, edf=edf_path, info=info_path))

    return jobs


def select_channels(common_easy: list[str], common_edf: list[str], requested: str | None) -> tuple[list[str], list[str]]:
    if not requested:
        return common_easy, common_edf
    wanted = [item.strip() for item in requested.split(",") if item.strip()]
    edf_lookup = {name.lower(): name for name in common_edf}
    easy_selected: list[str] = []
    edf_selected: list[str] = []
    for ch in common_easy:
        if ch.lower() in {item.lower() for item in wanted}:
            easy_selected.append(ch)
            edf_selected.append(edf_lookup[ch.lower()])
    if not easy_selected:
        raise ValueError("Ningun canal solicitado coincide entre `.easy` y `.edf`.")
    return easy_selected, edf_selected


def quantization_step_uv(signal_uv: np.ndarray) -> float:
    rounded = np.unique(np.round(signal_uv.astype(float), 6))
    if rounded.size < 2:
        return float("nan")
    diffs = np.diff(rounded)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return float("nan")
    return float(np.min(diffs))


def bandpower(signal_uv: np.ndarray, sfreq: float, fmin: float, fmax: float) -> float:
    if signal_uv.size < 8:
        return float("nan")
    nperseg = min(signal_uv.size, max(256, int(sfreq * 4)))
    freqs, psd = welch(signal_uv, fs=sfreq, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return float("nan")
    return float(np.trapezoid(psd[mask], freqs[mask]))


def linear_drift_uv_per_s(signal_uv: np.ndarray, sfreq: float) -> float:
    if signal_uv.size < 2:
        return float("nan")
    time_s = np.arange(signal_uv.size) / sfreq
    slope, _, _, _, _ = linregress(time_s, signal_uv)
    return float(slope)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    corr, _ = spearmanr(a, b)
    return float(corr)


def channel_stats(easy_uv: np.ndarray, edf_uv: np.ndarray, sfreq: float, channel: str) -> dict[str, Any]:
    easy_demeaned = easy_uv - np.mean(easy_uv)
    edf_demeaned = edf_uv - np.mean(edf_uv)
    diff_raw = easy_uv - edf_uv
    diff_demeaned = easy_demeaned - edf_demeaned

    return {
        "channel": channel,
        "mean_easy_uv": float(np.mean(easy_uv)),
        "mean_edf_uv": float(np.mean(edf_uv)),
        "median_easy_uv": float(np.median(easy_uv)),
        "median_edf_uv": float(np.median(edf_uv)),
        "variance_easy_uv2": float(np.var(easy_uv)),
        "variance_edf_uv2": float(np.var(edf_uv)),
        "std_easy_uv": float(np.std(easy_uv)),
        "std_edf_uv": float(np.std(edf_uv)),
        "mad_easy_uv": float(median_abs_deviation(easy_uv, scale="normal")),
        "mad_edf_uv": float(median_abs_deviation(edf_uv, scale="normal")),
        "iqr_easy_uv": float(iqr(easy_uv)),
        "iqr_edf_uv": float(iqr(edf_uv)),
        "ptp_easy_uv": float(np.ptp(easy_uv)),
        "ptp_edf_uv": float(np.ptp(edf_uv)),
        "pearson_raw": safe_corr(easy_uv, edf_uv),
        "pearson_demeaned": safe_corr(easy_demeaned, edf_demeaned),
        "spearman_raw": safe_spearman(easy_uv, edf_uv),
        "rmse_raw_uv": float(np.sqrt(np.mean(diff_raw**2))),
        "rmse_demeaned_uv": float(np.sqrt(np.mean(diff_demeaned**2))),
        "mean_abs_diff_raw_uv": float(np.mean(np.abs(diff_raw))),
        "mean_abs_diff_demeaned_uv": float(np.mean(np.abs(diff_demeaned))),
        "variance_ratio_easy_over_edf": float(np.var(easy_uv) / np.var(edf_uv)) if np.var(edf_uv) > 0 else float("nan"),
        "drift_easy_uv_per_s": linear_drift_uv_per_s(easy_uv, sfreq),
        "drift_edf_uv_per_s": linear_drift_uv_per_s(edf_uv, sfreq),
        "slow_power_easy_0_0p5_uv2": bandpower(easy_uv, sfreq, 0.0, 0.5),
        "slow_power_edf_0_0p5_uv2": bandpower(edf_uv, sfreq, 0.0, 0.5),
        "delta_power_easy_0p5_4_uv2": bandpower(easy_uv, sfreq, 0.5, 4.0),
        "delta_power_edf_0p5_4_uv2": bandpower(edf_uv, sfreq, 0.5, 4.0),
        "quant_step_easy_uv": quantization_step_uv(easy_uv),
        "quant_step_edf_uv": quantization_step_uv(edf_uv),
    }


def summary_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_for(key: str) -> float:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float)) and not math.isnan(row[key])]
        if not values:
            return float("nan")
        return float(np.mean(values))

    return {
        "channels": [row["channel"] for row in rows],
        "mean_pearson_raw": mean_for("pearson_raw"),
        "mean_pearson_demeaned": mean_for("pearson_demeaned"),
        "mean_spearman_raw": mean_for("spearman_raw"),
        "mean_rmse_raw_uv": mean_for("rmse_raw_uv"),
        "mean_rmse_demeaned_uv": mean_for("rmse_demeaned_uv"),
        "mean_variance_ratio_easy_over_edf": mean_for("variance_ratio_easy_over_edf"),
        "mean_slow_power_easy_0_0p5_uv2": mean_for("slow_power_easy_0_0p5_uv2"),
        "mean_slow_power_edf_0_0p5_uv2": mean_for("slow_power_edf_0_0p5_uv2"),
        "mean_quant_step_easy_uv": mean_for("quant_step_easy_uv"),
        "mean_quant_step_edf_uv": mean_for("quant_step_edf_uv"),
    }


def plot_overlays(
    easy_uv: np.ndarray,
    edf_uv: np.ndarray,
    sfreq: float,
    channels: list[str],
    output_path: Path,
    title: str,
    demean: bool,
    seconds: float,
) -> None:
    n_channels = len(channels)
    fig, axes = plt.subplots(n_channels, 1, figsize=(14, max(3 * n_channels, 4)), sharex=True)
    if n_channels == 1:
        axes = [axes]

    max_samples = min(easy_uv.shape[1], int(seconds * sfreq))
    time_s = np.arange(max_samples) / sfreq

    for idx, ax in enumerate(axes):
        easy_sig = easy_uv[idx, :max_samples]
        edf_sig = edf_uv[idx, :max_samples]
        if demean:
            easy_sig = easy_sig - np.mean(easy_sig)
            edf_sig = edf_sig - np.mean(edf_sig)

        ax.plot(time_s, easy_sig, label=".easy", linewidth=1.0, alpha=0.9)
        ax.plot(time_s, edf_sig, label=".edf", linewidth=1.0, alpha=0.8)
        ax.set_ylabel(f"{channels[idx]}\nuV")
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_histograms(
    easy_uv: np.ndarray,
    edf_uv: np.ndarray,
    channels: list[str],
    output_path: Path,
    title: str,
) -> None:
    n_channels = len(channels)
    fig, axes = plt.subplots(n_channels, 1, figsize=(14, max(3 * n_channels, 4)))
    if n_channels == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        ax.hist(easy_uv[idx], bins=80, density=True, alpha=0.55, label=".easy")
        ax.hist(edf_uv[idx], bins=80, density=True, alpha=0.55, label=".edf")
        ax.set_ylabel(channels[idx])
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("Amplitude (uV)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_heatmap(rows: list[dict[str, Any]], output_path: Path) -> None:
    channels = [row["channel"] for row in rows]
    metrics = [
        ("pearson_raw", "Pearson raw"),
        ("pearson_demeaned", "Pearson demeaned"),
        ("rmse_raw_uv", "RMSE raw"),
        ("rmse_demeaned_uv", "RMSE demeaned"),
        ("variance_ratio_easy_over_edf", "Var ratio"),
        ("slow_power_easy_0_0p5_uv2", "Slow power easy"),
        ("slow_power_edf_0_0p5_uv2", "Slow power edf"),
        ("quant_step_easy_uv", "Q step easy"),
        ("quant_step_edf_uv", "Q step edf"),
    ]

    matrix = np.array([[row[key] for row in rows] for key, _ in metrics], dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, len(channels) * 1.2), 6))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(channels)))
    ax.set_xticklabels(channels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels([label for _, label in metrics])
    fig.colorbar(image, ax=ax, shrink=0.85)
    ax.set_title("Statistical comparison heatmap")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_markdown_report(
    easy_path: Path,
    edf_path: Path,
    stats_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    comparison_sfreq: float,
    compared_seconds: float,
) -> str:
    best_corr = max(stats_rows, key=lambda item: item["pearson_demeaned"])
    worst_corr = min(stats_rows, key=lambda item: item["pearson_demeaned"])
    lines = [
        f"# Visual and Statistical Comparison: `{easy_path.name}` vs `{edf_path.name}`",
        "",
        "## Why prioritize `.easy` / `.nedf`",
        "- Neuroelectrics documents that `.nedf` is 24-bit, `.edf` is 16-bit, and the DC component is filtered when `.edf` is created.",
        "- For slow pre-impulse or slow cortical potential analysis, that makes `.easy` and especially `.nedf` the safer source of truth for raw amplitude and very-low-frequency content.",
        "",
        "## Run summary",
        f"- Compared channels: {', '.join(summary['channels'])}",
        f"- Sampling rate used for comparison: {comparison_sfreq:.3f} Hz",
        f"- Compared duration: {compared_seconds:.3f} s",
        f"- Mean Pearson correlation, raw: {summary['mean_pearson_raw']:.4f}",
        f"- Mean Pearson correlation, demeaned: {summary['mean_pearson_demeaned']:.4f}",
        f"- Mean Spearman correlation: {summary['mean_spearman_raw']:.4f}",
        f"- Mean RMSE, raw: {summary['mean_rmse_raw_uv']:.3f} uV",
        f"- Mean RMSE, demeaned: {summary['mean_rmse_demeaned_uv']:.3f} uV",
        f"- Mean variance ratio, easy/edf: {summary['mean_variance_ratio_easy_over_edf']:.4f}",
        "",
        "## Interpretation",
        f"- Best shape agreement after demeaning: `{best_corr['channel']}` with Pearson={best_corr['pearson_demeaned']:.4f}.",
        f"- Worst shape agreement after demeaning: `{worst_corr['channel']}` with Pearson={worst_corr['pearson_demeaned']:.4f}.",
        "- If raw RMSE is much larger than demeaned RMSE, the main mismatch is likely DC offset or reference rather than waveform shape.",
        "- If slow-band power (`0-0.5 Hz`) is systematically smaller in `.edf`, that is consistent with the documented DC filtering in EDF export.",
        "- Quantization-step estimates are heuristic, but consistently larger steps in `.edf` are directionally consistent with lower effective resolution.",
        "",
        "## Sources",
    ]
    lines.extend(f"- {item['title']}: {item['url']} ({item['note']})" for item in SOURCE_NOTES)
    return "\n".join(lines) + "\n"


def _run_job(job: AnalysisJob, args: argparse.Namespace) -> None:
    easy_recording = load_easy_as_raw(job.easy, job.info)
    edf_raw = load_edf(job.edf)

    common_easy, common_edf = pick_common_channels(easy_recording.raw, edf_raw)
    easy_chs, edf_chs = select_channels(common_easy, common_edf, args.channels)
    easy_v, edf_v, sfreq = prepare_comparison_arrays(
        easy_recording.raw,
        edf_raw,
        easy_chs,
        edf_chs,
        max_seconds=args.max_seconds,
    )

    easy_uv = easy_v * 1e6
    edf_uv = edf_v * 1e6
    compared_seconds = easy_uv.shape[1] / sfreq

    stats_rows = [
        channel_stats(easy_uv[idx], edf_uv[idx], sfreq, channel)
        for idx, channel in enumerate(easy_chs)
    ]
    summary = summary_stats(stats_rows)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = sanitize_stem(f"{job.easy.stem}__vs__{job.edf.stem}")

    raw_plot = output_dir / f"{stem}_raw_overlay.png"
    demean_plot = output_dir / f"{stem}_demeaned_overlay.png"
    hist_plot = output_dir / f"{stem}_histograms.png"
    heatmap_plot = output_dir / f"{stem}_stats_heatmap.png"
    csv_path = output_dir / f"{stem}_stats.csv"
    json_path = output_dir / f"{stem}_summary.json"
    md_path = output_dir / f"{stem}_report.md"

    plot_overlays(
        easy_uv,
        edf_uv,
        sfreq,
        easy_chs,
        raw_plot,
        title=f"Raw overlay: {job.easy.name} vs {job.edf.name}",
        demean=False,
        seconds=min(args.plot_seconds, compared_seconds),
    )
    plot_overlays(
        easy_uv,
        edf_uv,
        sfreq,
        easy_chs,
        demean_plot,
        title=f"Demeaned overlay: {job.easy.name} vs {job.edf.name}",
        demean=True,
        seconds=min(args.plot_seconds, compared_seconds),
    )
    plot_histograms(
        easy_uv,
        edf_uv,
        easy_chs,
        hist_plot,
        title=f"Amplitude distributions: {job.easy.name} vs {job.edf.name}",
    )
    plot_metric_heatmap(stats_rows, heatmap_plot)

    write_csv(stats_rows, csv_path)
    payload = {
        "easy": str(job.easy),
        "edf": str(job.edf),
        "info": str(job.info) if job.info else None,
        "comparison_sampling_rate_hz": sfreq,
        "compared_duration_seconds": compared_seconds,
        "summary": summary,
        "channels": stats_rows,
        "sources": SOURCE_NOTES,
        "plots": {
            "raw_overlay": str(raw_plot),
            "demeaned_overlay": str(demean_plot),
            "histograms": str(hist_plot),
            "stats_heatmap": str(heatmap_plot),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    md_path.write_text(
        build_markdown_report(job.easy, job.edf, stats_rows, summary, sfreq, compared_seconds),
        encoding="utf-8",
    )

    print(f"Saved raw overlay: {raw_plot}")
    print(f"Saved demeaned overlay: {demean_plot}")
    print(f"Saved histograms: {hist_plot}")
    print(f"Saved heatmap: {heatmap_plot}")
    print(f"Saved stats CSV: {csv_path}")
    print(f"Saved summary JSON: {json_path}")
    print(f"Saved report MD: {md_path}")


def main() -> None:
    args = parse_args()
    jobs = _resolve_jobs(args)

    failures = 0
    for idx, job in enumerate(jobs, start=1):
        print(f"\nProcesando par {idx}/{len(jobs)}")
        print(f"  easy: {job.easy}")
        print(f"  edf:  {job.edf}")
        if job.info is not None:
            print(f"  info: {job.info}")
        try:
            _run_job(job, args)
        except Exception as exc:
            failures += 1
            print(f"Error en par {idx}: {exc}")

    if failures:
        raise RuntimeError(f"Fallaron {failures} de {len(jobs)} pares.")


if __name__ == "__main__":
    main()
