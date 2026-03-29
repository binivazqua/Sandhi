
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np

# Bandas de frecuencia TÍPICAS, de acuerdo con Kilmesh (1999), Kornhuber y Deckee (1965) y Libet (1983)
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 45.0),
}


# Dataclass: arreglo aesthetic para objetos en python. Genera automáticamente init, repr, eq, etc. Ideal para estructuras de datos como esta.
@dataclass
class EasyRecording:
    """
    Estructura custom de datos que contiene la señal en formato .easy y su metadada.
    """
    path: Path # ruta de origen
    info_path: Path | None 
    raw: mne.io.RawArray # objeto compatible con MNE con la señal en Volts.
    raw: mne.io.RawArray 
    sfreq: float # frecuencia de muestreo de señal
    channel_names: list[str] # nombres de canales, idealmente extraídos de .info
    eeg_uv: np.ndarray # señal EEG en microvolts, extraída de raw
    markers: np.ndarray #eventos (markers)
    timestamps_ms: np.ndarray
    accelerometer: np.ndarray | None
    metadata: dict[str, Any] #dict de metadata default de NIC2


def parse_args() -> argparse.Namespace:
    """
    Creamos un parser de args (un objeto improvisado donde 
    se guardan atributos) para el cli y manejar input y 
    outputs del script de forma flexible.

    """
    parser = argparse.ArgumentParser(
        description=(
            "Compara archivos NIC2 .easy/.info contra .edf y genera un reporte "
            "con métricas, metadatos e interpretación."
        )
    )
    parser.add_argument("--easy", type=Path, help="Ruta al archivo .easy")
    parser.add_argument("--edf", type=Path, help="Ruta al archivo .edf")
    parser.add_argument(
        "--info",
        type=Path,
        help="Ruta al archivo .info. Si se omite, se intentará resolver por nombre.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Carpeta donde buscar pares .easy/.edf cuando no se pasen rutas explícitas.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("comparison_reports"),
        help="Carpeta de salida para CSV/JSON/Markdown.",
    )
    parser.add_argument(
        "--pair-mode",
        choices=("explicit", "stem"),
        default="stem",
        help="Cómo emparejar archivos cuando se busca en --data-dir.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=120.0,
        help="Ventana máxima a usar para la comparación de señal.",
    )
    return parser.parse_args()


def normalize_name(name: str) -> str:
    """
    Convierte un nombre de canal a una forma normalizada: 
    minúsculas y solo letras/números. 
    Iguala formatos .easy y .edf.

    """
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def read_info_metadata(info_path: Path | None) -> dict[str, Any]:
    """ 
    Crea un dict de metadata a partir del contenido de un archivo .info, 
    buscando patrones comunes (facilito).
    """
    metadata: dict[str, Any] = {
        "channels": [],
        "sampling_rate_hz": None,
        "device": None,
        "accelerometer_declared": None,
        "raw_lines": [],
    }
    if info_path is None or not info_path.exists():
        return metadata

    channel_pattern = re.compile(r"Channel\s+\d+.*?([A-Za-z][A-Za-z0-9_-]*)\s*$")
    number_pattern = re.compile(r"(-?\d+(?:\.\d+)?)")
    with info_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            metadata["raw_lines"].append(line)
            lowered = line.lower()
            match = channel_pattern.search(line)
            if match:
                metadata["channels"].append(match.group(1))
            if metadata["sampling_rate_hz"] is None and any(
                key in lowered for key in ("sampling", "sample rate", "frequency", "sfreq", "hz")
            ):
                num_match = number_pattern.search(line)
                if num_match:
                    metadata["sampling_rate_hz"] = float(num_match.group(1))
            if metadata["device"] is None and any(key in lowered for key in ("enobio", "starstim", "device")):
                metadata["device"] = line
            if "accelerometer data" in lowered:
                metadata["accelerometer_declared"] = "yes" in lowered or "true" in lowered
    return metadata


def infer_easy_layout(num_cols: int) -> tuple[int, bool]:
    """
    Fail safe de inferencia de formato .easy basado en número de columnas.
    """
    if num_cols in (10, 22, 34):
        return num_cols - 2, False
    if num_cols in (13, 25, 37):
        return num_cols - 5, True
    raise ValueError(
        f"Formato .easy no reconocido: {num_cols} columnas. "
        "Se esperaban 10/13/22/25/34/37."
    )


def infer_sfreq_from_timestamps(timestamps_ms: np.ndarray) -> float | None:
    """
    Estima la frecuencia de muestreo a partir de variaciones en las timestamps. 
    """
    if timestamps_ms.size < 2:
        return None
    diffs = np.diff(timestamps_ms.astype(float))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    median_ms = float(np.median(diffs))
    if median_ms <= 0:
        return None
    return 1000.0 / median_ms

## ********************* CARGA DE DATOS ***************************** #

def load_easy_as_raw(easy_path: Path, info_path: Path | None = None) -> EasyRecording:
    """
    Junta todo lo anterior y usa la estructura EasyRecording para cargar un .easy y su .info asociado (si existe),
    """
    if not easy_path.exists():
        raise FileNotFoundError(f"No existe el archivo .easy: {easy_path}")

    resolved_info = info_path or easy_path.with_suffix(".info")
    if resolved_info and not resolved_info.exists():
        resolved_info = None

    metadata = read_info_metadata(resolved_info)
    frame = np.loadtxt(easy_path)
    if frame.ndim == 1:
        frame = frame[np.newaxis, :]
    num_channels, has_acc = infer_easy_layout(frame.shape[1])

    channel_names = metadata["channels"][:num_channels]
    if len(channel_names) != num_channels:
        channel_names = [f"Ch{i}" for i in range(1, num_channels + 1)]

    eeg_uv = frame[:, :num_channels].astype(float) / 1000.0 # CONVERTIR A PICO VOLTS: CLAVE
    cursor = num_channels
    accelerometer = None
    if has_acc:
        accelerometer = frame[:, cursor : cursor + 3].astype(float)
        cursor += 3
    markers = frame[:, cursor].astype(float)
    timestamps_ms = frame[:, cursor + 1].astype(float)

    sfreq = metadata["sampling_rate_hz"] or infer_sfreq_from_timestamps(timestamps_ms) or 500.0
    info = mne.create_info(channel_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray((eeg_uv * 1e-6).T, info, verbose="ERROR")

    metadata.update(
        {
            "num_channels": num_channels,
            "has_accelerometer_columns": has_acc,
            "inferred_sampling_rate_hz": sfreq,
            "start_timestamp_ms": float(timestamps_ms[0]) if timestamps_ms.size else None,
            "end_timestamp_ms": float(timestamps_ms[-1]) if timestamps_ms.size else None,
            "num_samples": int(frame.shape[0]),
        }
    )

    return EasyRecording(
        path=easy_path,
        info_path=resolved_info,
        raw=raw,
        sfreq=sfreq,
        channel_names=channel_names,
        eeg_uv=eeg_uv,
        markers=markers,
        timestamps_ms=timestamps_ms,
        accelerometer=accelerometer,
        metadata=metadata,
    )


def load_edf(edf_path: Path) -> mne.io.BaseRaw:
    """
    Wrapper de mne, para convertir un archivo .edf a un objeto Raw de MNE.
    """
    if not edf_path.exists():
        raise FileNotFoundError(f"No existe el archivo .edf: {edf_path}")
    return mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")


# ******************** COMPARACIÓN DE EDF Y EASY ************************* #

def pick_common_channels(easy_raw: mne.io.BaseRaw, edf_raw: mne.io.BaseRaw) -> tuple[list[str], list[str]]:
    edf_lookup = {normalize_name(name): name for name in edf_raw.ch_names}
    easy_common: list[str] = []
    edf_common: list[str] = []
    for easy_name in easy_raw.ch_names:
        edf_name = edf_lookup.get(normalize_name(easy_name))
        if edf_name is not None:
            easy_common.append(easy_name)
            edf_common.append(edf_name)
    return easy_common, edf_common


def prepare_comparison_arrays(
    easy_raw: mne.io.BaseRaw,
    edf_raw: mne.io.BaseRaw,
    easy_chs: list[str],
    edf_chs: list[str],
    max_seconds: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    easy_data = easy_raw.copy().pick(easy_chs)
    edf_data = edf_raw.copy().pick(edf_chs)

    target_sfreq = min(float(easy_data.info["sfreq"]), float(edf_data.info["sfreq"]))
    if abs(easy_data.info["sfreq"] - target_sfreq) > 1e-6:
        easy_data.resample(target_sfreq, verbose="ERROR")
    if abs(edf_data.info["sfreq"] - target_sfreq) > 1e-6:
        edf_data.resample(target_sfreq, verbose="ERROR")

    max_samples = int(target_sfreq * max_seconds)
    usable_samples = min(easy_data.n_times, edf_data.n_times, max_samples)
    return (
        easy_data.get_data()[:, :usable_samples],
        edf_data.get_data()[:, :usable_samples],
        target_sfreq,
    )

# ********************* MÉTRICAS ESTADÍSTICAS CLAVE ************************+ #

def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula correlación de pearson (correlación lineal) entre las señales de 
    el formato .easy y el .edf para un canal dado, con manejo de casos edge (señales constantes o vacías).
    """
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def summarize_bandpower(raw: mne.io.BaseRaw) -> dict[str, float]:
    """
    Se avienta el resumen de PSD por banda, usando el método de Welch de MNE, para un objeto Raw dado.
    """
    spectrum = raw.compute_psd(fmin=1.0, fmax=45.0, verbose="ERROR")
    psd, freqs = spectrum.get_data(return_freqs=True)
    summary: dict[str, float] = {}
    for band_name, (fmin, fmax) in BANDS.items():
        mask = (freqs >= fmin) & (freqs < fmax)
        if not np.any(mask):
            summary[band_name] = float("nan")
            continue
        summary[band_name] = float(np.mean(psd[:, mask]))
    return summary


def edf_metadata(raw: mne.io.BaseRaw) -> dict[str, Any]:
    annotations = raw.annotations
    meas_date = raw.info.get("meas_date")
    if meas_date is not None:
        try:
            meas_date = meas_date.isoformat()
        except AttributeError:
            meas_date = str(meas_date)
    return {
        "num_channels": len(raw.ch_names),
        "channel_names": raw.ch_names,
        "sampling_rate_hz": float(raw.info["sfreq"]),
        "duration_seconds": float(raw.n_times / raw.info["sfreq"]),
        "highpass_hz": float(raw.info.get("highpass", np.nan)),
        "lowpass_hz": float(raw.info.get("lowpass", np.nan)),
        "measurement_date": meas_date,
        "annotations_count": len(annotations),
        "annotation_descriptions": sorted(set(annotations.description.tolist())),
        "subject_info_present": raw.info.get("subject_info") is not None,
    }


def easy_metadata(recording: EasyRecording) -> dict[str, Any]:
    duration_seconds = (
        float((recording.timestamps_ms[-1] - recording.timestamps_ms[0]) / 1000.0)
        if recording.timestamps_ms.size > 1
        else 0.0
    )
    return {
        "num_channels": recording.metadata["num_channels"],
        "channel_names": recording.channel_names,
        "sampling_rate_hz": float(recording.sfreq),
        "duration_seconds": duration_seconds,
        "marker_count_nonzero": int(np.count_nonzero(recording.markers)),
        "accelerometer_present": recording.accelerometer is not None,
        "info_file_present": recording.info_path is not None,
        "device_hint": recording.metadata.get("device"),
        "start_timestamp_ms": recording.metadata.get("start_timestamp_ms"),
        "end_timestamp_ms": recording.metadata.get("end_timestamp_ms"),
    }


def compare_recordings(
    easy_recording: EasyRecording,
    edf_raw: mne.io.BaseRaw,
    max_seconds: float,
) -> dict[str, Any]:
    """
    Función central que genera el reporte completo.
    Pasos:
    - Inicializa estructura de comparación (metadatos, canales comunes, etc.).
    - Calcula bandpower para .easy y .edf.
    - Por cada canal común:
        - Calcula correlación, std, RMSE, diferencia media, peak-to-peak.
        - Almacena en channel_metrics.
        - Agrega métricas: mean_corr, median_corr, mean_rmse, mean_abs_diff.
        - Genera interpretaciones basadas en umbrales (ej. "corr < 0.8 → revisar").
    - Retorna dict estructurado.
    """
    easy_common, edf_common = pick_common_channels(easy_recording.raw, edf_raw)

    comparison: dict[str, Any] = {
        "easy": easy_metadata(easy_recording),
        "edf": edf_metadata(edf_raw),
        "common_channels": easy_common,
        "common_channel_count": len(easy_common),
        "channel_metrics": [],
        "interpretation": [],
    }

    comparison["easy"]["bandpower"] = summarize_bandpower(easy_recording.raw)
    comparison["edf"]["bandpower"] = summarize_bandpower(edf_raw.copy().pick("eeg"))

    if not easy_common:
        comparison["interpretation"].append(
            "No hubo nombres de canal coincidentes entre .easy y .edf; revisa etiquetas o renombrado."
        )
        return comparison

    easy_data, edf_data, target_sfreq = prepare_comparison_arrays(
        easy_recording.raw,
        edf_raw,
        easy_common,
        edf_common,
        max_seconds=max_seconds,
    )
    comparison["comparison_sampling_rate_hz"] = target_sfreq
    comparison["compared_duration_seconds"] = float(easy_data.shape[1] / target_sfreq)

    for idx, easy_name in enumerate(easy_common):
        edf_name = edf_common[idx]
        easy_v = easy_data[idx]
        edf_v = edf_data[idx]
        diff_v = easy_v - edf_v
        comparison["channel_metrics"].append(
            {
                "easy_channel": easy_name,
                "edf_channel": edf_name,
                "corr": safe_corrcoef(easy_v, edf_v),
                "easy_std_uv": float(np.std(easy_v) * 1e6),
                "edf_std_uv": float(np.std(edf_v) * 1e6),
                "rmse_uv": float(np.sqrt(np.mean(diff_v**2)) * 1e6),
                "mean_abs_diff_uv": float(np.mean(np.abs(diff_v)) * 1e6),
                "easy_peak_to_peak_uv": float(np.ptp(easy_v) * 1e6),
                "edf_peak_to_peak_uv": float(np.ptp(edf_v) * 1e6),
            }
        )

    metrics = comparison["channel_metrics"]
    comparison["aggregate_metrics"] = {
        "mean_corr": float(np.nanmean([item["corr"] for item in metrics])),
        "median_corr": float(np.nanmedian([item["corr"] for item in metrics])),
        "mean_rmse_uv": float(np.nanmean([item["rmse_uv"] for item in metrics])),
        "mean_abs_diff_uv": float(np.nanmean([item["mean_abs_diff_uv"] for item in metrics])),
    }

    if easy_recording.info_path is not None:
        comparison["interpretation"].append(
            "El .easy conserva marcadores por muestra, timestamps Unix y, cuando existe, acelerometría; eso es útil para "
            "alineación temporal fina y control de artefactos por movimiento."
        )
    else:
        comparison["interpretation"].append(
            "El .easy puede abrirse sin .info, pero pierdes parte del contexto experimental, sobre todo nombres de canal "
            "fiables y configuración declarada."
        )

    if comparison["edf"]["annotations_count"] > 0:
        comparison["interpretation"].append(
            "El .edf aporta un contenedor más interoperable en MNE y suele preservar mejor metadatos clínicos/anotaciones."
        )
    else:
        comparison["interpretation"].append(
            "El .edf es el formato más portable para análisis y archivado, aunque en este archivo no aparecen anotaciones."
        )

    if comparison["aggregate_metrics"]["mean_corr"] < 0.8:
        comparison["interpretation"].append(
            "La similitud promedio entre señales es modesta; conviene revisar referencia, escalado, filtros aplicados al exportar "
            "y correspondencia exacta de canales."
        )
    else:
        comparison["interpretation"].append(
            "Las señales coinciden razonablemente bien; eso sugiere que .easy y .edf representan la misma adquisición con "
            "diferencias menores de formato o preprocesado."
        )

    return comparison


def build_markdown_report(result: dict[str, Any], easy_path: Path, edf_path: Path) -> str:
    easy_meta = result["easy"]
    edf_meta = result["edf"]
    agg = result.get("aggregate_metrics", {})
    lines = [
        f"# Comparación NIC2: `{easy_path.name}` vs `{edf_path.name}`",
        "",
        "## Resumen",
        f"- Canales comunes: {result['common_channel_count']}",
        f"- Frecuencia de muestreo .easy: {easy_meta['sampling_rate_hz']:.3f} Hz",
        f"- Frecuencia de muestreo .edf: {edf_meta['sampling_rate_hz']:.3f} Hz",
        f"- Duración .easy: {easy_meta['duration_seconds']:.3f} s",
        f"- Duración .edf: {edf_meta['duration_seconds']:.3f} s",
    ]
    if agg:
        lines.extend(
            [
                f"- Correlación media: {agg['mean_corr']:.4f}",
                f"- RMSE medio: {agg['mean_rmse_uv']:.3f} uV",
                f"- Diferencia absoluta media: {agg['mean_abs_diff_uv']:.3f} uV",
            ]
        )
    lines.extend(
        [
            "",
            "## Qué aporta cada formato",
            "- `.easy`: amplitudes tabulares por muestra, timestamps Unix, marcadores y a veces acelerometría. Es muy útil para auditoría del export, sincronización y trazabilidad de eventos.",
            "- `.info`: orden real de electrodos, frecuencia de muestreo declarada y pistas del dispositivo/configuración.",
            "- `.edf`: formato estándar, portable y listo para ecosistemas como MNE/EEGLAB; suele facilitar intercambio, anotación y archivado reproducible.",
            "",
            "## Interpretación",
        ]
    )
    lines.extend(f"- {item}" for item in result["interpretation"])
    lines.extend(["", "## Bandas de potencia promedio"])
    for band_name in BANDS:
        lines.append(
            f"- {band_name}: .easy={easy_meta['bandpower'][band_name]:.6e}, "
            f".edf={edf_meta['bandpower'][band_name]:.6e}"
        )
    return "\n".join(lines) + "\n"


def stem_key(path: Path) -> str:
    return normalize_name(path.stem)


def discover_pairs(data_dir: Path) -> list[tuple[Path, Path, Path | None]]:
    easy_files = sorted(data_dir.rglob("*.easy"))
    edf_files = sorted(data_dir.rglob("*.edf"))
    info_files = {stem_key(path): path for path in data_dir.rglob("*.info")}
    edf_by_stem = {stem_key(path): path for path in edf_files}

    pairs: list[tuple[Path, Path, Path | None]] = []
    for easy_path in easy_files:
        key = stem_key(easy_path)
        edf_path = edf_by_stem.get(key)
        if edf_path is None:
            continue
        pairs.append((easy_path, edf_path, info_files.get(key)))
    return pairs


def discover_easy_self_pairs(data_dir: Path) -> list[tuple[Path, Path, Path | None]]:
    easy_files = sorted(data_dir.rglob("*.easy"))
    info_files = {stem_key(path): path for path in data_dir.rglob("*.info")}

    pairs: list[tuple[Path, Path, Path | None]] = []
    for easy_path in easy_files:
        if easy_path.stem.endswith("_edfFiltered"):
            continue
        pairs.append((easy_path, easy_path, info_files.get(stem_key(easy_path))))
    return pairs


def write_outputs(result: dict[str, Any], easy_path: Path, edf_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{easy_path.stem}__vs__{edf_path.stem}"

    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    csv_path = output_dir / f"{stem}_channels.csv"

    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    md_path.write_text(build_markdown_report(result, easy_path, edf_path), encoding="utf-8")
    channel_metrics = result.get("channel_metrics", [])
    if channel_metrics:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(channel_metrics[0].keys()))
            writer.writeheader()
            writer.writerows(channel_metrics)
    else:
        csv_path.write_text("", encoding="utf-8")


def print_console_summary(result: dict[str, Any], easy_path: Path, edf_path: Path) -> None:
    """
    Imprime resumen en terminal: canales comunes, sfreq, correlación media, RMSE.
    Genera el yappeo. 
    """
    print(f"\nComparacion: {easy_path.name} vs {edf_path.name}")
    print(f"Canales comunes: {result['common_channel_count']}")
    print(f"Sfreq .easy: {result['easy']['sampling_rate_hz']:.3f} Hz")
    print(f"Sfreq .edf: {result['edf']['sampling_rate_hz']:.3f} Hz")
    if "aggregate_metrics" in result:
        print(f"Correlacion media: {result['aggregate_metrics']['mean_corr']:.4f}")
        print(f"RMSE medio: {result['aggregate_metrics']['mean_rmse_uv']:.3f} uV")
    print("Interpretacion:")
    for item in result["interpretation"]:
        print(f"  - {item}")


def run_single_pair(easy_path: Path, edf_path: Path, info_path: Path | None, args: argparse.Namespace) -> None:
    easy_recording = load_easy_as_raw(easy_path, info_path)
    edf_raw = easy_recording.raw.copy() if easy_path.resolve() == edf_path.resolve() else load_edf(edf_path)
    result = compare_recordings(easy_recording, edf_raw, max_seconds=args.max_seconds)
    write_outputs(result, easy_path, edf_path, args.output_dir)
    print_console_summary(result, easy_path, edf_path)


def resolve_input_pairs(args: argparse.Namespace) -> list[tuple[Path, Path, Path | None]]:
    if args.easy or args.edf:
        if not (args.easy and args.edf):
            raise ValueError("Si usas --easy o --edf, debes pasar ambos.")
        return [(args.easy, args.edf, args.info)]

    pairs = discover_pairs(args.data_dir)
    if not pairs:
        pairs = discover_easy_self_pairs(args.data_dir)
    if not pairs:
        raise FileNotFoundError(
            "No se encontraron pares .easy/.edf con el mismo stem en la carpeta indicada. "
            "Usa --easy/--edf explícitos o coloca los archivos en --data-dir."
        )
    return pairs


def main() -> None:
    args = parse_args()
    pairs = resolve_input_pairs(args)
    for easy_path, edf_path, info_path in pairs:
        run_single_pair(easy_path, edf_path, info_path, args)


if __name__ == "__main__":
    main()
