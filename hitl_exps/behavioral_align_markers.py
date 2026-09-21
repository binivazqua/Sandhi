"""
Sandhi Lab — alinear grabación Unicorn (CSV crudo) con el log de marcadores
=============================================================================

Qué hace:
Tu exportación de Unicorn Suite Recorder (CSV) NO trae hora de reloj -- solo
filas de muestras a una frecuencia fija (250 Hz en el Unicorn Hybrid Black).
Tu log de marcadores (sandhi_markers_fallback.py) SÍ trae hora de reloj por
evento. Este script ancla ambos con UNA sola hora de referencia (la hora real
a la que le diste "Record" en Unicorn Suite) y calcula en qué fila / segundo
de la grabación cayó cada marcador -- listo para epochear en tu pipeline.

Uso:
    python sandhi_align_markers.py <eeg.csv> <markers.csv> [HH:MM:SS] [--out salida.csv]

La hora de referencia (recording_start):
- Si tu log de marcadores ya trae una fila RECORDING_START_REFERENCE (la
  función nueva de sandhi_markers_fallback.py), este script la usa sola,
  automático -- no necesitas pasar nada a mano.
- Si no la trae (como en esta primera prueba), pásala tú como tercer
  argumento: la hora que Unicorn Suite mostraba al darle "Record".

Qué produce:
1. <eeg>_con_markers.csv -- tu CSV de EEG crudo intacto + 2 columnas nuevas
   ("evento", "detalle"), vacías salvo en la fila más cercana a cada
   marcador. Funciona como un canal STIM sintético: cárgalo directo en tu
   pipeline de epoching (MNE, numpy, lo que uses).
2. Un resumen en consola: evento -> segundo estimado -> fila -> detalle,
   para que verifiques a ojo que el orden y el espaciado tienen sentido
   ANTES de confiar en el epoching.
3. Avisos explícitos para cualquier marcador que caiga FUERA del rango real
   de la grabación (antes del inicio o después del último sample) -- nunca
   los fuerza ni los inventa, los reporta como no alineables.
"""

import argparse
import sys
from datetime import datetime, timedelta

import pandas as pd

FS_HZ = 250.0  # frecuencia de muestreo fija del Unicorn Hybrid Black


def parse_time_today(time_str: str, ref_date) -> datetime:
    """Convierte 'HH:MM:SS' o 'HH:MM:SS.ffffff' en datetime completo,
    usando la fecha tomada del propio log de marcadores."""
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
        try:
            t = datetime.strptime(time_str.strip(), fmt).time()
            return datetime.combine(ref_date, t)
        except ValueError:
            continue
    raise ValueError(f"No pude interpretar la hora '{time_str}'. Usa HH:MM:SS o HH:MM:SS.ffffff")


def main():
    ap = argparse.ArgumentParser(description="Alinea EEG crudo de Unicorn con el log de marcadores de Sandhi.")
    ap.add_argument("eeg_csv", help="CSV exportado de Unicorn Suite Recorder")
    ap.add_argument("markers_csv", help="CSV generado por sandhi_markers_fallback.py")
    ap.add_argument("recording_start", nargs="?", default=None,
                     help="Hora real de 'Record' en Unicorn Suite (HH:MM:SS). Omitir si el log ya trae RECORDING_START_REFERENCE.")
    ap.add_argument("--fs", type=float, default=FS_HZ, help=f"Frecuencia de muestreo (default {FS_HZ} Hz, fija en el Hybrid Black)")
    ap.add_argument("--out", default=None, help="Nombre del CSV de salida")
    args = ap.parse_args()

    markers = pd.read_csv(args.markers_csv)
    markers["timestamp_iso"] = pd.to_datetime(markers["timestamp_iso"])
    ref_date = markers["timestamp_iso"].iloc[0].date()

    # 1. Resolver la hora de anclaje -----------------------------------
    ref_rows = markers[markers["evento"] == "RECORDING_START_REFERENCE"]
    if not ref_rows.empty:
        anchor_str = str(ref_rows.iloc[0]["detalle"])
        recording_start = parse_time_today(anchor_str, ref_date)
        print(f">> Ancla tomada del log (RECORDING_START_REFERENCE): {anchor_str}")
    elif args.recording_start:
        recording_start = parse_time_today(args.recording_start, ref_date)
        print(f">> Ancla tomada del argumento: {args.recording_start}")
    else:
        print("ERROR: el log no trae RECORDING_START_REFERENCE y no pasaste una hora de anclaje.")
        print("Vuelve a correr con: python sandhi_align_markers.py eeg.csv markers.csv HH:MM:SS")
        sys.exit(1)

    # 2. Cargar EEG crudo y calcular su rango real de tiempo -------------
    eeg = pd.read_csv(args.eeg_csv)
    n_samples = len(eeg)
    duration_s = (n_samples - 1) / args.fs
    recording_end = recording_start + timedelta(seconds=duration_s)
    print(f">> Grabación: {n_samples} muestras a {args.fs:.0f} Hz = {duration_s:.2f} s")
    print(f">> Rango estimado: {recording_start.time()} -> {recording_end.time()}\n")

    # 3. Mapear cada marcador a una fila de muestra -----------------------
    eeg["evento"] = ""
    eeg["detalle"] = ""

    print(f"{'evento':<28} {'seg.':>8} {'fila':>8}   detalle")
    print("-" * 70)
    unaligned = []
    for _, row in markers.iterrows():
        if row["evento"] == "RECORDING_START_REFERENCE":
            continue
        offset_s = (row["timestamp_iso"] - recording_start).total_seconds()
        idx = round(offset_s * args.fs)
        detalle = "" if pd.isna(row.get("detalle")) else str(row["detalle"])
        if 0 <= idx < n_samples:
            eeg.at[idx, "evento"] = row["evento"]
            eeg.at[idx, "detalle"] = detalle
            print(f"{row['evento']:<28} {offset_s:8.2f} {idx:8d}   {detalle}")
        else:
            unaligned.append((row["evento"], offset_s, detalle))
            print(f"{row['evento']:<28} {offset_s:8.2f} {'FUERA':>8}   {detalle}")

    out_path = args.out or args.eeg_csv.rsplit(".", 1)[0] + "_con_markers.csv"
    eeg.to_csv(out_path, index=False)
    print(f"\n>> Guardado: {out_path}")

    if unaligned:
        print(f"\n⚠ {len(unaligned)} marcador(es) cayeron fuera del rango de la grabación:")
        for ev, off, det in unaligned:
            when = "antes del inicio" if off < 0 else "después del final"
            print(f"   - {ev} ({when}, offset {off:.2f}s) {('— ' + det) if det else ''}")
        print("   Revisa si tu ancla (hora de 'Record') es correcta -- unos segundos de error")
        print("   son normales y esperados, pero minutos de diferencia indican que la")
        print("   referencia está mal (o que la grabación no cubre toda la sesión).")


if __name__ == "__main__":
    main()