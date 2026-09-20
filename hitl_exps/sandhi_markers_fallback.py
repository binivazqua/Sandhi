"""
Sandhi Lab — Marcadores de respaldo (SIN LSL) para el trial de doomscrolling
=============================================================================

Por qué existe esta versión:
La sincronización real vía LSL requiere compilar la app UnicornLSL de g.tec
(no viene lista de fábrica) y validar que puede correr al mismo tiempo que
Unicorn Suite Recorder sin pelear por la conexión Bluetooth del gorro — algo
que no da tiempo de resolver antes de tu primer participante de mañana.

Esta versión NO se conecta a LSL ni a tu EEG. Simplemente registra, en un
archivo de texto, la hora exacta (reloj de la computadora) de cada fase del
protocolo, cuando tú presionas la tecla correspondiente. Sigues grabando con
Unicorn Suite Recorder exactamente como ya sabes hacerlo (con tu semáforo
verde/rojo intacto) — este script corre en una ventana aparte, en paralelo,
solo para dejar constancia de "a qué hora pasó qué".

Cómo alinear después en análisis:
Tu grabación de Unicorn Suite tiene su propia hora de inicio (visible en el
nombre del archivo o metadata). Resta esa hora de inicio a cada timestamp de
este log para saber en qué segundo de la grabación EEG ocurrió cada fase.
Es menos preciso que LSL (margen de error de uno o dos segundos, por la
latencia de que tú presiones la tecla a tiempo), pero es suficiente para
epochear bloques de minutos como los tuyos.

Uso:
    python sandhi_markers_fallback.py
Genera un archivo `sandhi_markers_LOG_<fecha_hora>.csv` en la misma carpeta,
uno nuevo por sesión.
"""

import csv
import time
from datetime import datetime

MARKERS = {
    "1":  "SESSION_START",
    "2":  "HAIR_CHECK_START",
    "3":  "HAIR_CHECK_END",
    "4":  "CAP_PLACEMENT_START",
    "5":  "CAP_GREEN_STABLE",
    "6":  "ARTIFACT_TEST_START",
    "7":  "JAW_CLENCH",
    "8":  "BLINK",
    "9":  "ARTIFACT_TEST_END",
    "10": "BASELINE1_EC_START",
    "11": "BASELINE1_EC_END",
    "12": "BASELINE1_EO_START",
    "13": "BASELINE1_EO_END",
    "14": "READING1_START",
    "15": "READING1_END",
    "16": "SCROLL_FEED_START",
    "17": "SCROLL_FEED_END",
    "18": "SCROLL_REELS_START",
    "19": "SCROLL_REELS_END",
    "20": "BASELINE2_EC_START",
    "21": "BASELINE2_EC_END",
    "22": "BASELINE2_EO_START",
    "23": "BASELINE2_EO_END",
    "24": "READING2_START",
    "25": "READING2_END",
    "26": "POST_SURVEY_START",
    "27": "POST_SURVEY_END",
    "28": "TESTIMONIAL_START",
    "29": "TESTIMONIAL_END",
    "30": "SESSION_END",
    "r": "REDIRECT_REELS",
    "n": "NOTE",
}


def print_menu():
    print("\nSandhi Lab — Marcadores de respaldo (log local, sin LSL)\n")
    for key, label in MARKERS.items():
        print(f"  [{key:>2}]  {label}")
    print("\n  [q]  guardar y salir")
    print("  [?]  mostrar esta lista de nuevo\n")


def ask_subject_id():
    """Pide el ID del sujeto (ej. S10) y lo deja limpio para usarlo en el nombre del archivo."""
    while True:
        raw = input("ID del sujeto (ej. S10): ").strip().upper()
        if not raw:
            print("   (no puede quedar vacío, intenta de nuevo)")
            continue
        # solo letras, números y guiones -- evita romper el nombre de archivo
        safe = "".join(c for c in raw if c.isalnum() or c in ("-", "_"))
        if safe != raw:
            print(f"   (se limpiaron caracteres no permitidos: quedó como '{safe}')")
        if not safe:
            print("   (después de limpiar quedó vacío, intenta de nuevo)")
            continue
        return safe


def main():
    subject_id = ask_subject_id()
    filename = f"{subject_id}_sandhi_markers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_iso", "hora_legible", "evento", "detalle"])

        print_menu()
        print(f">> Sujeto: {subject_id}")
        print(f">> Log guardándose en: {filename}")
        print(">> IMPORTANTE: anota también la hora de inicio de tu grabación en Unicorn Suite Recorder.\n")

        while True:
            key = input(">> Marcador: ").strip()

            if key == "q":
                print(f"Log guardado en {filename}. Cerrando.")
                break

            if key == "?":
                print_menu()
                continue

            if key not in MARKERS:
                print(f"   (tecla '{key}' no reconocida, escribe '?' para ver la lista)")
                continue

            label = MARKERS[key]

            # NOTE y REDIRECT_REELS piden un detalle en texto libre, concatenado
            # como columna aparte -- así queda registrado qué pasó, no solo que pasó algo.
            detalle = ""
            if key in ("n", "r"):
                detalle = input("   Detalle (Enter para dejar vacío): ").strip()

            now = datetime.now()
            writer.writerow([now.isoformat(), now.strftime("%H:%M:%S"), label, detalle])
            f.flush()  # escribe a disco de inmediato, por si el script se cierra mal
            suffix = f" — {detalle}" if detalle else ""
            print(f"   [{now.strftime('%H:%M:%S')}] -> {label}{suffix}")


if __name__ == "__main__":
    main()