# Comparación NIC2: `20260309140505_the_princess_001_Rest Eyes Open.easy` vs `20260309140505_the_princess_001_Rest Eyes Open.edf`

## Resumen
- Canales comunes: 8
- Frecuencia de muestreo .easy: 500.000 Hz
- Frecuencia de muestreo .edf: 500.000 Hz
- Duración .easy: 60.000 s
- Duración .edf: 60.000 s
- Correlación media: 0.6442
- RMSE medio: 26548.168 uV
- Diferencia absoluta media: 26547.990 uV

## Qué aporta cada formato
- `.easy`: amplitudes tabulares por muestra, timestamps Unix, marcadores y a veces acelerometría. Es muy útil para auditoría del export, sincronización y trazabilidad de eventos.
- `.info`: orden real de electrodos, frecuencia de muestreo declarada y pistas del dispositivo/configuración.
- `.edf`: formato estándar, portable y listo para ecosistemas como MNE/EEGLAB; suele facilitar intercambio, anotación y archivado reproducible.

## Interpretación
- El .easy conserva marcadores por muestra, timestamps Unix y, cuando existe, acelerometría; eso es útil para alineación temporal fina y control de artefactos por movimiento.
- El .edf es el formato más portable para análisis y archivado, aunque en este archivo no aparecen anotaciones.
- La similitud promedio entre señales es modesta; conviene revisar referencia, escalado, filtros aplicados al exportar y correspondencia exacta de canales.

## Bandas de potencia promedio
- delta: .easy=1.761965e-11, .edf=7.720473e-05
- theta: .easy=4.005166e-12, .edf=8.360226e-05
- alpha: .easy=2.598932e-12, .edf=8.001492e-05
- beta: .easy=1.080807e-12, .edf=7.026950e-05
- gamma: .easy=6.760604e-13, .edf=4.996048e-05
