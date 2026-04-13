# Comparación NIC2: `20260309140635_the_princess_001_Rest Eyes Closed.easy` vs `20260309140635_the_princess_001_Rest Eyes Closed.edf`

## Resumen
- Canales comunes: 8
- Frecuencia de muestreo .easy: 500.000 Hz
- Frecuencia de muestreo .edf: 500.000 Hz
- Duración .easy: 60.001 s
- Duración .edf: 60.000 s
- Correlación media: 0.7758
- RMSE medio: 25926.943 uV
- Diferencia absoluta media: 25926.334 uV

## Qué aporta cada formato
- `.easy`: amplitudes tabulares por muestra, timestamps Unix, marcadores y a veces acelerometría. Es muy útil para auditoría del export, sincronización y trazabilidad de eventos.
- `.info`: orden real de electrodos, frecuencia de muestreo declarada y pistas del dispositivo/configuración.
- `.edf`: formato estándar, portable y listo para ecosistemas como MNE/EEGLAB; suele facilitar intercambio, anotación y archivado reproducible.

## Interpretación
- El .easy conserva marcadores por muestra, timestamps Unix y, cuando existe, acelerometría; eso es útil para alineación temporal fina y control de artefactos por movimiento.
- El .edf es el formato más portable para análisis y archivado, aunque en este archivo no aparecen anotaciones.
- La similitud promedio entre señales es modesta; conviene revisar referencia, escalado, filtros aplicados al exportar y correspondencia exacta de canales.

## Bandas de potencia promedio
- delta: .easy=8.710879e-12, .edf=7.580435e-05
- theta: .easy=3.681318e-12, .edf=8.023746e-05
- alpha: .easy=6.196700e-12, .edf=7.504732e-05
- beta: .easy=1.132092e-12, .edf=6.734620e-05
- gamma: .easy=6.354558e-13, .edf=4.265906e-05
