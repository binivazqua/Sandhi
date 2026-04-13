# Comparación NIC2: `20260323165255_sandhi_sofi_002_Motor Imagery EO.easy` vs `20260323165255_sandhi_sofi_002_Motor Imagery EO.easy`

## Resumen
- Canales comunes: 8
- Frecuencia de muestreo .easy: 500.000 Hz
- Frecuencia de muestreo .edf: 500.000 Hz
- Duración .easy: 19.999 s
- Duración .edf: 20.000 s
- Correlación media: 1.0000
- RMSE medio: 0.000 uV
- Diferencia absoluta media: 0.000 uV

## Qué aporta cada formato
- `.easy`: amplitudes tabulares por muestra, timestamps Unix, marcadores y a veces acelerometría. Es muy útil para auditoría del export, sincronización y trazabilidad de eventos.
- `.info`: orden real de electrodos, frecuencia de muestreo declarada y pistas del dispositivo/configuración.
- `.edf`: formato estándar, portable y listo para ecosistemas como MNE/EEGLAB; suele facilitar intercambio, anotación y archivado reproducible.

## Interpretación
- El .easy conserva marcadores por muestra, timestamps Unix y, cuando existe, acelerometría; eso es útil para alineación temporal fina y control de artefactos por movimiento.
- El .edf es el formato más portable para análisis y archivado, aunque en este archivo no aparecen anotaciones.
- Las señales coinciden razonablemente bien; eso sugiere que .easy y .edf representan la misma adquisición con diferencias menores de formato o preprocesado.

## Bandas de potencia promedio
- delta: .easy=2.406055e-11, .edf=2.406055e-11
- theta: .easy=4.128227e-12, .edf=4.128227e-12
- alpha: .easy=1.049296e-12, .edf=1.049296e-12
- beta: .easy=7.107358e-13, .edf=7.107358e-13
- gamma: .easy=4.682498e-13, .edf=4.682498e-13
