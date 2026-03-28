# eno_love

Exploratory branch for EEG quality checks and format comparison in Neuroelectrics workflows.

## Objetivo

Este repo tiene dos usos principales:

1. Comparar exportaciones `.easy` vs `.edf` con métricas numéricas y reporte automático.
2. Generar gráficos y estadísticos por canal para inspección visual de calidad.

## Scripts

- `edf_conv.py`: comparación por pares y generación de `CSV + JSON + MD`.
- `easy_edf_graph.py`: comparación visual avanzada (`PNG + CSV + JSON + MD`) para un par específico.

## Setup rápido

```bash
python -m venv eno_venv
source eno_venv/bin/activate
pip install -r requirements.txt
```

## Flujo recomendado para futuros ensayos

### 1) Pre-chequeo de integridad del lote

Antes de correr análisis, valida que los archivos tengan contenido:

```bash
find data/test_XXX -type f \( -name '*.easy' -o -name '*.edf' -o -name '*.info' \) | sort
wc -l data/test_XXX/*.easy
```

Si un archivo aparece con `0` líneas o `0B`, debe excluirse o regenerarse.

### 2) Comparación automática por carpeta

Caso ideal (`.easy` y `.edf` con mismo stem):

```bash
./eno_venv/bin/python edf_conv.py \
	--data-dir data/test_XXX \
	--output-dir comparison_reports/test_XXX_single
```

Fallback implementado en este repo para lotes sin `.edf` válido:

- Si no encuentra pares `.easy/.edf`, el script entra en modo `.easy`-only.
- En ese modo compara cada `.easy` contra sí mismo para generar reporte estructural del lote.

Importante: en modo `.easy`-only la correlación media tenderá a `1.0` y el RMSE a `0`, lo cual no mide mejora real entre formatos; solo confirma consistencia interna del archivo.

### 3) Inspección visual de pares clave

Cuando sí exista `.edf` real, usa:

```bash
./eno_venv/bin/python easy_edf_graph.py \
	--easy "data/test_XXX/archivo.easy" \
	--edf "data/test_XXX/archivo.edf" \
	--info "data/test_XXX/archivo.info" \
	--output-dir comparison_reports/test_XXX_graphs
```

Esto genera overlays, histogramas y heatmap de métricas por canal.

## Cómo interpretar resultados de forma útil

### Señales de buena calidad del ensayo

- Duración y `sampling_rate_hz` estables entre sesiones.
- Sin canales con `std` o `peak-to-peak` anómalos extremos frente al resto.
- Tendencia consistente entre repeticiones (`001` vs `002`) en misma condición (EO, EC, MI).
- Menor dispersión en `mean_abs_diff_uv` y `rmse_uv` cuando compares contra `.edf` real.

### Sobre la "correlación media mejoró"

Ese indicador es valioso solamente cuando comparas dos fuentes distintas (`.easy` vs `.edf`).
Si corres en modo `.easy`-only, una correlación media alta no implica mejora de calidad fisiológica; para calidad del ensayo mira más los patrones por canal, estabilidad entre bloques y presencia de artefactos.

## Salidas esperadas

Por cada comparación, `edf_conv.py` crea:

- `*_channels.csv` (métricas por canal)
- `*.json` (resumen estructurado)
- `*.md` (interpretación textual)

Por cada comparación, `easy_edf_graph.py` crea además:

- `*_raw_overlay.png`
- `*_demeaned_overlay.png`
- `*_histograms.png`
- `*_stats_heatmap.png`

## Recomendaciones operativas

- Mantener nombres consistentes por sesión (timestamp + sujeto + condición + réplica).
- Guardar `.info` junto a cada `.easy` para preservar nombres de canal y metadatos.
- No usar archivos `*_edfFiltered.easy` vacíos; si existen con `0B`, ignorarlos o regenerarlos.
- Versionar siempre el comando usado y el `output-dir` del análisis para reproducibilidad.
