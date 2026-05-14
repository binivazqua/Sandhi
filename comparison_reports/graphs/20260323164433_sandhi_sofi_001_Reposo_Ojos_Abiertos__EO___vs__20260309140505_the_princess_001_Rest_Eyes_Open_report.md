# Visual and Statistical Comparison: `20260323164433_sandhi_sofi_001_Reposo_Ojos_Abiertos_(EO).easy` vs `20260309140505_the_princess_001_Rest Eyes Open.edf`

## Why prioritize `.easy` / `.nedf`
- Neuroelectrics documents that `.nedf` is 24-bit, `.edf` is 16-bit, and the DC component is filtered when `.edf` is created.
- For slow pre-impulse or slow cortical potential analysis, that makes `.easy` and especially `.nedf` the safer source of truth for raw amplitude and very-low-frequency content.

## Run summary
- Compared channels: Cz, C3
- Sampling rate used for comparison: 500.000 Hz
- Compared duration: 15.000 s
- Mean Pearson correlation, raw: -0.0097
- Mean Pearson correlation, demeaned: -0.0097
- Mean Spearman correlation: 0.0218
- Mean RMSE, raw: 38315.710 uV
- Mean RMSE, demeaned: 106.696 uV
- Mean variance ratio, easy/edf: 29.2367

## Interpretation
- Best shape agreement after demeaning: `Cz` with Pearson=0.0015.
- Worst shape agreement after demeaning: `C3` with Pearson=-0.0209.
- If raw RMSE is much larger than demeaned RMSE, the main mismatch is likely DC offset or reference rather than waveform shape.
- If slow-band power (`0-0.5 Hz`) is systematically smaller in `.edf`, that is consistent with the documented DC filtering in EDF export.
- Quantization-step estimates are heuristic, but consistently larger steps in `.edf` are directionally consistent with lower effective resolution.

## Sources
- NIC2 User Manual: https://www.neuroelectrics.com/api/downloads/NE_P3_UM004_EN_NIC2.1.0_1.pdf (NIC2 states `.nedf` is 24-bit, `.edf` is 16-bit, and the DC component is filtered in `.edf` exports.)
- Neuroelectrics EEGLAB Plugin: https://www.neuroelectrics.com/eeglab-plugin (Neuroelectrics exposes `.easy` and `.nedf` as native analysis-ready inputs for their tooling.)
- Enobio 8 Product Page: https://www.neuroelectrics.com/products/research/enobio/enobio8 (Neuroelectrics advertises 24-bit signal resolution, 0.05 uV resolution, and sample-precision data storage.)
