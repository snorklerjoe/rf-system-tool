# rf-system-tool
A tool for designing RF systems at the block-diagram level that is simpler and lighterweight than Simulink or QUCS but focused on things like noise figure and spurious tones.

## Math model (cascade + spectrum)

- **Linear cascade / Bode response** is evaluated per block across frequency and cascaded as 2-port networks with `scikit-rf`, then read from cascaded `S21` (`rf_tool/plots/plot_windows.py`, `compute_frequency_sweep`; `rf_tool/engine/cascade.py`, `cascade_networks` / `s21_to_gain_db`).
- **Noise figure** uses Friis in linear units stage-by-stage (`rf_tool/engine/cascade.py`, `cascade_noise_figure` and `compute_cascade_metrics`).
- **IP3 / P1dB** are cascaded with worst-case phase-aligned voltage-domain addition, including cumulative linear gain terms (`rf_tool/engine/cascade.py`, `cascade_iip3`, `cascade_p1db`).
- **Min/Max level propagation** is input-referred stage-by-stage by subtracting cumulative upstream gain (`rf_tool/engine/cascade.py`, `compute_cascade_metrics`).
- **Mixer products** are generated from every RF-tone × LO-tone combination using mixer coefficients; output power depends on both RF and LO component powers (`rf_tool/blocks/components.py`, `Mixer.process`).
- **Noise floor propagation** is tracked explicitly with the signal model and merged in linear power; spectrum viewers use that computed floor as the plot floor (`rf_tool/models/signal.py`, `rf_tool/gui/canvas.py`, `rf_tool/plots/plot_windows.py`).
