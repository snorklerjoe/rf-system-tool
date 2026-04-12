# rf-system-tool
A tool for designing RF systems at the block-diagram level that is simpler and lighterweight than Simulink or QUCS but focused on things like noise figure and spurious tones.

## Motivation & Alternatives

I was looking for a computer-based approach to high-level RF system design.  
I need an easy way to calculate noise figure, IP3, etc. through a cascaded system, while visualizing things in a nice block diagram.

Pozar's book has a lot of great explanations of how to do all of that by hand, and I started doing that, but I really want fast ways to compare what happens to the overall system when trying different components with different specs.  
Nothing here is super precise-- it isn't ADS. This just automates the back-of-the-envelope, idealized calculations that you might find in the later chapters of Pozar's book.

_Potential alternatives:_ 

- HP AppCAD is the closest thing I could find to what I want, but you can't save, and it is meant for Windows 98. A fun thing to check out, though!
- Simulink is too bloated for my use here.
- Scilab XCos is lightweight but not very useful nor extendable.
- Octave lacks a block-diagram GUI means of designing a system.
- ADS costs too much.
- QUCS-S is an awesome free tool, but it is really more geared towards full circuit simulation.

## This was created, pretty much entirely, with AI.

As a disclaimer, this tool is almost entirely created with GitHub Copilot. If you're interested in the prompts I used, check out the closed pull requests.  
I do try to have more of a hand in most of the things I write, but this application is really a means to an end.

## Theory of Operation / Math model (cascade + spectrum)

- **Linear cascade / Bode response** is evaluated per block across frequency and cascaded as 2-port networks with `scikit-rf`, then read from cascaded `S21` (`rf_tool/plots/plot_windows.py`, `compute_frequency_sweep`; `rf_tool/engine/cascade.py`, `cascade_networks` / `s21_to_gain_db`).
- **Noise figure** uses Friis in linear units stage-by-stage (`rf_tool/engine/cascade.py`, `cascade_noise_figure` and `compute_cascade_metrics`).
- **IP3 / P1dB** are cascaded with worst-case phase-aligned voltage-domain addition, including cumulative linear gain terms (`rf_tool/engine/cascade.py`, `cascade_iip3`, `cascade_p1db`).
- **Min/Max level propagation** is input-referred stage-by-stage by subtracting cumulative upstream gain (`rf_tool/engine/cascade.py`, `compute_cascade_metrics`).
- **Mixer products** are generated from every RF-tone × LO-tone combination using mixer coefficients; output power depends on both RF and LO component powers (`rf_tool/blocks/components.py`, `Mixer.process`).
- **Noise floor propagation** is tracked explicitly with the signal model and merged in linear power; spectrum viewers use that computed floor as the plot floor (`rf_tool/models/signal.py`, `rf_tool/gui/canvas.py`, `rf_tool/plots/plot_windows.py`).
