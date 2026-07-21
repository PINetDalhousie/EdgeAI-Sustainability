# Power Measurement Setup

Detailed companion to the **Power Measurement Setup** section of the
[main README](../README.md#power-measurement-setup). Because reproducibility is a core
contribution of the paper, the exact meter, wiring, and readout procedure are documented
here. Everything below is consistent with the measurement scheme described in the paper.

## Why a hardware power meter

No single software power tool works across all four heterogeneous devices (Raspberry Pi,
Intel NCS 2, Google Coral USB, Jetson Nano) and their vendor frameworks. We therefore use
an **external hardware USB power meter** that is device- and framework-agnostic: it sits on
the USB power line and measures the actual electrical draw regardless of platform, so
readings are directly comparable across all configurations.

## Hardware

- **Meter:** **FNB48S** USB-C Tester / USB Multimeter Power Meter (156 W) — an inline USB
  power meter that measures bus voltage and current and derives power, energy, and capacity.
- **Operating range / accuracy (as used):** 4–24 V, 0–6.5 A, **±1 % voltage / ±2 % current**.
- **Sampling rate:** **16 Hz** (16 time-stamped voltage/current/power readings per second).
- **Placement:** inserted **inline between the device's official power adapter and the
  device under test (DUT)** — meter input to the power supply, meter output to the device —
  so all current the device draws flows through the meter. For the USB accelerators (INCS,
  GCU) attached to the Raspberry Pi, the meter captures the combined RPi + accelerator draw
  on the RPi's supply.

## Software (measurement readout)

- **On-device offline recording:** the meter records voltage/current to its internal memory
  for the duration of a session (start/stop controllable by current thresholds).
- **PC application (Windows):** **Power-Z (KM001)** —
  <https://sourceforge.net/projects/power-z-km001/>. Connect the meter to the PC with a
  micro-USB cable, install the driver once, then use the app to read out a recorded session,
  view the V/I/P curve, and **export the time-stamped samples to a file (CSV)** for
  post-processing. Live PC-side ("online") recording is also supported.

## Measurement procedure

Stabilize → record → run → export → align → integrate, repeated **10×** per configuration.

1. **Stabilize the device.** Terminate background processes, keep the device under stable
   thermal conditions (active cooling on the Jetson Nano where needed), and insert a short
   sleep between successive runs so measurements are not taken while readings fluctuate.
2. **Start the recording** (offline on the meter, or online via the PC app). Set the current
   start/stop thresholds so the session captures the whole run (e.g. a `0 mA` start threshold
   records immediately; a `0 mA` stop threshold prevents premature auto-stop).
3. **Run the inference script** (`python3 <model>_inference.py`). It prints `Start`/`End`
   timestamps for each of the five phases — initialize/baseline, load dataset, load model,
   inference, memory + accuracy — via `time.strftime()`, while the meter samples power
   continuously at 16 Hz.
4. **Stop and export** the recorded samples (time-stamped V/I/P) to a CSV via the PC app.
5. **Baseline subtraction.** The first ~3 s of near-idle operation (the first **48 samples**
   at 16 Hz) give the baseline power; its mean is subtracted from the inference-phase
   readings to isolate compute-specific power.
6. **Align & integrate.** [`phase_identification.py`](../phase_identification.py) splits the
   exported log into the five phases using the script's per-phase timestamps;
   [`measurments.py`](../measurments.py) adds a timestamp column (from the record's start
   time and sampling interval) and sums energy over the inference window.
7. **Repeat 10×** and report the mean ± 95 % confidence interval.

## Post-processing scripts

| Script | Role |
|--------|------|
| [`phase_identification.py`](../phase_identification.py) | Groups the exported meter-log lines into the five phases using the timestamps the inference script printed. |
| [`measurments.py`](../measurments.py) | Adds timestamps to the meter CSV (start time + sampling rate) and sums energy within a given time range. |

<!-- ## Consistency with the paper

- Power is sampled at **16 Hz**; the baseline is the **first 48 samples over the first 3 s**,
  subtracted to isolate inference power — matching the paper's measurement scheme.
- The meter and its PC app capture only the raw electrical readings. The other three metrics
  come from the inference scripts: **F1-score/accuracy** from predictions vs. ground truth,
  **inference time** from `time.strftime()` start/finish timestamps, and **memory
  utilization** from `psutil`. -->
