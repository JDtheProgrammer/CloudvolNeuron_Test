"""
NonIsopotentialCable.py
=======================

Purpose
-------
Numerically explore the predictions of *passive cable theory* for a finite,
non-isopotential cylindrical cable. The script performs four parameter
sweeps -- on stimulus amplitude (I), cable diameter (d), specific membrane
resistance (Rm) and axial resistivity (Ra) -- and, for each sweep, compares
the simulated steady-state voltage deflection at the injection site (centre
of the cable) and at the distal end against analytical predictions for an
infinite cable. Each section produces a 2 x 3 matplotlib figure summarising
voltage traces, the swept parameter and linearised theory/regression
overlays.

The relevant analytical relations (for an infinite passive cable) are:
    space constant      lambda = sqrt(Rm * d / (4 * Ra))
    input resistance    Rin    = sqrt(Rm * Ra / (pi * d))     (at injection site)
Both scale predictably with d, Rm and Ra, so log-log or appropriately
transformed plots should yield straight lines.

Inputs
------
None as files. Hard-coded baseline parameters (modified per sweep):
    L  = 1000 um (one millimetre long cable, sufficiently long compared
         to lambda that finite-length corrections are modest)
    d  = 1 um  (varied to 2, 4, 8 um in section 2)
    cm = 1 uF/cm^2 (specific membrane capacitance, canonical value)
    Rm = 1000 Ohm*cm^2 (varied to 500, 1000, 2000, 4000 in section 3)
    Ra = 100 Ohm*cm (varied to 50, 100, 200, 400 in section 4)
    Stimulus: IClamp at cable(0.5), delay = 0 ms, dur = 15 ms,
              amp = 0.1 nA baseline (swept 0.075..0.300 nA in section 1)
    Integration: finitialize(v_rest), continuerun(25 ms),
                 with v_rest = -70 mV.

Outputs
-------
Console: NEURON version banner, default parameter echo, and for every sweep
         the theoretical (lambda, Rin) plus the simulated deflections
         (delta_V_center, delta_V_end) and the fitted regression slopes.
Figures: four matplotlib windows (one per sweep), each 2 x 3:
         column 1 -- voltage traces at centre (top) and end (bottom)
         column 2 -- swept parameter visualisation + lambda vs sqrt(param)
         column 3 -- delta-V vs the appropriate transformed parameter,
                     with simulated points, theoretical curve and the
                     least-squares linear regression.

How the script works
--------------------
1. Build a single passive cable section.
2. Define small helpers ``calculate_lambda`` and ``calculate_Rin_cable``
   that implement the analytical formulae with consistent CGS-style units
   (cm internally, um at the interfaces).
3. Sweep amplitude (Section 1), diameter (Section 2), Rm (Section 3) and
   Ra (Section 4). For each parameter setting:
       reset the cable property, record v at cable(0.5) and cable(1.0),
       finitialize -> continuerun, store peak delta-V relative to rest,
       and update the corresponding subplot.
4. After each sweep call ``plt.show(block=False)`` so figures stack; the
   last sweep uses a blocking ``plt.show()`` to keep windows open.

Required modules
----------------
    neuron (with both ``h`` and ``n`` interfaces and ``neuron.units``),
    numpy, scipy (only ``scipy.optimize.curve_fit`` is imported -- left
        available for future nonlinear fits even though not used here),
    matplotlib, pandas, plotnine,
    Python stdlib: csv, math, pprint, textwrap.

Author / Project: CloudvolNeuron_Test
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import csv
import math
import pprint
import textwrap
from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
from scipy.optimize import curve_fit  # reserved for nonlinear fits

import neuron
from neuron import h
from neuron import n
from neuron.units import ms, mV, um

print("\n NEURON Version:\n", neuron.__version__)
print("\n" + "=" * 80)
print("NON-ISOPOTENTIAL CABLE THEORY ANALYSIS")
print("=" * 80)

# -----------------------------------------------------------------------------
# Build the baseline cable
# -----------------------------------------------------------------------------
cable = h.Section(name="cable")
h.load_file("stdrun.hoc")  # required for finitialize / continuerun

# Stimulus -- always at the geometrical centre of the cable.
iclamp = h.IClamp(cable(0.5))
iclamp.delay = 0      # ms
iclamp.dur   = 15     # ms  - long enough that voltage approximates steady state
iclamp.amp   = 0.1    # nA  - baseline current (overwritten in sweeps)

# Passive properties: insert "pas" mechanism on every segment.
cable.insert("pas")

# Baseline parameters (modified per sweep below).
v_rest      = -70 * mV    # mV   - resting membrane potential
cable.L     = 1000 * um   # um   - 1 mm long cable
cable.diam  = 1 * um      # um
cable.cm    = 1.0         # uF/cm^2
cable(0.5).pas.g = 0.001  # S/cm^2   ->  Rm = 1 / g = 1000 Ohm*cm^2
cable(0.5).pas.e = v_rest # mV
cable.Ra    = 100         # Ohm*cm

print("\nDefault Cable Parameters:")
print(f"  Length L = {cable.L} um")
print(f"  Diameter d = {cable.diam} um")
print(f"  Membrane capacitance Cm = {cable.cm} uF/cm^2")
print(f"  Membrane conductance g = {cable(0.5).pas.g} S/cm^2 "
      f"(Rm = {1 / cable(0.5).pas.g} Ohm*cm^2)")
print(f"  Axial resistivity Ra = {cable.Ra} Ohm*cm")


# -----------------------------------------------------------------------------
# Analytical helpers
# -----------------------------------------------------------------------------
def calculate_lambda(diam_um, Rm, Ra):
    """Return the passive cable space constant lambda in micrometres.

    Formula (infinite cylinder):  lambda = sqrt(Rm * d / (4 * Ra))
    Units inside the function are CGS-like (cm); the conversion factor
    1e-4 converts um -> cm on input and 1e4 converts cm -> um on output.

    Parameters
    ----------
    diam_um : float    diameter in micrometres
    Rm      : float    specific membrane resistance in Ohm*cm^2
    Ra      : float    axial resistivity in Ohm*cm

    Returns
    -------
    float : space constant in micrometres.
    """
    d_cm = diam_um * 1e-4
    lambda_cm = np.sqrt(Rm * d_cm / (4 * Ra))
    return lambda_cm * 1e4


def calculate_Rin_cable(diam_um, Rm, Ra, L=None):
    """Return the input resistance at the injection point for a semi-infinite
    passive cable.

    Formula:  Rin = sqrt(Rm * Ra / (pi * d))
    For a finite cable of length L the expression has additional tanh()
    corrections; for L >> lambda it converges to the value computed here.

    Parameters
    ----------
    diam_um : float    diameter in micrometres
    Rm      : float    specific membrane resistance in Ohm*cm^2
    Ra      : float    axial resistivity in Ohm*cm
    L       : float, optional   ignored (kept for future extension)

    Returns
    -------
    float : input resistance in Ohms.
    """
    d_cm = diam_um * 1e-4
    return np.sqrt(Rm * Ra / (np.pi * d_cm))


# =============================================================================
# SECTION 1: VARYING CURRENT PULSE AMPLITUDE
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 1: VARYING CURRENT PULSE AMPLITUDE")
print("=" * 80)

# Four amplitudes spanning a roughly 4x range; small enough to stay in the
# linear (passive) regime so delta_V should scale linearly with I.
amps = [0.075 * k for k in range(1, 5)]   # nA: [0.075, 0.150, 0.225, 0.300]
colors = ["green", "blue", "red", "black"]

# Theoretical predictions for the default cable.
Rm = 1 / cable(0.5).pas.g
Ra = cable.Ra
diam = cable.diam

lambda_val          = calculate_lambda(diam, Rm, Ra)
Rin_center          = calculate_Rin_cable(diam, Rm, Ra)
# 1 Ohm * 1 nA = 1e-6 mV, so convert Rin (Ohm) to mV/nA by * 1e-6.
theoretical_slope   = Rin_center * 1e-6

print("\nTheoretical Predictions:")
print(f"  Space constant lambda = {lambda_val:.2f} um")
print(f"  Input resistance at center Rin = {Rin_center:.2e} Ohm")
print(f"  Expected slope (deltaV vs I) = {theoretical_slope:.2e} mV/nA")

delta_v_center = []
delta_v_end    = []

fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
ax_volt_center      = axes[0, 0]
ax_volt_end         = axes[1, 0]
ax_pulse            = axes[0, 1]
ax_relation_center  = axes[0, 2]
ax_relation_end     = axes[1, 2]
axes[1, 1].axis("off")

for amp, color in zip(amps, colors):
    iclamp.amp = amp

    # Record at injection point (centre) and far end.
    v_center = h.Vector().record(cable(0.5)._ref_v)
    v_end    = h.Vector().record(cable(1.0)._ref_v)
    t        = h.Vector().record(h._ref_t)

    h.finitialize(v_rest * mV)
    h.continuerun(25 * ms)

    t_list        = list(t)
    v_center_list = list(v_center)
    v_end_list    = list(v_end)

    ax_volt_center.plot(t_list, v_center_list, color=color, linewidth=2,
                        label=f"I={amp:.3f} nA")
    ax_volt_end.plot(t_list, v_end_list, color=color, linewidth=2,
                     label=f"I={amp:.3f} nA")

    # Peak deflection above rest -- in this passive cable the trace is
    # monotonic during the pulse, so max() is the steady-state value.
    dv_center = max(v_center_list) - v_rest
    dv_end    = max(v_end_list)    - v_rest
    delta_v_center.append(dv_center)
    delta_v_end.append(dv_end)

    print(f"I = {amp:.3f} nA -> dV_center = {dv_center:.3f} mV, "
          f"dV_end = {dv_end:.3f} mV")

ax_volt_center.set(xlabel="Time (ms)", ylabel="Voltage (mV)",
                   title="Voltage at Center (injection site)")
ax_volt_center.legend(); ax_volt_center.grid(True)
ax_volt_end.set(xlabel="Time (ms)", ylabel="Voltage (mV)",
                title="Voltage at End (cable(1.0))")
ax_volt_end.legend(); ax_volt_end.grid(True)

# Visualise the input current pulses.
t_array = np.linspace(0, 25, 500)
for amp, color in zip(amps, colors):
    pulse = np.where((t_array >= iclamp.delay) &
                     (t_array < iclamp.delay + iclamp.dur), amp, 0)
    ax_pulse.plot(t_array, pulse, color=color, linewidth=3, label=f"{amp:.2f} nA")
ax_pulse.set(xlabel="Time (ms)", ylabel="Current (nA)",
             title="Input Current Pulses")
ax_pulse.legend(); ax_pulse.grid(True)

# Linear regression delta-V vs I at the centre.
slope_center, intercept_center = np.polyfit(amps, delta_v_center, 1)
r2_center = np.corrcoef(amps, delta_v_center)[0, 1] ** 2

amps_theory = np.linspace(0, max(amps) * 1.1, 20)
dv_theory   = amps_theory * theoretical_slope

ax_relation_center.plot(amps, delta_v_center, "o", color="blue", markersize=10,
                        label="Simulated")
ax_relation_center.plot(amps_theory, dv_theory, "x", color="red", markersize=8,
                        linewidth=2,
                        label=f"Theory: slope={theoretical_slope:.2e} mV/nA")
x_fit = np.linspace(0, max(amps) * 1.1, 100)
y_fit = slope_center * x_fit + intercept_center
ax_relation_center.plot(x_fit, y_fit, "--", color="black", linewidth=2,
                        label=f"Regression: slope={slope_center:.2e}")
ax_relation_center.set(xlabel="Current (nA)", ylabel="Peak dV (mV)",
                       title="dV vs Current (Center)")
ax_relation_center.legend(fontsize=9); ax_relation_center.grid(True)
ax_relation_center.text(0.05, 0.95, f"R^2 = {r2_center:.4f}",
                        transform=ax_relation_center.transAxes,
                        fontsize=10, va="top")

# Linear regression delta-V vs I at the far end.
slope_end, intercept_end = np.polyfit(amps, delta_v_end, 1)
r2_end = np.corrcoef(amps, delta_v_end)[0, 1] ** 2

ax_relation_end.plot(amps, delta_v_end, "o", color="purple", markersize=10,
                     label="Simulated (end)")
ax_relation_end.plot(amps_theory, dv_theory * 0.5, "x", color="orange",
                     markersize=8, linewidth=2, label="Theory (approx)")
y_fit_end = slope_end * x_fit + intercept_end
ax_relation_end.plot(x_fit, y_fit_end, "--", color="black", linewidth=2,
                     label=f"Regression: slope={slope_end:.2e}")
ax_relation_end.set(xlabel="Current (nA)", ylabel="Peak dV (mV)",
                    title="dV vs Current (End)")
ax_relation_end.legend(fontsize=9); ax_relation_end.grid(True)
ax_relation_end.text(0.05, 0.95, f"R^2 = {r2_end:.4f}",
                     transform=ax_relation_end.transAxes,
                     fontsize=10, va="top")

print("\nRegression Results:")
print(f"  Center: slope = {slope_center:.2e} mV/nA "
      f"(theory = {theoretical_slope:.2e})")
print(f"  End:    slope = {slope_end:.2e} mV/nA")
print(f"  Ratio (end/center) = {slope_end / slope_center:.3f}")

plt.tight_layout()
plt.show(block=False)


# =============================================================================
# SECTION 2: VARYING CABLE DIAMETER
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 2: VARYING CABLE DIAMETER")
print("=" * 80)

# Diameters chosen so that sqrt(d) and 1/sqrt(d) span ~3x, which is enough
# to clearly resolve the theoretical lambda ~ sqrt(d), Rin ~ 1/sqrt(d) scalings.
diameters = [1, 2, 4, 8]    # um
colors = ["green", "blue", "red", "black"]
amp = 0.1                   # nA - fixed across the sweep
iclamp.amp = amp

Rm = 1 / cable(0.5).pas.g   # restore baseline Rm
Ra = cable.Ra               # baseline Ra

delta_v_center_diam = []
delta_v_end_diam    = []
lambda_values       = []
Rin_theory_values   = []

fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
ax_volt_center  = axes[0, 0]
ax_volt_end     = axes[1, 0]
ax_lambda       = axes[0, 1]
ax_dv_center    = axes[0, 2]
ax_dv_end       = axes[1, 2]
ax_diam_bar     = axes[1, 1]

for diam, color in zip(diameters, colors):
    cable.diam = diam

    lambda_val = calculate_lambda(diam, Rm, Ra)
    Rin_center = calculate_Rin_cable(diam, Rm, Ra)
    lambda_values.append(lambda_val)
    Rin_theory_values.append(Rin_center)

    v_center = h.Vector().record(cable(0.5)._ref_v)
    v_end    = h.Vector().record(cable(1.0)._ref_v)
    t        = h.Vector().record(h._ref_t)

    h.finitialize(v_rest * mV)
    h.continuerun(25 * ms)

    t_list        = list(t)
    v_center_list = list(v_center)
    v_end_list    = list(v_end)

    ax_volt_center.plot(t_list, v_center_list, color=color, linewidth=2,
                        label=f"d={diam} um")
    ax_volt_end.plot(t_list, v_end_list, color=color, linewidth=2,
                     label=f"d={diam} um")

    dv_center = max(v_center_list) - v_rest
    dv_end    = max(v_end_list)    - v_rest
    delta_v_center_diam.append(dv_center)
    delta_v_end_diam.append(dv_end)

    print(f"d = {diam} um -> lambda = {lambda_val:.1f} um, "
          f"Rin = {Rin_center:.2e} Ohm")
    print(f"           dV_center = {dv_center:.2f} mV, dV_end = {dv_end:.2f} mV")

ax_volt_center.set(xlabel="Time (ms)", ylabel="Voltage (mV)",
                   title="Voltage at Center")
ax_volt_center.legend(); ax_volt_center.grid(True)
ax_volt_end.set(xlabel="Time (ms)", ylabel="Voltage (mV)",
                title="Voltage at End")
ax_volt_end.legend(); ax_volt_end.grid(True)

ax_diam_bar.bar(range(len(diameters)), diameters, color=colors,
                tick_label=[f"{d}" for d in diameters])
ax_diam_bar.set(xlabel="Condition", ylabel="Diameter (um)",
                title="Tested Diameters")
ax_diam_bar.grid(True, axis="y")

# lambda vs sqrt(d): expected straight line through origin.
sqrt_d        = np.sqrt(diameters)
sqrt_d_theory = np.linspace(0, max(sqrt_d) * 1.1, 20)
lambda_theory = np.sqrt(Rm / (4 * Ra) * 1e-4) * 1e4 * sqrt_d_theory
slope_lambda, _ = np.polyfit(sqrt_d, lambda_values, 1)
x_fit = np.linspace(0, max(sqrt_d) * 1.1, 100)
y_fit = slope_lambda * x_fit

ax_lambda.plot(sqrt_d, lambda_values, "o", color="green", markersize=10,
               label="Simulated lambda")
ax_lambda.plot(sqrt_d_theory, lambda_theory, "x", color="red", markersize=8,
               linewidth=2, label="Theory")
ax_lambda.plot(x_fit, y_fit, "--", color="black", linewidth=2,
               label=f"Regression: slope={slope_lambda:.1f}")
ax_lambda.set(xlabel="sqrt(Diameter) (sqrt(um))",
              ylabel="Space Constant lambda (um)",
              title="Space Constant vs sqrt(Diameter)")
ax_lambda.legend(); ax_lambda.grid(True)

# delta-V vs 1/sqrt(d): expected straight line through origin.
inv_sqrt_d        = 1 / np.sqrt(diameters)
inv_sqrt_d_theory = np.linspace(0, max(inv_sqrt_d) * 1.1, 20)
dV_theory_center  = amp * np.sqrt(Rm * Ra / (np.pi * 1e-4)) * 1e-6 * inv_sqrt_d_theory
slope_dv_center, _ = np.polyfit(inv_sqrt_d, delta_v_center_diam, 1)
y_fit_center = slope_dv_center * np.linspace(0, max(inv_sqrt_d) * 1.1, 100)

ax_dv_center.plot(inv_sqrt_d, delta_v_center_diam, "o", color="blue",
                  markersize=10, label="Simulated")
ax_dv_center.plot(inv_sqrt_d_theory, dV_theory_center, "x", color="red",
                  markersize=8, linewidth=2, label="Theory")
ax_dv_center.plot(np.linspace(0, max(inv_sqrt_d) * 1.1, 100), y_fit_center,
                  "--", color="black", linewidth=2,
                  label=f"Regression: m={slope_dv_center:.2e}")
ax_dv_center.set(xlabel="1/sqrt(Diameter) (1/sqrt(um))",
                 ylabel="Peak dV (mV)", title="dV vs 1/sqrt(d) (Center)")
ax_dv_center.legend(fontsize=9); ax_dv_center.grid(True)

slope_dv_end, _ = np.polyfit(inv_sqrt_d, delta_v_end_diam, 1)
y_fit_end = slope_dv_end * np.linspace(0, max(inv_sqrt_d) * 1.1, 100)
ax_dv_end.plot(inv_sqrt_d, delta_v_end_diam, "o", color="purple",
               markersize=10, label="Simulated (end)")
ax_dv_end.plot(np.linspace(0, max(inv_sqrt_d) * 1.1, 100), y_fit_end, "--",
               color="black", linewidth=2,
               label=f"Regression: m={slope_dv_end:.2e}")
ax_dv_end.set(xlabel="1/sqrt(Diameter) (1/sqrt(um))",
              ylabel="Peak dV (mV)", title="dV vs 1/sqrt(d) (End)")
ax_dv_end.legend(fontsize=9); ax_dv_end.grid(True)

plt.tight_layout()
plt.show(block=False)


# =============================================================================
# SECTION 3: VARYING MEMBRANE RESISTIVITY (Rm)
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 3: VARYING MEMBRANE RESISTIVITY")
print("=" * 80)

# Rm spans 8x, so lambda ~ sqrt(Rm) should span ~2.8x.
Rm_values = [500, 1000, 2000, 4000]   # Ohm*cm^2
colors    = ["green", "blue", "red", "black"]

cable.diam = 2          # um  - reset diameter to a mid-range value
amp        = 0.1        # nA  - reset stimulus
iclamp.amp = amp

delta_v_center_Rm = []
delta_v_end_Rm    = []
lambda_values_Rm  = []

fig3, axes = plt.subplots(2, 3, figsize=(18, 10))
ax_volt_center   = axes[0, 0]
ax_volt_end      = axes[1, 0]
ax_Rm_bar        = axes[1, 1]
ax_lambda_Rm     = axes[0, 1]
ax_dv_center_Rm  = axes[0, 2]
ax_dv_end_Rm     = axes[1, 2]

for Rm, color in zip(Rm_values, colors):
    cable(0.5).pas.g = 1.0 / Rm        # set conductance from Rm
    lambda_val = calculate_lambda(cable.diam, Rm, cable.Ra)
    lambda_values_Rm.append(lambda_val)

    v_center = h.Vector().record(cable(0.5)._ref_v)
    v_end    = h.Vector().record(cable(1.0)._ref_v)
    t        = h.Vector().record(h._ref_t)

    h.finitialize(v_rest * mV)
    h.continuerun(25 * ms)

    t_list        = list(t)
    v_center_list = list(v_center)
    v_end_list    = list(v_end)

    ax_volt_center.plot(t_list, v_center_list, color=color, linewidth=2,
                        label=f"Rm={Rm} Ohm*cm^2")
    ax_volt_end.plot(t_list, v_end_list, color=color, linewidth=2,
                     label=f"Rm={Rm} Ohm*cm^2")

    dv_center = max(v_center_list) - v_rest
    dv_end    = max(v_end_list)    - v_rest
    delta_v_center_Rm.append(dv_center)
    delta_v_end_Rm.append(dv_end)

    print(f"Rm = {Rm} Ohm*cm^2 -> lambda = {lambda_val:.1f} um")
    print(f"              dV_center = {dv_center:.2f} mV, dV_end = {dv_end:.2f} mV")

ax_volt_center.set(xlabel="Time (ms)", ylabel="Voltage (mV)",
                   title="Voltage at Center")
ax_volt_center.legend(); ax_volt_center.grid(True)
ax_volt_end.set(xlabel="Time (ms)", ylabel="Voltage (mV)",
                title="Voltage at End")
ax_volt_end.legend(); ax_volt_end.grid(True)

ax_Rm_bar.bar(range(len(Rm_values)), Rm_values, color=colors,
              tick_label=[f"{Rm}" for Rm in Rm_values])
ax_Rm_bar.set(xlabel="Condition", ylabel="Rm (Ohm*cm^2)",
              title="Tested Rm Values")
ax_Rm_bar.grid(True, axis="y")

# lambda vs sqrt(Rm): expected straight line through origin.
sqrt_Rm        = np.sqrt(Rm_values)
sqrt_Rm_theory = np.linspace(0, max(sqrt_Rm) * 1.1, 20)
lambda_theory_Rm = np.sqrt(cable.diam * 1e-4 / (4 * cable.Ra)) * 1e4 * sqrt_Rm_theory
slope_lambda_Rm, _ = np.polyfit(sqrt_Rm, lambda_values_Rm, 1)
y_fit = slope_lambda_Rm * np.linspace(0, max(sqrt_Rm) * 1.1, 100)

ax_lambda_Rm.plot(sqrt_Rm, lambda_values_Rm, "o", color="green",
                  markersize=10, label="Simulated lambda")
ax_lambda_Rm.plot(sqrt_Rm_theory, lambda_theory_Rm, "x", color="red",
                  markersize=8, linewidth=2, label="Theory")
ax_lambda_Rm.plot(np.linspace(0, max(sqrt_Rm) * 1.1, 100), y_fit, "--",
                  color="black", linewidth=2,
                  label=f"Regression: m={slope_lambda_Rm:.2f}")
ax_lambda_Rm.set(xlabel="sqrt(Rm) (sqrt(Ohm*cm^2))",
                 ylabel="lambda (um)", title="Space Constant vs sqrt(Rm)")
ax_lambda_Rm.legend(); ax_lambda_Rm.grid(True)

# delta-V vs sqrt(Rm): proportional to Rin ~ sqrt(Rm).
slope_dv_Rm, _ = np.polyfit(sqrt_Rm, delta_v_center_Rm, 1)
y_fit_center = slope_dv_Rm * np.linspace(0, max(sqrt_Rm) * 1.1, 100)

ax_dv_center_Rm.plot(sqrt_Rm, delta_v_center_Rm, "o", color="blue",
                     markersize=10, label="Simulated")
ax_dv_center_Rm.plot(np.linspace(0, max(sqrt_Rm) * 1.1, 100), y_fit_center,
                     "--", color="black", linewidth=2,
                     label=f"Regression: m={slope_dv_Rm:.2e}")
ax_dv_center_Rm.set(xlabel="sqrt(Rm) (sqrt(Ohm*cm^2))",
                    ylabel="Peak dV (mV)", title="dV vs sqrt(Rm) (Center)")
ax_dv_center_Rm.legend(); ax_dv_center_Rm.grid(True)

slope_dv_end_Rm, _ = np.polyfit(sqrt_Rm, delta_v_end_Rm, 1)
y_fit_end = slope_dv_end_Rm * np.linspace(0, max(sqrt_Rm) * 1.1, 100)
ax_dv_end_Rm.plot(sqrt_Rm, delta_v_end_Rm, "o", color="purple",
                  markersize=10, label="Simulated (end)")
ax_dv_end_Rm.plot(np.linspace(0, max(sqrt_Rm) * 1.1, 100), y_fit_end, "--",
                  color="black", linewidth=2,
                  label=f"Regression: m={slope_dv_end_Rm:.2e}")
ax_dv_end_Rm.set(xlabel="sqrt(Rm) (sqrt(Ohm*cm^2))",
                 ylabel="Peak dV (mV)", title="dV vs sqrt(Rm) (End)")
ax_dv_end_Rm.legend(); ax_dv_end_Rm.grid(True)

plt.tight_layout()
plt.show(block=False)


# =============================================================================
# SECTION 4: VARYING AXIAL RESISTIVITY (Ra)
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 4: VARYING AXIAL RESISTIVITY (Ra)")
print("=" * 80)

# Ra spans 8x; lambda ~ 1/sqrt(Ra) so lambda should drop ~2.8x across the range.
Ra_values = [50, 100, 200, 400]   # Ohm*cm
colors    = ["green", "blue", "red", "black"]

cable.diam       = 2            # um  - reset diameter
cable(0.5).pas.g = 0.001        # S/cm^2  -> Rm = 1000
Rm                = 1000
amp               = 0.1
iclamp.amp        = amp

delta_v_center_Ra = []
delta_v_end_Ra    = []
lambda_values_Ra  = []

fig4, axes = plt.subplots(2, 3, figsize=(18, 10))
ax_volt_center   = axes[0, 0]
ax_volt_end      = axes[1, 0]
ax_Ra_bar        = axes[1, 1]
ax_lambda_Ra     = axes[0, 1]
ax_dv_center_Ra  = axes[0, 2]
ax_dv_end_Ra     = axes[1, 2]

for Ra, color in zip(Ra_values, colors):
    cable.Ra = Ra
    lambda_val = calculate_lambda(cable.diam, Rm, Ra)
    lambda_values_Ra.append(lambda_val)

    v_center = h.Vector().record(cable(0.5)._ref_v)
    v_end    = h.Vector().record(cable(1.0)._ref_v)
    t        = h.Vector().record(h._ref_t)

    h.finitialize(v_rest * mV)
    h.continuerun(25 * ms)

    t_list        = list(t)
    v_center_list = list(v_center)
    v_end_list    = list(v_end)

    ax_volt_center.plot(t_list, v_center_list, color=color, linewidth=2,
                        label=f"Ra={Ra} Ohm*cm")
    ax_volt_end.plot(t_list, v_end_list, color=color, linewidth=2,
                     label=f"Ra={Ra} Ohm*cm")

    dv_center = max(v_center_list) - v_rest
    dv_end    = max(v_end_list)    - v_rest
    delta_v_center_Ra.append(dv_center)
    delta_v_end_Ra.append(dv_end)

    print(f"Ra = {Ra} Ohm*cm -> lambda = {lambda_val:.1f} um")
    print(f"             dV_center = {dv_center:.2f} mV, dV_end = {dv_end:.2f} mV")

ax_volt_center.set(xlabel="Time (ms)", ylabel="Voltage (mV)",
                   title="Voltage at Center")
ax_volt_center.legend(); ax_volt_center.grid(True)
ax_volt_end.set(xlabel="Time (ms)", ylabel="Voltage (mV)",
                title="Voltage at End")
ax_volt_end.legend(); ax_volt_end.grid(True)

ax_Ra_bar.bar(range(len(Ra_values)), Ra_values, color=colors,
              tick_label=[f"{Ra}" for Ra in Ra_values])
ax_Ra_bar.set(xlabel="Condition", ylabel="Ra (Ohm*cm)",
              title="Tested Ra Values")
ax_Ra_bar.grid(True, axis="y")

# lambda vs 1/sqrt(Ra): expected straight line through origin.
inv_sqrt_Ra        = 1 / np.sqrt(Ra_values)
inv_sqrt_Ra_theory = np.linspace(0, max(inv_sqrt_Ra) * 1.1, 20)
lambda_theory_Ra   = (np.sqrt(Rm * cable.diam * 1e-4 / 4) * 1e4) * inv_sqrt_Ra_theory
slope_lambda_Ra, _ = np.polyfit(inv_sqrt_Ra, lambda_values_Ra, 1)
y_fit              = slope_lambda_Ra * np.linspace(0, max(inv_sqrt_Ra) * 1.1, 100)

ax_lambda_Ra.plot(inv_sqrt_Ra, lambda_values_Ra, "o", color="green",
                  markersize=10, label="Simulated lambda")
ax_lambda_Ra.plot(inv_sqrt_Ra_theory, lambda_theory_Ra, "x", color="red",
                  markersize=8, linewidth=2, label="Theory")
ax_lambda_Ra.plot(np.linspace(0, max(inv_sqrt_Ra) * 1.1, 100), y_fit, "--",
                  color="black", linewidth=2,
                  label=f"Regression: m={slope_lambda_Ra:.1f}")
ax_lambda_Ra.set(xlabel="1/sqrt(Ra) (1/sqrt(Ohm*cm))",
                 ylabel="lambda (um)", title="Space Constant vs 1/sqrt(Ra)")
ax_lambda_Ra.legend(); ax_lambda_Ra.grid(True)

# delta-V vs sqrt(Ra): Rin ~ sqrt(Ra), so delta-V is also linear in sqrt(Ra).
sqrt_Ra            = np.sqrt(Ra_values)
slope_dv_Ra, _     = np.polyfit(sqrt_Ra, delta_v_center_Ra, 1)
y_fit_center       = slope_dv_Ra * np.linspace(0, max(sqrt_Ra) * 1.1, 100)
ax_dv_center_Ra.plot(sqrt_Ra, delta_v_center_Ra, "o", color="blue",
                     markersize=10, label="Simulated")
ax_dv_center_Ra.plot(np.linspace(0, max(sqrt_Ra) * 1.1, 100), y_fit_center,
                     "--", color="black", linewidth=2,
                     label=f"Regression: m={slope_dv_Ra:.2e}")
ax_dv_center_Ra.set(xlabel="sqrt(Ra) (sqrt(Ohm*cm))",
                    ylabel="Peak dV (mV)", title="dV vs sqrt(Ra) (Center)")
ax_dv_center_Ra.legend(); ax_dv_center_Ra.grid(True)

slope_dv_end_Ra, _ = np.polyfit(sqrt_Ra, delta_v_end_Ra, 1)
y_fit_end          = slope_dv_end_Ra * np.linspace(0, max(sqrt_Ra) * 1.1, 100)
ax_dv_end_Ra.plot(sqrt_Ra, delta_v_end_Ra, "o", color="purple",
                  markersize=10, label="Simulated (end)")
ax_dv_end_Ra.plot(np.linspace(0, max(sqrt_Ra) * 1.1, 100), y_fit_end, "--",
                  color="black", linewidth=2,
                  label=f"Regression: m={slope_dv_end_Ra:.2e}")
ax_dv_end_Ra.set(xlabel="sqrt(Ra) (sqrt(Ohm*cm))",
                 ylabel="Peak dV (mV)", title="dV vs sqrt(Ra) (End)")
ax_dv_end_Ra.legend(); ax_dv_end_Ra.grid(True)

plt.tight_layout()
plt.show()
