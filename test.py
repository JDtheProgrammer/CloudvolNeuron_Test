import neuron
import pprint
import textwrap
import matplotlib.pyplot as plt
import csv
import plotnine as p9
import pandas as pd
import numpy as np
import math
from math import pi

print('\n NEURON Version:\n', neuron.__version__)

from neuron import n 
from neuron.units import ms, mV, um

# creat a cell with a single section (soma)
soma = n.Section('soma')
#print(soma.Ra)
n.load_file("stdrun.hoc") # For higher-level simulation control specification, 
#we load NEURON’s stdrun library

iclamp = n.IClamp(soma(0.5)) # create an IClamp object at the center of the soma

iclamp.delay = 0
iclamp.dur = 15
iclamp.amp = 0.9

#----------------------------------------------------------------------------------------
#                                 1. verying Current Pulses
#----------------------------------------------------------------------------------------

#print('\n1. Varying Current Pulses\n')
soma.insert('pas') # insert passive properties
#print(soma(0.5).pas.g)
#print(soma(0.5).pas.e)
soma(0.5).pas.g = 0.002  # S/cm²
Rm = 1 / soma(0.5).pas.g  # Ω-cm²
#print(soma(0.5).pas.g)
soma(0.5).pas.e = -70   # Set reversal potential
#print(", ".join(item for item in dir(neuron) if not item.startswith("__")))
#print(vars(pas))
#print(soma.insert('pas'))
soma.L = 20 * um  # µm
L = soma.L 
v_sim = [] 

diam = 1 * um  # Diameter list
colors = ["green", "blue", "red", "black"]
amps = [(0.075) * k for k in range(1, 5)]
v_rest = -70 # mV

# Surface area in cm²
A = math.pi * (diam*1e-4) * (L*1e-4)  # um → cm

#Resisitvity and resistance calculation
Rm = 1 / soma(0.5).pas.g  # Ω·cm²
Rn = Rm / A  # Ω

# Convert amps from nA to A
amps_A = np.array(amps) * 1e-6  # nA → mA
amps_theoretical = np.linspace(0,0.35,20)
v_theoretical = amps_theoretical * Rn *1e-6 # mV
# Create figure with 1 row and 2 columns (side by side)
fig, (ax_volt, ax_pulse, ax_relation) = plt.subplots(1, 3, figsize=(18, 5))

# LEFT PLOT: Voltage vs Time (superimposed)
for amp, color in zip(amps, colors):
    soma.diam = diam
    soma.cm = 1 
    iclamp.amp = amp # to introduce the amperage in the simulation
    
    # Recreate recording vectors for each simulation
    v = n.Vector().record(soma(0.5)._ref_v)  # ← Records voltage
    t = n.Vector().record(n._ref_t)          # ← Records time
    
    n.finitialize(v_rest * mV)  # ← Initialize simulation at resting potential
    n.continuerun(25 * ms)      # ← **RUN THE SIMULATION for 25 ms**
    
    # Plot voltage on left subplot
    ax_volt.plot(list(t), list(v), color=color, linewidth=2, 
             label=f"diam={diam:.2f} µm, g={soma(0.5).pas.g:.2f} S/cm², amp={amp:.3f} nA")
    
    v_list = list(v)
    t_list = list(t)
    ax_volt.plot(t_list, v_list, color=color, linewidth=4, label=f"amp={amp:.3f} nA")

    # ✅ Find absolute voltage deflection
    
    v_max = max(v_list)
    delta_v = abs(v_max - v_rest)
    v_sim.append(delta_v)

    print(f"Amperages = {amp:.3f} nA → ΔV = {delta_v:.2f} mV (Max V = {v_max:.2f} mV)")
    
ax_volt.set_xlim(0, 25)
#ax_volt.set_ylim(-75, 5000)
ax_volt.set_xlabel("Time (ms)")
ax_volt.set_ylabel("Voltage (mV)")
ax_volt.set_title("Soma Voltage vs Time")
ax_volt.legend(fontsize=8)
ax_volt.grid(True)


# Right side superimposed plot of all pulses

t_array = np.linspace(0, 25, 500)
for amp, color in zip(amps, colors):
    pulse = np.where((t_array >= iclamp.delay) & (t_array < iclamp.delay + iclamp.dur), amp, 0)
    ax_pulse.plot(t_array, pulse, color=color, linewidth=4, label=f"amp={amp:.3f} nA")

ax_pulse.set_xlim(0, 25)
ax_pulse.set_ylim(0, max(amps) * 1.2)
ax_pulse.set_xlabel("Time (ms)")
ax_pulse.set_ylabel("Current (nA)")
ax_pulse.set_title("Input Current Pulses (Superimposed)")
ax_pulse.grid(True)
ax_pulse.legend(fontsize=11)

# linear regression for section 1
#------------------------------------------------------------------
# Calculate slope (m) and intercept (b) using numpy's polyfit
coefficients = np.polyfit(amps, v_sim, 1)  # degree=1 for linear fit
slope = coefficients[0]
intercept = coefficients[1]

# Generate regression line points
x_regression = np.linspace(0, max(amps), 100)
y_regression = slope * x_regression + intercept

# Calculate R² value
correlation_matrix = np.corrcoef(amps, v_sim)
correlation = correlation_matrix[0, 1]
r_squared = correlation**2

print(f"\nLinear Regression Results:")
print(f"Slope: {slope:.4f} mV/nA")
print(f"Intercept: {intercept:.4f} mV")
print(f"R² value: {r_squared:.6f}")
print(f"Equation: ΔV = {slope:.4f} * I + {intercept:.4f}")

# Relationship Graph Voltage vs. Current pulse
# Prepend 0 to start from origin
ax_relation.set_xlim(0, max(amps) * 1.2)
ax_relation.set_ylim(0, max(v_sim) * 1.2)
ax_relation.plot(x_regression, y_regression, '-', color='black',
                  label=f'Linear Fit (R²={r_squared:.4f})', 
                 linewidth=4, alpha=1)
ax_relation.plot(amps_theoretical, v_theoretical, color='red', label='Theoretical', linewidth=4, 
                  linestyle=(0, (2, 2)))  # (offset, (dash_length, gap_length))
ax_relation.plot(amps, v_sim, 'o', label='Experimental',
                 color='blue', linewidth=4, markersize=9)
ax_relation.set_xlabel("Injected Current (nA)")
ax_relation.set_ylabel("Peak Voltage Difference (mV)")
ax_relation.set_title("Peak Voltage Difference vs Injected Current")
ax_relation.grid(True)
ax_relation.legend(fontsize=11)
#print(soma.L)
#print(soma.psection())
plt.tight_layout()
plt.show(block=False)

#----------------------------------------------------------------------------------------
# Section 2: Varying Diameters + Linear Regression (Transposed Layout)
#----------------------------------------------------------------------------------------

# Simulation settings
diameters = [(1 * um) * k for k in range(1, 5)]   # 1–4 µm
colors = ["green", "blue", "red", "black"]
amp = 0.075
v_rest = -70

delta_v_list = []
diam_um_list = []

# Figure layout: 2 rows × 3 columns
fig2, ax = plt.subplots(2, 3, figsize=(18, 10))

# Transposed plot positions
ax_volt = ax[0, 0]            # Voltage vs time
ax_diam = ax[1, 0]            # Diameter bar plot
ax_relation = ax[0, 1]        # ΔV vs 1/D
ax_RNDiamRelation = ax[1, 1]  # Rin vs 1/D
ax_relation_reg = ax[0, 2]    # Regression ΔV
ax_RN_reg = ax[1, 2]          # Regression Rin

# ---------------------------------------------------------------
# Run simulations for different diameters
# ---------------------------------------------------------------
for d, color in zip(diameters, colors):
    soma.diam = d
    iclamp.amp = amp

    v = n.Vector().record(soma(0.5)._ref_v)
    t = n.Vector().record(n._ref_t)

    n.finitialize(v_rest * mV)
    n.continuerun(25 * ms)

    v_list = list(v)
    t_list = list(t)

    # Voltage traces
    ax_volt.plot(t_list, v_list, color=color, linewidth=2,
                 label=f"diam={d/um:.1f} µm")

    # Store ΔV and diameters
    delta_v = abs(max(v_list) - v_rest)
    delta_v_list.append(delta_v)
    diam_um_list.append(d / um)

ax_volt.set_title("Soma Voltage vs Time")
ax_volt.set_xlabel("Time (ms)")
ax_volt.set_ylabel("Voltage (mV)")
ax_volt.legend(fontsize=11)
ax_volt.grid(True)

# ---------------------------------------------------------------
# Diameter bar plot
# ---------------------------------------------------------------
ax_diam.bar(range(len(diam_um_list)), diam_um_list,
            color=colors,
            tick_label=[f"{d:.1f}" for d in diam_um_list])

ax_diam.set_title("Diameters Used")
ax_diam.set_xlabel("Condition #")
ax_diam.set_ylabel("Diameter (µm)")
ax_diam.grid(True, axis='y')

# ---------------------------------------------------------------
# Compute geometry + theory - FIXED
# ---------------------------------------------------------------
L_cm = soma.L * 1e-4  # µm → cm
diam_cm_list = [d * 1e-4 for d in diam_um_list]  # µm → cm
A_list = [np.pi * d * L_cm for d in diam_cm_list]  # cm²

Rin_list = [1/soma(0.5).pas.g / A for A in A_list]  # Ω
I_amp = amp * 1e-9  # nA → A

inv_diam_list = [1/d for d in diam_um_list]  # 1/µm

# CREATE INDEPENDENT THEORETICAL CURVES (starting from 0)
inv_diam_theory = np.linspace(0, max(inv_diam_list)*1.1, 20)  # 1/µm

# Theory: Rin = Rm / (π × d × L)
# inv_diam is in (1/µm), but formula needs (1/cm)
# So: Rin = Rm / (π × L) × (1/d_cm) = Rm / (π × L) × (1/d_µm) × (1e4)
Rm = 1 / soma(0.5).pas.g  # Ω·cm²
k_Rin = Rm / (np.pi * L_cm) * 1e4  # Ω·µm (multiply by 1e4 to convert 1/cm to 1/µm)
Rin_theory = k_Rin * inv_diam_theory  # Ω

# Theory: ΔV = I × Rin (in mV)
k_deltaV = I_amp * k_Rin * 1000  # A × Ω·µm × (mV/V) = mV·µm
deltaV_theory = k_deltaV * inv_diam_theory  # mV
# After calculating the theory, add these prints:
print(f"\nDEBUG INFO:")
print(f"L_cm = {L_cm}")
print(f"Rm = {Rm}")
print(f"I_amp = {I_amp}")
print(f"k_Rin = {k_Rin}")
print(f"k_deltaV = {k_deltaV}")
print(f"\ninv_diam_list = {inv_diam_list}")
print(f"delta_v_list = {delta_v_list}")
print(f"Rin_list = {Rin_list}")
print(f"\ninv_diam_theory[:5] = {inv_diam_theory[:5]}")
print(f"deltaV_theory[:5] = {deltaV_theory[:5]}")
print(f"Rin_theory[:5] = {Rin_theory[:5]}")
# ---------------------------------------------------------------
# ΔV vs 1/D (Simulation points + Theory line)
# ---------------------------------------------------------------
ax_relation.plot(inv_diam_theory, deltaV_theory, "x", markersize=10,
                 linewidth=2, color=(0.0,0.8,0.0), label="Theoretical ∝ 1/D")
ax_relation.plot(inv_diam_list, delta_v_list, "o",
                 markersize=11, markeredgewidth=3, color=(0.8,0.2,0.8), 
                 label="Simulated ΔV", linestyle='None')

ax_relation.set_title("Peak Voltage vs 1/Diameter")
ax_relation.set_xlabel("1 / Diameter (1/µm)")
ax_relation.set_ylabel("Peak Voltage ΔV (mV)")
ax_relation.grid(True)
ax_relation.legend(fontsize=11)

# ---------------------------------------------------------------
# Rin vs 1/D (Simulation points + Theory line)
# ---------------------------------------------------------------
ax_RNDiamRelation.plot(inv_diam_theory, Rin_theory, "x", markersize=10,
                       linewidth=2, color=(0.8,0.0,0.0), label="Theoretical ∝ 1/D")
ax_RNDiamRelation.plot(inv_diam_list, Rin_list, "o", 
                       markersize=11, markeredgewidth=3, color=(0.2,0.8,0.8), 
                       label="Simulated Rin", linestyle='None')

ax_RNDiamRelation.set_title("Input Resistance vs 1/Diameter")
ax_RNDiamRelation.set_xlabel("1 / Diameter (1/µm)")
ax_RNDiamRelation.set_ylabel("Rin (Ω)")
ax_RNDiamRelation.grid(True)
ax_RNDiamRelation.legend(fontsize=11)

# ---------------------------------------------------------------
# Linear Regression — ΔV (NO FAKE ORIGIN)
# ---------------------------------------------------------------
X = np.array(inv_diam_list)
y = np.array(delta_v_list)

slope, intercept = np.polyfit(X, y, 1)
X_fit = np.linspace(0, max(inv_diam_list)*1.1, 100)
y_fit = slope * X_fit + intercept

ax_relation_reg.plot(inv_diam_list, delta_v_list, "o", 
                     markersize=10, color="blue", label="Simulated")
ax_relation_reg.plot(X_fit, y_fit, "r--", linewidth=2,
                     label=f"Fit: y={slope:.2f}x + {intercept:.2e}")

ax_relation_reg.set_title("Regression: ΔV vs 1/D")
ax_relation_reg.set_xlabel("1 / Diameter (1/µm)")
ax_relation_reg.set_ylabel("ΔV (mV)")
ax_relation_reg.grid(True)
ax_relation_reg.legend(fontsize=11)

# ---------------------------------------------------------------
# Linear Regression — Rin (NO FAKE ORIGIN)
# ---------------------------------------------------------------
y_rin = np.array(Rin_list)
slope_rin, intercept_rin = np.polyfit(X, y_rin, 1)
y_rin_fit = slope_rin * X_fit + intercept_rin

ax_RN_reg.plot(inv_diam_list, Rin_list, "o", 
               markersize=10, color="purple", label="Simulated")
ax_RN_reg.plot(X_fit, y_rin_fit, "r--", linewidth=2,
               label=f"Fit: y={slope_rin:.2e}x + {intercept_rin:.2e}")

ax_RN_reg.set_title("Regression: Rin vs 1/D")
ax_RN_reg.set_xlabel("1 / Diameter (1/µm)")
ax_RN_reg.set_ylabel("Rin (Ω)")
ax_RN_reg.grid(True)
ax_RN_reg.legend(fontsize=11)

plt.tight_layout()
plt.show()
