import neuron
import pprint
import textwrap
import matplotlib.pyplot as plt
import csv
import plotnine as p9
import pandas as pd
import numpy as np
import math
from scipy.optimize import curve_fit    
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
soma(0.5).pas.g = 0.002  # sets conductance in S/cm²
Rm = 1 / soma(0.5).pas.g # set resistance to 500 # Ω-cm²
#print(soma(0.5).pas.g)
soma(0.5).pas.e = -70   # Set reversal potential
#print(", ".join(item for item in dir(neuron) if not item.startswith("__")))
#print(vars(pas))
#print(soma.insert('pas'))
soma.L = 20 * um  # sets length of soma to 20 µm
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
amps_A = np.array(amps)  # nA → mA
amps_theoretical = np.linspace(0,0.35,20) 
v_theoretical = amps_theoretical * Rn * 1e-9*1000 # mV
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
tau = Rm * soma.cm # time constant in ms where some.cm is in uF/cm²
print(f"Time constant τ = {tau:.2f} ms")
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
ax_pulse.set_title("Input Current Pulses")
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
                  label=f'Linear Fit: y={slope:3e}x + {intercept:.2f}', 
                 linewidth=4, alpha=1)
ax_relation.plot(amps, v_sim, 'o', label='Experimental',
                 color='blue', linewidth=4, markersize=9)
ax_relation.plot(amps_theoretical, v_theoretical, 'x',color='red', label=f"Theoretical={Rn:.2e} Ω",
                  linewidth=6,
                 markersize=9) 
                 # linestyle=(0, (2, 2)))  # (offset, (dash_length, gap_length))
ax_relation.set_xlabel("Injected Current (nA)")
ax_relation.set_ylabel("Peak Voltage Difference (mV)")
ax_relation.set_title("Peak Voltage Difference vs Injected Current")
ax_relation.grid(True)
ax_relation.legend(fontsize=11)
ax_relation.text(
    0.05, 0.95,
    f"$R^2$ = {r_squared:.2f}",
    transform=ax_relation.transAxes,
    fontsize=11,
    verticalalignment="top"
)

plt.tight_layout()
plt.show(block=False)
print(soma.L)
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

# Figure layout: 2 rows × 2 columns
fig2, ax = plt.subplots(2, 2, figsize=(14, 10))

# Transposed plot positions
ax_volt = ax[0, 0]            # Voltage vs time
ax_diam = ax[1, 0]            # Diameter bar plot
ax_relation = ax[0, 1]        # ΔV vs 1/D
ax_RNDiamRelation = ax[1, 1]  # Rin vs 1/D

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
m_Rin = Rm / (np.pi * L_cm) * 1e4  # Ω·µm (multiply by 1e4 to convert 1/cm to 1/µm)
Rin_theory = m_Rin * inv_diam_theory  # Ω

# Theory: ΔV = I × Rin (in mV)
m_deltV = I_amp * m_Rin * 1000  # A × Ω·µm × (mV/V) = mV·µm
deltaV_theory = m_deltV * inv_diam_theory  # mV
# After calculating the theory, add these prints:
#print(f"\nDEBUG INFO:")
#print(f"L_cm = {L_cm}")
#print(f"Rm = {Rm}")
#print(f"I_amp = {I_amp}")
#print(f"m_Rin = {m_Rin}")
#print(f"m_deltV = {m_deltV}")
#print(f"\ninv_diam_list = {inv_diam_list}")
#print(f"delta_v_list = {delta_v_list}")
#print(f"Rin_list = {Rin_list}")
#print(f"\ninv_diam_theory[:5] = {inv_diam_theory[:5]}")
#print(f"deltaV_theory[:5] = {deltaV_theory[:5]}")
#print(f"Rin_theory[:5] = {Rin_theory[:5]}")

# ---------------------------------------------------------------
# Linear Regression — ΔV (NO FAKE ORIGIN)
# ---------------------------------------------------------------
X = np.array(inv_diam_list)
y = np.array(delta_v_list)

slope, intercept = np.polyfit(X, y, 1)
X_fit = np.linspace(0, max(inv_diam_list)*1.1, 100)
y_fit = slope * X_fit + intercept
# R² for ΔV vs 1/D
r_matrix = np.corrcoef(inv_diam_list, delta_v_list)
r_squared_dV = r_matrix[0, 1]**2

# ---------------------------------------------------------------
# ΔV vs 1/D (Simulation points + Theory line)
# ---------------------------------------------------------------
ax_relation.plot(inv_diam_theory, deltaV_theory, "x", markersize=10,
                 linewidth=2, color=(0.0,0.8,0.0), label=f"Theoretical={m_deltV:.2f} mV·µm")
ax_relation.plot(inv_diam_list, delta_v_list, "o",
                 markersize=11, markeredgewidth=3, color=(0.8,0.2,0.8), 
                 label="Simulated ΔV", linestyle='None')
# Regression line OVERLAID on ΔV vs 1/D plot
ax_relation.plot(X_fit, y_fit, "r--", linewidth=3,
                 label=f"Linear fit: ΔV = {slope:.2f}x + {intercept:.2e}")

ax_relation.set_title("Peak Voltage vs 1/Diameter")
ax_relation.set_xlabel("1 / Diameter (1/µm)")
ax_relation.set_ylabel("Peak Voltage ΔV (mV)")
ax_relation.grid(True)
ax_relation.legend(fontsize=11)
ax_relation.text(
    0.05, 0.95,
    f"$R^2$ = {r_squared_dV:.4f}",
    transform=ax_relation.transAxes,
    fontsize=11,
    verticalalignment="top"
)

# ---------------------------------------------------------------
# Linear Regression — Rin (NO FAKE ORIGIN)
# ---------------------------------------------------------------
y_rin = np.array(Rin_list)
slope_rin, intercept_rin = np.polyfit(X, y_rin, 1)
y_rin_fit = slope_rin * X_fit + intercept_rin
# R² for Rin vs 1/D
r_matrix = np.corrcoef(inv_diam_list, Rin_list)
r_squared_Rin = r_matrix[0, 1]**2

# ---------------------------------------------------------------
# Rin vs 1/D (Simulation points + Theory line)
# ---------------------------------------------------------------
ax_RNDiamRelation.plot(inv_diam_theory, Rin_theory, "x", markersize=10,
                       linewidth=2, color=(0.4,0.0,0.0), label=f"Theoretical={m_Rin:.2f} Ω·µm")
ax_RNDiamRelation.plot(inv_diam_list, Rin_list, "o", 
                       markersize=11, markeredgewidth=3, color=(0.2,0.8,0.8), 
                       label="Simulated Rin", linestyle='None')
# Regression line OVERLAID on Rin vs 1/D plot
ax_RNDiamRelation.plot(X_fit, y_rin_fit, "r--", linewidth=3,
                       label=f"Linear Fit: Rin = {slope_rin:.2e}x + {intercept_rin:.2e}")

ax_RNDiamRelation.set_title("Input Resistance vs 1/Diameter")
ax_RNDiamRelation.set_xlabel("1 / Diameter (1/µm)")
ax_RNDiamRelation.set_ylabel("Rin (Ω)")
ax_RNDiamRelation.grid(True)
ax_RNDiamRelation.legend(fontsize=11)
ax_RNDiamRelation.text(
    0.05, 0.95,
    f"$R^2$ = {r_squared_Rin:.4f}",
    transform=ax_RNDiamRelation.transAxes,
    fontsize=11,
    verticalalignment="top"
)


plt.tight_layout()

plt.show(block=False)
#save from code
# no markers at zero for regression lines (accomplished)
# get rid of extension at origin for 2 COLUMN GRAPHS (accomplished)
# Simulation settings


#----------------------------------------------------------------------------------------
#                                 3. verying Resistivity
#----------------------------------------------------------------------------------------

print("\n" + "="*60)
print("SECTION 3: VARYING SPECIFIC MEMBRANE RESISTANCE (Rm)")
print("="*60)

# Rm values in Ω·cm² (common biological range)
Rm_values = [500, 1e3, 1.5e3, 2e3]  # 500, 1k, 1.5k, 2k Ω·cm²
colors = ["green", "blue", "red", "black"]

# Fixed soma parameters
diam = 1 * um
soma.diam = diam

# Stimulation parameters
amp = 0.075  # nA
v_rest = -70

iclamp.delay = 0
iclamp.dur = 15
iclamp.amp = amp

# Calculate geometry (independent of Rm)
d_cm = soma.diam * 1e-4          # µm → cm
L_cm = soma.L * 1e-4             # µm → cm
A = math.pi * d_cm * L_cm        # cm²
I = amp * 1e-9                   # nA → A

# THEORETICAL PROPORTIONALITY CONSTANTS (independent of simulation)
# For ΔV vs Rm: ΔV = I × (Rm/A) × 1000 = (I/A × 1000) × Rm
m_deltaV_theory = (I / A) * 1000  # mV per (Ω·cm²)

# For Rin vs Rm: Rin = Rm/A
m_Rin_theory = 1 / A  # Ω per (Ω·cm²), which simplifies to 1/cm²

print(f"\nGeometry:")
print(f"  Diameter = {soma.diam:.2f} µm")
print(f"  Length = {soma.L:.2f} µm")
print(f"  Surface Area = {A:.6e} cm²")
print(f"\nStimulation:")
print(f"  Current injection = {amp:.3f} nA = {I:.3e} A")
print(f"  Pulse duration = {iclamp.dur:.1f} ms")
print(f"\nTheoretical Proportionality Constants:")
print(f"  k_ΔV = I/A × 1000 = {m_deltaV_theory:.6e} mV/(Ω·cm²)")
print(f"  m_Rin = 1/A = {m_Rin_theory:.6e} Ω/(Ω·cm²) = {m_Rin_theory:.6e} cm⁻²")
print(f"\n  Expected: ΔV = {m_deltaV_theory:.6e} × Rm")
print(f"  Expected: Rin = {m_Rin_theory:.6e} × Rm")
print("="*60 + "\n")

# Create figure
fig3, ax = plt.subplots(2, 2, figsize=(14, 10))

ax_volt3 = ax[0, 0]           # Voltage traces
ax_Rm = ax[1, 0]              # Rm bar graph
ax_relation3 = ax[0, 1]       # ΔV vs Rm (sim + theory)
ax_Rin = ax[1, 1]             # Rin vs Rm (sim + theory)

# Storage for simulation results
delta_v_Rm = []
Rin_values = []
Rm_values_list = []

# Run simulations
print("Simulation Results:")
for Rm, color in zip(Rm_values, colors):
    soma(0.5).pas.g = 1.0 / Rm

    v = n.Vector().record(soma(0.5)._ref_v)
    t = n.Vector().record(n._ref_t)

    n.finitialize(v_rest * mV)
    n.continuerun(25 * ms)

    v_list = list(v)
    t_list = list(t)

    # Plot voltage trace
    ax_volt3.plot(t_list, v_list, color=color, linewidth=2,
                  label=f"Rm = {Rm/1000:.1f} kΩ·cm²")

    # Compute voltage deflection
    v_max = max(v_list)
    delta_v = abs(v_max - v_rest)
    
    # Compute Rin for this Rm
    Rin = Rm / A
    
    delta_v_Rm.append(delta_v)
    Rin_values.append(Rin)
    Rm_values_list.append(Rm)

    # Theoretical predictions
    delta_v_theory = m_deltaV_theory * Rm
    Rin_theory = m_Rin_theory * Rm

    print(f"  Rm = {Rm:6.0f} Ω·cm² → ΔV_sim = {delta_v:.4f} mV, ΔV_theory = {delta_v_theory:.4f} mV")
    print(f"                    → Rin_sim = {Rin:.4e} Ω, Rin_theory = {Rin_theory:.4e} Ω")

# Voltage plot
ax_volt3.set_xlim(0, 25)
ax_volt3.set_xlabel("Time (ms)")
ax_volt3.set_ylabel("Voltage (mV)")
ax_volt3.set_title("Voltage Response vs Specific Membrane Resistance (Rm)")
ax_volt3.legend(fontsize=11)
ax_volt3.grid(True)

# Rm bar plot
ax_Rm.bar(range(len(Rm_values)), Rm_values, color=colors, 
          tick_label=[f"{Rm:.0f}" for Rm in Rm_values])
ax_Rm.set_xlabel("Condition #")
ax_Rm.set_ylabel("Rm (Ω·cm²)")
ax_Rm.set_title("Resistivities Used in Simulation")
ax_Rm.grid(True, axis='y')

# INDEPENDENT THEORETICAL CURVES (starting from 0)
Rm_theory = np.linspace(0, max(Rm_values) * 1.1, 20)
delta_v_theory_curve = m_deltaV_theory * Rm_theory
Rin_theory_curve = m_Rin_theory * Rm_theory

# Linear Regression — ΔV vs Rm (independent from theory)
X = np.array(Rm_values_list)
y = np.array(delta_v_Rm)
slope_deltaV, intercept_deltaV = np.polyfit(X, y, 1)

X_fit = np.linspace(0, max(Rm_values) * 1.1, 100)
y_deltaV_fit = slope_deltaV * X_fit + intercept_deltaV
r2_dV = np.corrcoef(X, y)[0, 1]**2

# ΔV vs Rm plot
ax_relation3.plot(Rm_theory, delta_v_theory_curve, 'x', color='red',
                  linewidth=3, alpha=0.8, label=f'Theoretical m={m_deltaV_theory:.2e} A/cm²')
ax_relation3.plot(Rm_values_list, delta_v_Rm, 'o', color='blue',
                  markersize=10, label='Simulated', linestyle='None')
ax_relation3.plot(X_fit, y_deltaV_fit, '--', color='black', linewidth=2, 
                  label=f"linear Fit: y={slope_deltaV:.2f}x + {intercept_deltaV:.2e}")
ax_relation3.set_xlabel("Membrane Resistivity Rm (Ω·cm²)")
ax_relation3.set_ylabel("Peak Voltage ΔV (mV)")
ax_relation3.set_title("Peak Voltage vs Specific Membrane Resistance")
ax_relation3.grid(True)
ax_relation3.legend(fontsize=11)
ax_relation3.set_xlim(0, max(Rm_values) * 1.1)
ax_relation3.set_ylim(0, max(max(delta_v_Rm), max(delta_v_theory_curve)) * 1.1)

ax_relation3.text(
    0.05, 0.95,
    f"$R^2$ = {r2_dV:.2f}",
    transform=ax_relation3.transAxes,
    fontsize=11,
    verticalalignment="top"
)

# Linear Regression — Rin vs Rm (independent from theory)
y_rin = np.array(Rin_values)
slope_Rin, intercept_Rin = np.polyfit(X, y_rin, 1)
y_Rin_fit = slope_Rin * X_fit + intercept_Rin
r2_Rin = np.corrcoef(X, y_rin)[0, 1]**2

# Rin vs Rm plot
ax_Rin.plot(Rm_theory, Rin_theory_curve, 'x', color='red',
            linewidth=3, alpha=0.8, label=f"Theoretical m={m_Rin_theory:.2e} 1/cm²")
ax_Rin.plot(Rm_values_list, Rin_values, 'o', color='purple',
            markersize=10, label='Simulated', linestyle='None')
ax_Rin.plot(X_fit, y_Rin_fit, '--', color='black', linewidth=2,
             label=f"Linear Fit: y={slope_Rin:.6e}x + {intercept_Rin:.2e}")

ax_Rin.set_xlabel("Specific Membrane Resistance Rm (Ω·cm²)")
ax_Rin.set_ylabel("Input Resistance Rin (Ω)")
ax_Rin.set_title("Input Resistance vs Membrane Resistivity")
ax_Rin.grid(True)
ax_Rin.legend(fontsize=11)
ax_Rin.set_xlim(0, max(Rm_values) * 1.1)
ax_Rin.set_ylim(0, max(max(Rin_values), max(Rin_theory_curve)) * 1.1)

ax_Rin.text(
    0.05, 0.95,
    f"$R^2$ = {r2_Rin:.2f}",
    transform=ax_Rin.transAxes,
    fontsize=11,
    verticalalignment="top"
)


# Print regression results
print(f"\nLinear Regression Results (ΔV vs Rm):")
print(f"  Simulated slope (k_sim) = {slope_deltaV:.6e} mV/(Ω·cm²)")
print(f"  Theoretical slope (k_theory) = {m_deltaV_theory:.6e} mV/(Ω·cm²)")
print(f"  Intercept = {intercept_deltaV:.6e} mV")
print(f"  Difference = {abs(m_deltaV_theory - slope_deltaV):.6e} ({abs(m_deltaV_theory - slope_deltaV)/m_deltaV_theory*100:.2f}%)")

print(f"\nLinear Regression Results (Rin vs Rm):")
print(f"  Simulated slope (k_sim) = {slope_Rin:.6e} Ω/(Ω·cm²)")
print(f"  Theoretical slope (k_theory) = {m_Rin_theory:.6e} Ω/(Ω·cm²)")
print(f"  Intercept = {intercept_Rin:.6e} Ω")
print(f"  Difference = {abs(m_Rin_theory - slope_Rin):.6e} ({abs(m_Rin_theory - slope_Rin)/m_Rin_theory*100:.2f}%)")

plt.tight_layout()
plt.show(block=False)

# ----------------------------------------------------------------------------------------
# SECTION 4: VARYING MEMBRANE CAPACITANCE (Cm) – demonstrates passive RC behavior
# ----------------------------------------------------------------------------------------

print("\n" + "="*60)
print("SECTION 4: VARYING MEMBRANE CAPACITANCE (Cm)")
print("="*60)

# Test values – biological range is ~0.5–2 µF/cm²; we go up to 4 to exaggerate effect
Cm_values = [0.5, 1.0, 2.0, 4.0]          # µF/cm²
colors    = ["green", "blue", "red", "black"]

# Fixed passive leak conductance (same as earlier sections)
soma(0.5).pas.g = 0.002                   # S/cm² → Rm = 500 Ω·cm²
Rm = 1 / soma(0.5).pas.g

# Geometry (using current soma dimensions)
soma.diam = 1 * um
d_cm = soma.diam * 1e-4
L_cm = soma.L * 1e-4
A = math.pi * d_cm * L_cm                 # surface area in cm²

# Stimulation – same small current used before
amp = 0.075                               # nA
I   = amp * 1e-9                          # A

# Theoretical predictions (independent of simulation)
deltaV_theory    = I * (Rm / A) * 1000    # mV – should be ~constant
tau_theory_slope = Rm * 0.001             # ms per (µF/cm²) → 0.500 ms/(µF/cm²)

print(f"\nGeometry & parameters:")
print(f"  Diameter   = {soma.diam/um:.1f} µm")
print(f"  Length     = {soma.L/um:.1f} µm")
print(f"  Area A     = {A:.2e} cm²")
print(f"  Rm         = {Rm:.0f} Ω·cm²")
print(f"  I          = {amp:.3f} nA  →  {I:.2e} A")
print(f"  Expected steady-state ΔV ≈ {deltaV_theory:.2f} mV (independent of Cm)")
print(f"  Theoretical τ = Rm × Cm × 0.001 → slope = {tau_theory_slope:.3f} ms/(µF/cm²)")
print("="*60 + "\n")

# Exponential charging function: V(t) = V0 + ΔV × (1 − exp(−t/τ))
def charging_func(t, V0, deltaV, tau):
    return V0 + deltaV * (1 - np.exp(-t / tau))

# Exponential form for ΔV vs Cm: ΔV = a * (1 - exp(-b / Cm))  (due to finite pulse duration)
def deltaV_vs_Cm_func(Cm, a, b):
    return a * (1 - np.exp(-b / Cm))

# Layout: 2×2 figure
fig4, ax = plt.subplots(2, 2, figsize=(15, 10))
ax_volt    = ax[0, 0]     # Voltage traces + exponential fits
ax_dV      = ax[0, 1]     # Peak ΔV vs Cm (should be flat)
ax_Cm_bar  = ax[1, 0]     # Bar plot of tested Cm values
ax_tau     = ax[1, 1]     # τ vs Cm (should be linear)

# Storage for results
peak_dV_sim = []          # will store simulated measured ΔV
tau_fitted  = []

# ────────────────────────────────────────────────
# Main simulation loop – one run per Cm value
# ────────────────────────────────────────────────
for Cm, color in zip(Cm_values, colors):
    soma.cm = Cm                          # set specific capacitance

    # Record voltage and time
    v = n.Vector().record(soma(0.5)._ref_v)
    t = n.Vector().record(n._ref_t)

    n.finitialize(v_rest * mV)
    n.continuerun(25 * ms)

    t_arr = np.array(t)
    v_arr = np.array(v)

    # Measured peak (simulated raw max value)
    V_rest_sim = v_arr[0] # This line gets the resting potential from the start of the trace in the form of an array.
    dV_measured = v_arr.max() - V_rest_sim # this operation gives peak ΔV by subtracting resting potential
    peak_dV_sim.append(dV_measured) # here append means to add the measured peak ΔV to the list for later plotting

    # Fit exponential only to the rising phase (during current pulse)
    mask_pulse = (t_arr >= iclamp.delay) & (t_arr <= iclamp.delay + iclamp.dur)
    if np.sum(mask_pulse) > 20:           # need enough points for reliable fit
        t_fit = t_arr[mask_pulse]         # extrancects time from simulation
        v_fit = v_arr[mask_pulse]         # extracts voltage from simulation
        t_fit -= t_fit[0]                 # shift to start at t=0
        v_fit -= V_rest_sim               # shift to start at 0 mV deflection

        # Initial guess: reasonable values based on theory
        p0 = [0, dV_measured * 0.95, Rm * Cm * 0.001 * 1.1] # the 0.95 and 1.1 are just to help the fitting process
        # the 0.001 factor converts from µF/cm² and Ω·cm² to ms because before the conversion the units are in microseconds
        try: # the try-except block is used to catch any errors during the fitting process
            popt, _ = curve_fit(charging_func, t_fit, v_fit, # popt stands for optimal parameters
                                p0=p0,
                                bounds=([-np.inf, 0, 0], [np.inf, np.inf, 1000]))
            V0_fit, dV_fit, tau_fit = popt

            tau_fitted.append(tau_fit) 

            # Plot smooth fitted curve
            t_smooth = np.linspace(0, iclamp.dur, 200)
            v_smooth = charging_func(t_smooth, V0_fit, dV_fit, tau_fit) + V_rest_sim
            t_plot_smooth = t_smooth + iclamp.delay
            ax_volt.plot(t_plot_smooth, v_smooth, '--', color=color, alpha=0.7, lw=1.8,
                         label=f"fit τ={tau_fit:.1f} ms")

            print(f"Cm = {Cm:4.1f} → ΔV_sim = {dV_measured:5.2f} mV   τ_fit = {tau_fit:5.1f} ms")

        except Exception as e:
            print(f"  Fit failed for Cm = {Cm}: {e}")
            tau_fitted.append(np.nan)
    else:
        tau_fitted.append(np.nan)

    # Plot full simulated trace
    ax_volt.plot(t_arr, v_arr, color=color, lw=2.2,
                 label=f"Cm = {Cm:.1f} µF/cm²")

# ────────────────────────────────────────────────
# Fit exponential regression to ΔV vs Cm (captures slight dependence due to finite pulse)
# ────────────────────────────────────────────────
try:
    popt_dV, _ = curve_fit(deltaV_vs_Cm_func, Cm_values, peak_dV_sim,
                            p0=[deltaV_theory, iclamp.dur / tau_theory_slope])
    a_fit, b_fit = popt_dV
    Cm_smooth = np.linspace(min(Cm_values)*0.8, max(Cm_values)*1.2, 100)
    dV_smooth = deltaV_vs_Cm_func(Cm_smooth, a_fit, b_fit)
except Exception as e:
    print(f"ΔV vs Cm fit failed: {e}")
    Cm_smooth = []
    dV_smooth = []

# ────────────────────────────────────────────────
# Final plots – with improved labels & comments
# ────────────────────────────────────────────────

# A. Voltage traces + exponential fits
ax_volt.set_title("Voltage Response to Current Pulse\n(exponential fits during pulse)")
ax_volt.set_xlabel("Time (ms)")
ax_volt.set_ylabel("Membrane Potential (mV)")
ax_volt.set_xlim(0, 25)
ax_volt.grid(True)
ax_volt.legend(fontsize=9, loc='lower right')

# B. Peak ΔV vs Cm – should be flat (theoretical line added) + exponential regression
ax_dV.plot(Cm_values, peak_dV_sim, 'o', color='navy', markersize=9, lw=1.5, #lw stands for line width
           label="Simulated ΔV")
ax_dV.axhline(deltaV_theory, color='red', ls='--', lw=2.2,
              label=f"Theory: ΔV = {deltaV_theory:.1f} mV (independent of Cm)")
ax_dV.plot(Cm_smooth, dV_smooth, 'g--', lw=2.0,
           label=f"Exponential fit: ΔV = {a_fit:.1f} (1 - exp(-{b_fit:.2f}/Cm))")
ax_dV.set_title("Steady-State Voltage Deflection vs Cm")
ax_dV.set_xlabel("Specific capacitance Cm (µF/cm²)")
ax_dV.set_ylabel("ΔV (mV)")
ax_dV.grid(True)
ax_dV.legend(fontsize=10)

# C. Bar plot of tested values
ax_Cm_bar.bar(range(len(Cm_values)), Cm_values, color=colors,
              tick_label=[f"{c:.1f}" for c in Cm_values])
ax_Cm_bar.set_title("Tested Cm Values")
ax_Cm_bar.set_xlabel("Condition")
ax_Cm_bar.set_ylabel("Cm (µF/cm²)")
ax_Cm_bar.grid(True, axis='y')

# D. Time constant τ vs Cm – should be linear
ax_tau.plot(Cm_values, tau_fitted, 'mo-', markersize=9, lw=1.5,
            label="Fitted τ from voltage rise")
ax_tau.plot(Cm_values, tau_theory_slope * np.array(Cm_values), 'r--', lw=2.5,
            label=f"Theory: τ = {tau_theory_slope:.3f} × Cm  ms")
ax_tau.set_title("Time Constant vs Membrane Capacitance\n(τ = Rm × Cm × 0.001)")
ax_tau.set_xlabel("Specific capacitance Cm (µF/cm²)")
ax_tau.set_ylabel("Time constant τ (ms)")
ax_tau.grid(True)
ax_tau.legend(fontsize=10)

plt.tight_layout()
plt.show()


#----------------------------------------------------------------------------------------
#                                 tESTING REPO CIMMITTING
#----------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------
#                                Testing ffrom JD's computer
#-------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------
#                                Testing from Aksay_lab's computer
#-------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------
#                                Testing from JD's computer
#-------------------------------------------------------------------------------------------

# It seems as if the repo commmtting Process is operating as intended.43ersdsdsdsdsdsdsdsdsdsdsdsdsdsdsdsdsdsdsdsdsdsdsdsdsdsdsdtree