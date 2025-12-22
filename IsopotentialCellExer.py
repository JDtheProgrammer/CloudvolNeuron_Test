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
soma(0.5).pas.e = -70  
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

# Create figure with 1 row and 2 columns (side by side)
fig, (ax_volt, ax_pulse, ax_relation) = plt.subplots(1, 3, figsize=(18, 5))

# LEFT PLOT: Voltage vs Time (superimposed)
for amp, color in zip(amps, colors):
    soma.diam = diam
    soma.cm = 1 
    iclamp.amp = amp # to introduce the amperage in the simulation
    
    # Recreate recording vectors for each simulation
    v = n.Vector().record(soma(0.5)._ref_v)
    t = n.Vector().record(n._ref_t)
    
    n.finitialize(v_rest * mV)
    n.continuerun(25 * ms)
    
    # Plot voltage on left subplot
    ax_volt.plot(list(t), list(v), color=color, linewidth=2, 
             label=f"diam={diam:.2f} µm, g={soma(0.5).pas.g:.2f} S/cm², amp={amp:.3f} nA")
    
    v_list = list(v)
    t_list = list(t)
    ax_volt.plot(t_list, v_list, color=color, linewidth=2, label=f"amp={amp:.3f} nA")

    # ✅ Find absolute voltage deflection
    
    v_max = max(v_list)
    delta_v = abs(v_max - v_rest)
    v_sim.append(delta_v)

    # Surface area in cm²
    A = math.pi * (diam*1e-4) * (L*1e-4)  # um → cm

    Rm = 1 / soma(0.5).pas.g  # Ω·cm²
    Rn = Rm / A  # Ω

    # Convert amps from nA to A
    amps_A = np.array(amps) * 1e-6  # nA → mA
    v_theoretical = amps_A * Rn  # mV

    print(f"Amperages = {amp:.3f} nA → ΔV = {delta_v:.2f} mV (Max V = {v_max:.2f} mV)")
    
ax_volt.set_xlim(0, 25)
#ax_volt.set_ylim(-75, 5000)
ax_volt.set_xlabel("Time (ms)")
ax_volt.set_ylabel("Voltage (mV)")
ax_volt.set_title("Soma Voltage vs Time")
ax_volt.legend(fontsize=8)
ax_volt.grid(True)

# ✅ RIGHT SIDE: 2×2 grid of pulse inputs (subplots)
'''# Create a 2x2 grid INSIDE the right-side subplot (ax_pulse)
# ------------------------------------------------------
# 1️⃣ ax_pulse.get_subplotspec() gets the "layout spec" of the current subplot ax_pulse
# 2️⃣ .subgridspec(2, 2, hspace=0.4, wspace=0.3)
#     subdivides ax_pulse into a 2×2 grid of smaller axes (with spacing)
#     - hspace: vertical spacing between rows
#     - wspace: horizontal spacing between columns
gs = ax_pulse.get_subplotspec().subgridspec(2, 2, hspace=0.4, wspace=0.3)

# Create a list of the 4 new subplot axes using list comprehension
# fig.add_subplot(gs[i, j]) adds a subplot at grid position (i, j)
# for i in range(2) → rows 0,1
# for j in range(2) → columns 0,1
# The result: pulse_axes = [ax(0,0), ax(0,1), ax(1,0), ax(1,1)]
pulse_axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

# Loop through all current amperages and colors
# enumerate gives both index (idx = 0..3) and value (amp)

for idx, (amp, color) in enumerate(zip(amps, colors)):
    # Create a 1D array of time values from 0 to 25 ms with 500 points
    t_array = np.linspace(0, 25, 500)
     # Create a square pulse (unit step function)
    # np.where(condition, value_if_true, value_if_false)
    # → For time between iclamp.delay and iclamp.delay + iclamp.dur:
    #     pulse = amp
    #   otherwise:
    #     pulse = 0
    pulse = np.where((t_array >= iclamp.delay) & (t_array < iclamp.delay + iclamp.dur), amp, 0)

    # Plot this pulse in its corresponding mini subplot
    pulse_axes[idx].plot(t_array, pulse, color=color, linewidth=2)

    # Add a small title showing the amplitude
    pulse_axes[idx].set_title(f"amp={amp:.3f} nA", fontsize=8)

    # Set y-axis limit so all pulses fit nicely (20% padding above max)
    pulse_axes[idx].set_ylim(0, max(amps) * 1.2)

# Turn on grid lines for better readability
    pulse_axes[idx].grid(True)

    # Add axis labels selectively:
    # bottom row subplots (idx 2 or 3) get x-label
    if idx >= 2:
        pulse_axes[idx].set_xlabel("Time (ms)", fontsize=8)

    # left column subplots (idx 0 or 2) get y-label
    if idx % 2 == 0:
        pulse_axes[idx].set_ylabel("Current (nA)", fontsize=8)

# Add a big shared title for the entire figure
fig.suptitle("Voltage Response and Input Pulses", fontsize=14)

# Adjust subplot layout to prevent overlaps
plt.tight_layout()

# Display the full figure (left = voltages, right = 4 pulse plots)
plt.show(block=False)

# RIGHT PLOT: Unit Step Functions (2x2 grid)
# Create sub-axes within the right subplot
#print(soma.Ra)'''

# Right side superimposed plot of all pulses

t_array = np.linspace(0, 25, 500)
for amp, color in zip(amps, colors):
    pulse = np.where((t_array >= iclamp.delay) & (t_array < iclamp.delay + iclamp.dur), amp, 0)
    ax_pulse.plot(t_array, pulse, color=color, linewidth=2, label=f"amp={amp:.3f} nA")

ax_pulse.set_xlim(0, 25)
ax_pulse.set_ylim(0, max(amps) * 1.2)
ax_pulse.set_xlabel("Time (ms)")
ax_pulse.set_ylabel("Current (nA)")
ax_pulse.set_title("Input Current Pulses (Superimposed)")
ax_pulse.grid(True)
ax_pulse.legend(fontsize=8)


# Relationship Graph Voltage vs. Current pulse
# Prepend 0 to start from origin
x_stem = amps                   # Injected current (nA)
v_sim_stem = v_sim              # Experimental ΔV
v_theoretical_stem = list(v_theoretical)  # Theoretical ΔV
ax_relation.set_xlim(0, max(x_stem)*1.015)
ax_relation.set_ylim(0, max(v_sim_stem)*1.02)
ax_relation.plot(x_stem, v_sim_stem, 'o-', label='Experimental',
                 color='blue', linewidth=2, markersize=8)
ax_relation.plot(x_stem, v_theoretical_stem, 'r--', label='Theoretical', linewidth=2)
ax_relation.set_xlabel("Injected Current (nA)")
ax_relation.set_ylabel("Peak Voltage Difference (mV)")
ax_relation.set_title("Peak Voltage Difference vs Injected Current")
ax_relation.grid(True)
ax_relation.legend(fontsize=8)
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
ax_volt.legend(fontsize=8)
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
# Compute geometry + theory
# ---------------------------------------------------------------
L_cm = soma.L * 1e-4
diam_cm_list = [d * 1e-4 for d in diam_um_list]
A_list = [np.pi * d * L_cm for d in diam_cm_list]

Rin_list = [1/soma(0.5).pas.g / A for A in A_list]
I_amp = amp * 1e-9
deltaV_theory = [I_amp * Rin * 1000 for Rin in Rin_list]

inv_diam_list = [1/d for d in diam_um_list]
# Extend both lists with origin (0,0)
inv_diam_ext = [0] + list(inv_diam_list)
delta_v_ext  = [0] + list(delta_v_list)
Rin_ext      = [0] + list(Rin_list)

# ---------------------------------------------------------------
# ADD ORIGIN EXTENSION FOR SIMULATION + REGRESSION ONLY
# ---------------------------------------------------------------
X0 = [0] + inv_diam_list
delta_v_sim0 = [0] + delta_v_list
Rin_sim0 = [0] + Rin_list

# ---------------------------------------------------------------
# ΔV vs 1/D (Simulation + Theory)
# ---------------------------------------------------------------
ax_relation.plot(inv_diam_list, deltaV_theory, "o-", 
                 markeredgewidth = 3, linewidth=2, color=(0.0,0.8,0.0), label="Theory ∝ 1/D")
ax_relation.plot(inv_diam_list, delta_v_list, "x--",
                 markersize=15, linewidth=2, color=(0.8,0.2,0.8), label="Simulated ΔV",
                 markevery=slice(1, None))

ax_relation.set_title("Peak Voltage vs 1/Diameter")
ax_relation.set_xlabel("1 / Diameter (1/µm)")
ax_relation.set_ylabel("Peak Voltage ΔV (mV)")
ax_relation.grid(True)
ax_relation.legend(fontsize=8)
ax_relation.set_xlim(0, max(inv_diam_list)*1.05)
ax_relation.set_ylim(0, max(delta_v_list)*1.02)

# ---------------------------------------------------------------
# Rin vs 1/D (Simulation + Theory)
# ---------------------------------------------------------------
RN_theory = [Rin_list[0] * (inv_d / inv_diam_list[0]) for inv_d in inv_diam_list]

ax_RNDiamRelation.plot(inv_diam_list, RN_theory, "o-", 
                       markeredgewidth = 3, linewidth=2, color=(0.8,0.0,0.0), label="Theory ∝ 1/D")
ax_RNDiamRelation.plot(inv_diam_list, Rin_list, "x--", linewidth=2,markersize=15, color=(0.2,0.8,0.8), 
                       label="Simulated Rin",
                       markevery=slice(1, None))

ax_RNDiamRelation.set_title("Input Resistance vs 1/Diameter")
ax_RNDiamRelation.set_xlabel("1 / Diameter (1/µm)")
ax_RNDiamRelation.set_ylabel("Rin (Ω)")
ax_RNDiamRelation.grid(True)
ax_RNDiamRelation.legend(fontsize=8)
ax_RNDiamRelation.set_xlim(0, max(inv_diam_list)*1.05)
ax_RNDiamRelation.set_ylim(0, max(Rin_list)*1.02)


# ---------------------------------------------------------------
# Linear Regression (NumPy) — ΔV
# ---------------------------------------------------------------
X = np.array(inv_diam_list)
y = np.array(delta_v_list)

slope, intercept = np.polyfit(X, y, 1) # y= a_1 + a_0 -> slope = a_1, intercept = a_0
#   np.polyfit does a least-squares linear fit: y = slope * X + intercept
#   Result example: slope ≈ 5.97–6.00    intercept ≈ 0.01 (numerical noise)
y_pred = slope * X + intercept

ax_relation_reg.plot(X0, delta_v_sim0, "o-", color="blue", label="Simulated",
                     markevery=slice(1, None))
ax_relation_reg.plot([0] + list(X),
                     [intercept] + list(y_pred),
                     "r--",
                     label=f"Fit slope={slope:.2f}, \nintercept={intercept:.2e}")

ax_relation_reg.set_title("Regression: ΔV vs 1/D")
ax_relation_reg.set_xlabel("1 / Diameter (1/µm)")
ax_relation_reg.set_ylabel("ΔV (mV)")
ax_relation_reg.grid(True)
ax_relation_reg.legend(fontsize=8)

# ---------------------------------------------------------------
# Linear Regression (NumPy) — Rin
# ---------------------------------------------------------------
y_rin = np.array(Rin_list)
slope_rin, intercept_rin = np.polyfit(X, y_rin, 1)
y_rin_pred = slope_rin * X + intercept_rin

ax_RN_reg.plot(X0, Rin_sim0, "o-", color="purple", label="Simulated",
               markevery=slice(1, None))
ax_RN_reg.plot([0] + list(X),
               [intercept_rin] + list(y_rin_pred),
               "r--",
               label=f"Fit slope={slope_rin:.2e}")

ax_RN_reg.set_title("Regression: Rin vs 1/D")
ax_RN_reg.set_xlabel("1 / Diameter (1/µm)")
ax_RN_reg.set_ylabel("Rin (Ω)")
ax_RN_reg.grid(True)
ax_RN_reg.legend(fontsize=8)

plt.tight_layout()
plt.show(block=False)
#save from code
# no markers at zero for regression lines (accomplished)
# get rid of extension at origin for 2 COLUMN GRAPHS (accomplished)


#----------------------------------------------------------------------------------------
#                                 3. verying Resistivity
#----------------------------------------------------------------------------------------

print("\n3. Varying Membrane Resistivity (Rm)\n")

# Rm values in kΩ·cm² (common biological range)
Rm_values = [500, 1e3, 1.5e3, 2e3]  # 500, 1k, 1.5k, 2k Ω·cm²
colors = ["green", "blue", "red", "black"]

# Fixed soma parameters
diam = 1 * um
soma.diam = diam

# Stimulation parameters
amp = 0.075
v_rest = -70

iclamp.delay = 0
iclamp.dur = 15
iclamp.amp = amp

fig3, ax = plt.subplots(2, 3, figsize=(18, 10))

ax_volt3 = ax[0, 0 ] # left plot for voltage
ax_Rm = ax[1, 0]   # right plot for Rm bar graph
ax_relation3 = ax[0, 1 ] # bottom left for relation plot
ax_Rin = ax[1, 1 ]    # bottom right (unused)
ax_relation3_reg = ax[0, 2 ] # bottom left for regression
ax_RN_reg = ax[1, 2 ]    # bottom right for regression


# Run simulation for different resistivities
#---------------------------------------------------------------
Rin_values = []  # store Rin values for printing

delta_v_Rm = []  # store ΔV for each Rm
Rm_values_list = []  # store Rm values for x-axis

for Rm, color in zip(Rm_values, colors):

    # Convert membrane resistance Rm (Ω·cm²)
    # to passive conductance g_pas = 1/Rm (S/cm²)

    soma(0.5).pas.g = 1.0 / Rm

    # Recording vectors
    v = n.Vector().record(soma(0.5)._ref_v)
    t = n.Vector().record(n._ref_t)

    # Run the simulation
    n.finitialize(v_rest * mV)
    n.continuerun(25 * ms)

    # Extract for plotting
    v_list = list(v)
    t_list = list(t)

    # Plot voltage trace
    ax_volt3.plot(t_list, v_list, color=color, linewidth=2,
                  label=f"Rm = {Rm/1000:.1f} kΩ·cm²")

    # Compute voltage deflection
    v_max = max(v_list)
    delta_v = abs(v_max - v_rest)
    print( "check delta v", delta_v)
    
    delta_v_Rm.append(delta_v)
    Rm_values_list.append(Rm)

    print(f"Rm = {Rm/1000:.1f} kΩ·cm² → ΔV = {delta_v:.2f} mV (Max = {v_max:.2f} mV)")

# Format voltage plot
ax_volt3.set_xlim(0, 25)
ax_volt3.set_xlabel("Time (ms)")
ax_volt3.set_ylabel("Voltage (mV)")
ax_volt3.set_title("Voltage Response vs Membrane Resistivity (Rm)")
ax_volt3.legend(fontsize=8)
ax_volt3.grid(True)

# 2row-1cul: Relationship between voltage and Rm
#---------------------------------------------------------------
ax_Rm.bar(range(len(Rm_values)), # bars of range of the length of diam_values_um
    Rm_values, color=colors, tick_label=[f"{Rm:.1f}" for Rm in Rm_values])

ax_Rm.set_xlabel("Condition #")
ax_Rm.set_ylabel("Rm (Ω·cm²)")
ax_Rm.set_title("Resistivities Used in Simulation")
ax_Rm.grid(True, axis='y')
print(soma.cm)

# 1Row-2cul: ΔV vs Rm
#---------------------------------------------------------------
# compute theoretical ΔV
d_cm = soma.diam * 1e-4          # µm → cm
L_cm = soma.L * 1e-4             # µm → cm
A = math.pi * d_cm * L_cm        # cm²

I = amp * 1e-9                    # nA → A

delta_v_theory3 = []

for Rm in Rm_values:
    Rin = Rm / A                  # Ω
    dv = I * Rin * 1000           # V → mV
    delta_v_theory3.append(dv)
    Rin_values.append(Rin)

# Add origin point (0, 0) to both simulated and theoretical curves
Rm_plot_sim  = Rm_values_list
delta_v_list_sim  = delta_v_Rm

Rm_plot_theory = Rm_values
delta_v_list_theory = delta_v_theory3

# Simulated curve
ax_relation3.plot(
    Rm_plot_sim, delta_v_list_sim,
    "o-", color="blue", linewidth=2, markersize=8,
    label="Simulated"
)

# Theoretical curve
ax_relation3.plot(
    Rm_plot_theory, delta_v_list_theory,
    "r--", linewidth=2, label="Theoretical"
)

ax_relation3.set_xlabel("Membrane Resistivity (Ω·cm²)")
ax_relation3.set_ylabel("Peak Voltage ΔV (mV)")
ax_relation3.set_title("Peak Voltage vs Diameter (Sim vs Theory)")
ax_relation3.grid(True)
ax_relation3.legend(fontsize=8)
ax_relation3.set_xlim(0, max(Rm_values)*1.1)
ax_relation3.set_ylim(0, max(delta_v_Rm)*1.02)  

# 2row-2cul: Linearized Rin vs Rm (slope = 1/A)
#---------------------------------------------------------------
# Compute soma area in cm²
d_cm = soma.diam * 1e-4   # µm → cm
L_cm = soma.L * 1e-4
A = math.pi * d_cm * L_cm

# Theoretical slope
slope = 1 / A

# Plot simulated Rin vs Rm
ax_Rin.plot(
    Rm_values, Rin_values,
    "o-", color="purple", linewidth=2, markersize=8,
    label="Simulated Rin"
)

# Plot theoretical line through origin: Rin = (1/A) * Rm
Rm_theory = [0, max(Rm_values)*1.1]
Rin_theory = [0, slope*Rm_theory[1]]

ax_Rin.plot(
    Rm_theory, Rin_theory,
    "r--", linewidth=2,
    label=f"Theoretical slope = 1/A = {slope:.2e} Ω/cm²"
)

ax_Rin.set_xlabel("Membrane Resistivity Rm (Ω·cm²)")
ax_Rin.set_ylabel("Input Resistance Rin (Ω)")
ax_Rin.set_title("Input Resistance vs Membrane Resistivity (Linearized)")
ax_Rin.grid(True)
ax_Rin.legend(fontsize=8)

# 1row-3cul: Linear Regression (NumPy) — ΔV vs Rm
#---------------------------------------------------------------
X = np.array(Rm_values_list)
y = np.array(delta_v_Rm)
slope, intercept = np.polyfit(X, y, 1) # y= a_1 + a_0 -> slope = a_1, intercept = a_0
y_pred = slope * X + intercept # y_pred: this is the predicted y value based on the slope and intercept
# Plot simulated ΔV vs Rm with regression line
ax_relation3_reg.plot(
    X, y_pred,
    "g--", linewidth=2,
    label=f"Regression line: y = {slope:.2e}x + {intercept:.2e}"
)
ax_relation3_reg.set_xlabel("Membrane Resistivity (Ω·cm²)")
ax_relation3_reg.set_ylabel("Peak Voltage ΔV (mV)") 
ax_relation3_reg.set_title("Regression: Peak Voltage vs Membrane Resistivity")
ax_relation3_reg.grid(True)
ax_relation3_reg.legend(fontsize=8)
ax_relation3_reg.set_xlim(0, max(Rm_values)*1.1)
ax_relation3_reg.set_ylim(0, max(delta_v_Rm)*1.02
)

# 2row-3cul: Linear Regression (NumPy) — Rin vs Rm
#---------------------------------------------------------------
y_rin = np.array(Rin_values)
slope_rin, intercept_rin = np.polyfit(X, y_rin, 1)
y_rin_pred = slope_rin * X + intercept_rin
# Plot simulated Rin vs Rm with regression line
ax_RN_reg.plot(
    X, y_rin_pred,  # regression line
    "r--", linewidth=2, # regression line
    label=f"Regression line: y = {slope_rin:.2e}x + {intercept_rin:.2e}"
)   # regression line
ax_RN_reg.set_xlabel("Membrane Resistivity (Ω·cm²)")  # x-axis label
ax_RN_reg.set_ylabel("Input Resistance Rin (Ω)")  # y-axis label   
ax_RN_reg.set_title("Regression: Input Resistance vs Membrane Resistivity")  #  title
ax_RN_reg.grid(True)  # grid lines 
ax_RN_reg.legend(fontsize=8)  # legend
ax_RN_reg.set_xlim(0, max(Rm_values)*1.1)
ax_RN_reg.set_ylim(0, max(Rin_values)*1.1)

plt.tight_layout()
plt.show(block=False)

#----------------------------------------------------------------------------------------
#                                 4. verying Capacitance (for time constant)
#----------------------------------------------------------------------------------------
print("\n4. Varying Membrane Capacitance (Cm)\n")

# Capacitance values (µF/cm²)
Cm_values = [0.5, 1.0, 2.0, 4.0]
colors = ["green", "blue", "red", "black"]

# Fixed leak conductance → fixed Rm
soma(0.5).pas.g = 0.002        # S/cm²
g = soma(0.5).pas.g
Rm = 1 / g                     # Ω·cm²

# Prepare 2x2 subplot figure
fig4, ax = plt.subplots(2, 2, figsize=(14, 10))

ax_volt4       = ax[0, 0]   # top-left
ax_CmBar       = ax[0, 1]   # top-right
ax_relation4   = ax[1, 0]   # bottom-left
ax_tauRelation = ax[1, 1]   # bottom-right

delta_v_list = []
tau_list = []

for Cm, color in zip(Cm_values, colors):

    soma.cm = Cm    # set membrane capacitance

    # Recording vectors
    v = n.Vector().record(soma(0.5)._ref_v)
    t = n.Vector().record(n._ref_t)

    n.finitialize(v_rest * mV)
    n.continuerun(25 * ms)

    v_list = list(v)
    t_list = list(t)

    # Compute time constant τ in ms (Rm·Cm gives units of ms)
    tau = Rm * Cm 
    tau_list.append(tau)

    # Compute voltage deflection
    v_max = max(v_list)
    delta_v = abs(v_max - v_rest)
    delta_v_list.append(delta_v)

    print(f"Cm={Cm:.1f} → ΔV={delta_v:.2f} mV | τ={tau:.2f} ms")

    # Plot voltage trace with τ in label
    ax_volt4.plot(
        t_list, v_list, color=color, linewidth=2,
        label=f"Cm={Cm:.1f} µF/cm² | τ={tau:.2f} ms"
    )


# TOP-LEFT: Voltage responses
ax_volt4.set_xlim(0, 25)
ax_volt4.set_xlabel("Time (ms)")
ax_volt4.set_ylabel("Voltage (mV)")
ax_volt4.set_title("Voltage Response vs Membrane Capacitance (Cm)")
ax_volt4.legend(fontsize=8)
ax_volt4.grid(True)

# TOP-RIGHT: Bar plot of Cm values
ax_CmBar.bar(
    range(len(Cm_values)),
    Cm_values,
    color=colors,
    tick_label=[f"{Cm:.1f}" for Cm in Cm_values]
)
ax_CmBar.set_xlabel("Condition #")
ax_CmBar.set_ylabel("Membrane Capacitance (µF/cm²)")
ax_CmBar.set_title("Membrane Capacitances Used in Simulation")
ax_CmBar.grid(True, axis='y')


# Compute theoretical values

# ΔV_theory does NOT depend on Cm (Rm and area are constant)
d_cm = soma.diam * 1e-4      # µm → cm
L_cm = soma.L * 1e-4
A = math.pi * d_cm * L_cm    # cm²
I_amp = amp * 1e-9           # nA → A

Rin = Rm / A                 # constant input resistance
dV_theory_value = I_amp * Rin * 1000   # V → mV

# Same value repeated for each Cm
dV_theory_list =  [dV_theory_value for _ in Cm_values]

# Time constant τ_theory = Rm * Cm
tau_theory_list = [Rm * Cm  for Cm in Cm_values]


# BOTTOM-LEFT: Relationship between Cm and ΔV

# Experimental curve
ax_relation4.plot(
    Cm_values, delta_v_list,
    "o-", linewidth=2, markersize=8, color="blue",
    label="Experimental"
)

# Theoretical curve (horizontal line)
ax_relation4.plot(
    Cm_values, dV_theory_list,
    "r--", linewidth=2,
    label="Theoretical (constant ΔV)"
)

ax_relation4.set_xlabel("Membrane Capacitance (µF/cm²)")
ax_relation4.set_ylabel("Peak Voltage ΔV (mV)")
ax_relation4.set_title("Peak Voltage vs Membrane Capacitance")
ax_relation4.grid(True)
ax_relation4.legend(fontsize=8)
ax_relation4.set_xlim(0, max(Cm_values)*1.1)
ax_relation4.set_ylim(0, max(delta_v_list)*1.02)


# BOTTOM-RIGHT: Cm vs τ relationship

tau_plot = tau_list   # experimental τ

# Experimental τ
ax_tauRelation.plot(
    Cm_values, tau_plot,
    "o-", linewidth=2, markersize=8, color="purple",
    label="Experimental"
)

# Theoretical τ = Rm * Cm
ax_tauRelation.plot(
    Cm_values, tau_theory_list,
    "r--", linewidth=2,
    label="Theoretical τ = Rm·Cm"
)

ax_tauRelation.set_xlabel("Membrane Capacitance (µF/cm²)")
ax_tauRelation.set_ylabel("Time Constant τ (ms)")
ax_tauRelation.set_title("Membrane Time Constant vs Capacitance")
ax_tauRelation.grid(True)
ax_tauRelation.legend(fontsize=8)
ax_tauRelation.set_xlim(0, max(Cm_values)*1.1)
ax_tauRelation.set_ylim(0, max(tau_list)*1.02)

plt.tight_layout()
plt.show()

helllo
oedhfgehrfiogher