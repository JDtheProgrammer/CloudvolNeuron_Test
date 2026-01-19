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
from neuron import h

print('\n NEURON Version:\n', neuron.__version__)

from neuron import n 
from neuron.units import ms, mV, um

print('\nNEURON Version:', neuron.__version__)
print('\n' + '='*80)
print('NON-ISOPOTENTIAL CABLE THEORY ANALYSIS')
print('='*80)

# Create cable section
cable = h.Section(name='cable')
h.load_file("stdrun.hoc") # loads standard run library (Hoc (pronounced "hoak") is NEURON's original,
#C-like interpreted scripting language for defining neuron models, controlling simulations, 
# and creating graphical interfaces)

# Current clamp at center
iclamp = h.IClamp(cable(0.5))
iclamp.delay = 0
iclamp.dur = 15
iclamp.amp = 0.1  # nA

# Insert passive properties
cable.insert('pas')

# Fixed parameters (will vary these one at a time)
v_rest = -70 * mV # mV
cable.L = 1000 * um  # μm (1 mm cable)
cable.diam = 1 * um  # μm
cable.cm = 1.0  # μF/cm²
cable(0.5).pas.g = 0.001  # S/cm² → Rm = 1000 Ω·cm²
cable(0.5).pas.e = v_rest
cable.Ra = 100  # Ω·cm (axial resistivity)

print(f'\nDefault Cable Parameters:')
print(f'  Length L = {cable.L} μm')
print(f'  Diameter d = {cable.diam} μm')
print(f'  Membrane capacitance Cm = {cable.cm} μF/cm²')
print(f'  Membrane conductance g = {cable(0.5).pas.g} S/cm² (Rm = {1/cable(0.5).pas.g} Ω·cm²)')
print(f'  Axial resistivity Ra = {cable.Ra} Ω·cm')

# Calculate space constant λ
def calculate_lambda(diam, Rm, Ra):
    """
    Space constant: λ = sqrt(Rm·d / (4·Ra))
    where d is diameter in cm, Rm in Ω·cm², Ra in Ω·cm
    """
    d_cm = diam * 1e-4  # μm → cm
    lambda_cm = np.sqrt(Rm * d_cm / (4 * Ra))
    return lambda_cm * 1e4  # cm → μm

# Calculate input resistance at injection point for semi-infinite cable
def calculate_Rin_cable(diam, Rm, Ra, L=None):
    """
    For semi-infinite cable: Rin = sqrt(Rm·Ra / (π·d))
    For finite cable of length L, it's more complex but approaches this for L >> λ
    """
    d_cm = diam * 1e-4  # μm → cm
    Rin = np.sqrt(Rm * Ra / (np.pi * d_cm))
    return Rin

#========================================================================================
#                    SECTION 1: VARYING CURRENT PULSE AMPLITUDE
#========================================================================================

print('\n' + '='*80)
print('SECTION 1: VARYING CURRENT PULSE AMPLITUDE')
print('='*80)

amps = [(0.075) * k for k in range(1, 5)] # nA
colors = ["green", "blue", "red", "black"]

# Calculate theoretical predictions
Rm = 1 / cable(0.5).pas.g
Ra = cable.Ra
diam = cable.diam

lambda_val = calculate_lambda(diam, Rm, Ra)
Rin_center = calculate_Rin_cable(diam, Rm, Ra)
theoretical_slope = Rin_center * 1e-6  # Ω → mV/nA conversion

print(f'\nTheoretical Predictions:')
print(f'  Space constant λ = {lambda_val:.2f} μm')
print(f'  Input resistance at center Rin = {Rin_center:.2e} Ω')
print(f'  Expected slope (ΔV vs I) = {theoretical_slope:.2e} mV/nA')

# Storage for results
delta_v_center = []
delta_v_end = []

fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
ax_volt_center = axes[0, 0]
ax_volt_end = axes[1, 0]
ax_pulse = axes[0, 1]
ax_relation_center = axes[0, 2]
ax_relation_end = axes[1, 2]
axes[1, 1].axis('off')

for amp, color in zip(amps, colors):
    iclamp.amp = amp
    
    # Record voltage at center (0.5) and end (1.0)
    v_center = h.Vector().record(cable(0.5)._ref_v)
    v_end = h.Vector().record(cable(1.0)._ref_v)
    t = h.Vector().record(h._ref_t)
    
    h.finitialize(v_rest * mV)
    h.continuerun(25 * ms)
    
    t_list = list(t)
    v_center_list = list(v_center)
    v_end_list = list(v_end)
    
    # Plot voltage traces
    ax_volt_center.plot(t_list, v_center_list, color=color, linewidth=2,
                       label=f'I={amp:.3f} nA')
    ax_volt_end.plot(t_list, v_end_list, color=color, linewidth=2,
                    label=f'I={amp:.3f} nA')
    
    # Calculate peak deflections
    dv_center = max(v_center_list) - v_rest
    dv_end = max(v_end_list) - v_rest
    delta_v_center.append(dv_center)
    delta_v_end.append(dv_end)
    
    print(f'I = {amp:.3f} nA → ΔV_center = {dv_center:.3f} mV, ΔV_end = {dv_end:.3f} mV')

# Voltage plots
ax_volt_center.set_xlabel('Time (ms)')
ax_volt_center.set_ylabel('Voltage (mV)')
ax_volt_center.set_title('Voltage at Center (injection site)')
ax_volt_center.legend()
ax_volt_center.grid(True)

ax_volt_end.set_xlabel('Time (ms)')
ax_volt_end.set_ylabel('Voltage (mV)')
ax_volt_end.set_title('Voltage at End (cable(1.0))')
ax_volt_end.legend()
ax_volt_end.grid(True)

# Current pulse visualization
t_array = np.linspace(0, 25, 500)
for amp, color in zip(amps, colors):
    pulse = np.where((t_array >= iclamp.delay) & (t_array < iclamp.delay + iclamp.dur), amp, 0)
    ax_pulse.plot(t_array, pulse, color=color, linewidth=3, label=f'{amp:.2f} nA')

ax_pulse.set_xlabel('Time (ms)')
ax_pulse.set_ylabel('Current (nA)')
ax_pulse.set_title('Input Current Pulses')
ax_pulse.legend()
ax_pulse.grid(True)

# Linear regression - Center
slope_center, intercept_center = np.polyfit(amps, delta_v_center, 1)
r2_center = np.corrcoef(amps, delta_v_center)[0, 1]**2

# Theoretical curve
amps_theory = np.linspace(0, max(amps)*1.1, 20)
dv_theory = amps_theory * theoretical_slope

# Plot - Center
ax_relation_center.plot(amps, delta_v_center, 'o', color='blue', markersize=10,
                       label='Simulated')
ax_relation_center.plot(amps_theory, dv_theory, 'x', color='red', markersize=8,
                       linewidth=2, label=f'Theory: slope={theoretical_slope:.2e} mV/nA')
x_fit = np.linspace(0, max(amps)*1.1, 100)
y_fit = slope_center * x_fit + intercept_center
ax_relation_center.plot(x_fit, y_fit, '--', color='black', linewidth=2,
                       label=f'Regression: slope={slope_center:.2e}')
ax_relation_center.set_xlabel('Current (nA)')
ax_relation_center.set_ylabel('Peak ΔV (mV)')
ax_relation_center.set_title('ΔV vs Current (Center)')
ax_relation_center.legend(fontsize=9)
ax_relation_center.grid(True)
ax_relation_center.text(0.05, 0.95, f'R² = {r2_center:.4f}',
                       transform=ax_relation_center.transAxes, fontsize=10, va='top')

# Linear regression - End
slope_end, intercept_end = np.polyfit(amps, delta_v_end, 1)
r2_end = np.corrcoef(amps, delta_v_end)[0, 1]**2

ax_relation_end.plot(amps, delta_v_end, 'o', color='purple', markersize=10,
                    label='Simulated (end)')
ax_relation_end.plot(amps_theory, dv_theory * 0.5, 'x', color='orange', markersize=8,
                    linewidth=2, label='Theory (approx)')
y_fit_end = slope_end * x_fit + intercept_end
ax_relation_end.plot(x_fit, y_fit_end, '--', color='black', linewidth=2,
                    label=f'Regression: slope={slope_end:.2e}')
ax_relation_end.set_xlabel('Current (nA)')
ax_relation_end.set_ylabel('Peak ΔV (mV)')
ax_relation_end.set_title('ΔV vs Current (End)')
ax_relation_end.legend(fontsize=9)
ax_relation_end.grid(True)
ax_relation_end.text(0.05, 0.95, f'R² = {r2_end:.4f}',
                    transform=ax_relation_end.transAxes, fontsize=10, va='top')

print(f'\nRegression Results:')
print(f'  Center: slope = {slope_center:.2e} mV/nA (theory = {theoretical_slope:.2e})')
print(f'  End: slope = {slope_end:.2e} mV/nA')
print(f'  Ratio (end/center) = {slope_end/slope_center:.3f}')

plt.tight_layout()
plt.show(block=False)

#========================================================================================
#                    SECTION 2: VARYING CABLE DIAMETER
#========================================================================================

print('\n' + '='*80)
print('SECTION 2: VARYING CABLE DIAMETER')
print('='*80)

diameters = [1, 2, 4, 8]  # μm
colors = ["green", "blue", "red", "black"]
amp = 0.1  # nA fixed
iclamp.amp = amp

# Theoretical predictions: Rin ∝ 1/sqrt(d), λ ∝ sqrt(d)
Rm = 1 / cable(0.5).pas.g
Ra = cable.Ra

delta_v_center_diam = []
delta_v_end_diam = []
lambda_values = []
Rin_theory_values = []

fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
ax_volt_center = axes[0, 0]
ax_volt_end = axes[1, 0]
ax_lambda = axes[0, 1]
ax_dv_center = axes[0, 2]
ax_dv_end = axes[1, 2]
ax_diam_bar = axes[1, 1]

for diam, color in zip(diameters, colors):
    cable.diam = diam
    
    # Calculate theoretical values
    lambda_val = calculate_lambda(diam, Rm, Ra)
    Rin_center = calculate_Rin_cable(diam, Rm, Ra)
    lambda_values.append(lambda_val)
    Rin_theory_values.append(Rin_center)
    
    v_center = h.Vector().record(cable(0.5)._ref_v)
    v_end = h.Vector().record(cable(1.0)._ref_v)
    t = h.Vector().record(h._ref_t)
    
    h.finitialize(v_rest * mV)
    h.continuerun(25 * ms)
    
    t_list = list(t)
    v_center_list = list(v_center)
    v_end_list = list(v_end)
    
    ax_volt_center.plot(t_list, v_center_list, color=color, linewidth=2,
                       label=f'd={diam} μm')
    ax_volt_end.plot(t_list, v_end_list, color=color, linewidth=2,
                    label=f'd={diam} μm')
    
    dv_center = max(v_center_list) - v_rest
    dv_end = max(v_end_list) - v_rest
    delta_v_center_diam.append(dv_center)
    delta_v_end_diam.append(dv_end)
    
    print(f'd = {diam} μm → λ = {lambda_val:.1f} μm, Rin = {Rin_center:.2e} Ω')
    print(f'           ΔV_center = {dv_center:.2f} mV, ΔV_end = {dv_end:.2f} mV')

ax_volt_center.set_xlabel('Time (ms)')
ax_volt_center.set_ylabel('Voltage (mV)')
ax_volt_center.set_title('Voltage at Center')
ax_volt_center.legend()
ax_volt_center.grid(True)

ax_volt_end.set_xlabel('Time (ms)')
ax_volt_end.set_ylabel('Voltage (mV)')
ax_volt_end.set_title('Voltage at End')
ax_volt_end.legend()
ax_volt_end.grid(True)

# Bar chart of diameters
ax_diam_bar.bar(range(len(diameters)), diameters, color=colors,
               tick_label=[f'{d}' for d in diameters])
ax_diam_bar.set_xlabel('Condition')
ax_diam_bar.set_ylabel('Diameter (μm)')
ax_diam_bar.set_title('Tested Diameters')
ax_diam_bar.grid(True, axis='y')

# Lambda vs sqrt(d)
sqrt_d = np.sqrt(diameters)
sqrt_d_theory = np.linspace(0, max(sqrt_d)*1.1, 20)
# λ = sqrt(Rm·d/(4·Ra)), so λ ∝ sqrt(d)
lambda_theory = np.sqrt(Rm / (4 * Ra) * 1e-4) * 1e4 * sqrt_d_theory

slope_lambda, _ = np.polyfit(sqrt_d, lambda_values, 1)
x_fit = np.linspace(0, max(sqrt_d)*1.1, 100)
y_fit = slope_lambda * x_fit

ax_lambda.plot(sqrt_d, lambda_values, 'o', color='green', markersize=10, label='Simulated λ')
ax_lambda.plot(sqrt_d_theory, lambda_theory, 'x', color='red', markersize=8,
              linewidth=2, label='Theory')
ax_lambda.plot(x_fit, y_fit, '--', color='black', linewidth=2,
              label=f'Regression: slope={slope_lambda:.1f}')
ax_lambda.set_xlabel('√(Diameter) (√μm)')
ax_lambda.set_ylabel('Space Constant λ (μm)')
ax_lambda.set_title('Space Constant vs √(Diameter)')
ax_lambda.legend()
ax_lambda.grid(True)

# ΔV vs 1/sqrt(d) - Center
inv_sqrt_d = 1 / np.sqrt(diameters)
inv_sqrt_d_theory = np.linspace(0, max(inv_sqrt_d)*1.1, 20)
# Rin ∝ 1/sqrt(d), so ΔV ∝ 1/sqrt(d)
dV_theory_center = amp * np.sqrt(Rm * Ra / (np.pi * 1e-4)) * 1e-6 * inv_sqrt_d_theory

slope_dv_center, _ = np.polyfit(inv_sqrt_d, delta_v_center_diam, 1)
y_fit_center = slope_dv_center * np.linspace(0, max(inv_sqrt_d)*1.1, 100)

ax_dv_center.plot(inv_sqrt_d, delta_v_center_diam, 'o', color='blue', markersize=10,
                 label='Simulated')
ax_dv_center.plot(inv_sqrt_d_theory, dV_theory_center, 'x', color='red', markersize=8,
                 linewidth=2, label='Theory')
ax_dv_center.plot(np.linspace(0, max(inv_sqrt_d)*1.1, 100), y_fit_center, '--',
                 color='black', linewidth=2, label=f'Regression: m={slope_dv_center:.2e}')
ax_dv_center.set_xlabel('1/√(Diameter) (1/√μm)')
ax_dv_center.set_ylabel('Peak ΔV (mV)')
ax_dv_center.set_title('ΔV vs 1/√d (Center)')
ax_dv_center.legend(fontsize=9)
ax_dv_center.grid(True)

# ΔV vs 1/sqrt(d) - End
slope_dv_end, _ = np.polyfit(inv_sqrt_d, delta_v_end_diam, 1)
y_fit_end = slope_dv_end * np.linspace(0, max(inv_sqrt_d)*1.1, 100)

ax_dv_end.plot(inv_sqrt_d, delta_v_end_diam, 'o', color='purple', markersize=10,
              label='Simulated (end)')
ax_dv_end.plot(np.linspace(0, max(inv_sqrt_d)*1.1, 100), y_fit_end, '--',
              color='black', linewidth=2, label=f'Regression: m={slope_dv_end:.2e}')
ax_dv_end.set_xlabel('1/√(Diameter) (1/√μm)')
ax_dv_end.set_ylabel('Peak ΔV (mV)')
ax_dv_end.set_title('ΔV vs 1/√d (End)')
ax_dv_end.legend(fontsize=9)
ax_dv_end.grid(True)

plt.tight_layout()
plt.show(block=False)

#========================================================================================
#                    SECTION 3: VARYING MEMBRANE RESISTIVITY (Rm)
#========================================================================================

print('\n' + '='*80)
print('SECTION 3: VARYING MEMBRANE RESISTIVITY')
print('='*80)

Rm_values = [500, 1000, 2000, 4000]  # Ω·cm²
colors = ["green", "blue", "red", "black"]

cable.diam = 2  # reset
amp = 0.1
iclamp.amp = amp

delta_v_center_Rm = []
delta_v_end_Rm = []
lambda_values_Rm = []

fig3, axes = plt.subplots(2, 3, figsize=(18, 10))
ax_volt_center = axes[0, 0]
ax_volt_end = axes[1, 0]
ax_Rm_bar = axes[1, 1]
ax_lambda_Rm = axes[0, 1]
ax_dv_center_Rm = axes[0, 2]
ax_dv_end_Rm = axes[1, 2]

for Rm, color in zip(Rm_values, colors):
    cable(0.5).pas.g = 1.0 / Rm
    
    lambda_val = calculate_lambda(cable.diam, Rm, cable.Ra)
    lambda_values_Rm.append(lambda_val)
    
    v_center = h.Vector().record(cable(0.5)._ref_v)
    v_end = h.Vector().record(cable(1.0)._ref_v)
    t = h.Vector().record(h._ref_t)
    
    h.finitialize(v_rest * mV)
    h.continuerun(25 * ms)
    
    t_list = list(t)
    v_center_list = list(v_center)
    v_end_list = list(v_end)
    
    ax_volt_center.plot(t_list, v_center_list, color=color, linewidth=2,
                       label=f'Rm={Rm} Ω·cm²')
    ax_volt_end.plot(t_list, v_end_list, color=color, linewidth=2,
                    label=f'Rm={Rm} Ω·cm²')
    
    dv_center = max(v_center_list) - v_rest
    dv_end = max(v_end_list) - v_rest
    delta_v_center_Rm.append(dv_center)
    delta_v_end_Rm.append(dv_end)
    
    print(f'Rm = {Rm} Ω·cm² → λ = {lambda_val:.1f} μm')
    print(f'              ΔV_center = {dv_center:.2f} mV, ΔV_end = {dv_end:.2f} mV')

ax_volt_center.set_xlabel('Time (ms)')
ax_volt_center.set_ylabel('Voltage (mV)')
ax_volt_center.set_title('Voltage at Center')
ax_volt_center.legend()
ax_volt_center.grid(True)

ax_volt_end.set_xlabel('Time (ms)')
ax_volt_end.set_ylabel('Voltage (mV)')
ax_volt_end.set_title('Voltage at End')
ax_volt_end.legend()
ax_volt_end.grid(True)

ax_Rm_bar.bar(range(len(Rm_values)), Rm_values, color=colors,
             tick_label=[f'{Rm}' for Rm in Rm_values])
ax_Rm_bar.set_xlabel('Condition')
ax_Rm_bar.set_ylabel('Rm (Ω·cm²)')
ax_Rm_bar.set_title('Tested Rm Values')
ax_Rm_bar.grid(True, axis='y')

# λ vs sqrt(Rm)
sqrt_Rm = np.sqrt(Rm_values)
sqrt_Rm_theory = np.linspace(0, max(sqrt_Rm)*1.1, 20)
lambda_theory_Rm = np.sqrt(cable.diam * 1e-4 / (4 * cable.Ra)) * 1e4 * sqrt_Rm_theory

slope_lambda_Rm, _ = np.polyfit(sqrt_Rm, lambda_values_Rm, 1)
y_fit = slope_lambda_Rm * np.linspace(0, max(sqrt_Rm)*1.1, 100)

ax_lambda_Rm.plot(sqrt_Rm, lambda_values_Rm, 'o', color='green', markersize=10,
                 label='Simulated λ')
ax_lambda_Rm.plot(sqrt_Rm_theory, lambda_theory_Rm, 'x', color='red', markersize=8,
                 linewidth=2, label='Theory')
ax_lambda_Rm.plot(np.linspace(0, max(sqrt_Rm)*1.1, 100), y_fit, '--',
                 color='black', linewidth=2, label=f'Regression: m={slope_lambda_Rm:.2f}')
ax_lambda_Rm.set_xlabel('√(Rm) (√(Ω·cm²))')
ax_lambda_Rm.set_ylabel('λ (μm)')
ax_lambda_Rm.set_title('Space Constant vs √(Rm)')
ax_lambda_Rm.legend()
ax_lambda_Rm.grid(True)

# ΔV vs sqrt(Rm) - proportional to Rin ∝ sqrt(Rm)
slope_dv_Rm, _ = np.polyfit(sqrt_Rm, delta_v_center_Rm, 1)
y_fit_center = slope_dv_Rm * np.linspace(0, max(sqrt_Rm)*1.1, 100)

ax_dv_center_Rm.plot(sqrt_Rm, delta_v_center_Rm, 'o', color='blue', markersize=10,
                    label='Simulated')
ax_dv_center_Rm.plot(np.linspace(0, max(sqrt_Rm)*1.1, 100), y_fit_center, '--',
                    color='black', linewidth=2, label=f'Regression: m={slope_dv_Rm:.2e}')
ax_dv_center_Rm.set_xlabel('√(Rm) (√(Ω·cm²))')
ax_dv_center_Rm.set_ylabel('Peak ΔV (mV)')
ax_dv_center_Rm.set_title('ΔV vs √(Rm) (Center)')
ax_dv_center_Rm.legend()
ax_dv_center_Rm.grid(True)

slope_dv_end_Rm, _ = np.polyfit(sqrt_Rm, delta_v_end_Rm, 1)
y_fit_end = slope_dv_end_Rm * np.linspace(0, max(sqrt_Rm)*1.1, 100)

ax_dv_end_Rm.plot(sqrt_Rm, delta_v_end_Rm, 'o', color='purple', markersize=10,
                 label='Simulated (end)')
ax_dv_end_Rm.plot(np.linspace(0, max(sqrt_Rm)*1.1, 100), y_fit_end, '--',
                 color='black', linewidth=2, label=f'Regression: m={slope_dv_end_Rm:.2e}')
ax_dv_end_Rm.set_xlabel('√(Rm) (√(Ω·cm²))')
ax_dv_end_Rm.set_ylabel('Peak ΔV (mV)')
ax_dv_end_Rm.set_title('ΔV vs √(Rm) (End)')
ax_dv_end_Rm.legend()
ax_dv_end_Rm.grid(True)

plt.tight_layout()
plt.show(block=False)

#========================================================================================
#                    SECTION 4: VARYING AXIAL RESISTIVITY (Ra)
#========================================================================================

print('\n' + '='*80)
print('SECTION 4: VARYING AXIAL RESISTIVITY (Ra)')
print('='*80)

Ra_values = [50, 100, 200, 400]  # Ω·cm
colors = ["green", "blue", "red", "black"]

cable.diam = 2
cable(0.5).pas.g = 0.001  # reset Rm = 1000
Rm = 1000
amp = 0.1
iclamp.amp = amp

delta_v_center_Ra = []
delta_v_end_Ra = []
lambda_values_Ra = []

fig4, axes = plt.subplots(2, 3, figsize=(18, 10))
ax_volt_center = axes[0, 0]
ax_volt_end = axes[1, 0]
ax_Ra_bar = axes[1, 1]
ax_lambda_Ra = axes[0, 1]
ax_dv_center_Ra = axes[0, 2]
ax_dv_end_Ra = axes[1, 2]

for Ra, color in zip(Ra_values, colors):
    cable.Ra = Ra
    
    lambda_val = calculate_lambda(cable.diam, Rm, Ra)
    lambda_values_Ra.append(lambda_val)
    
    v_center = h.Vector().record(cable(0.5)._ref_v)
    v_end = h.Vector().record(cable(1.0)._ref_v)
    t = h.Vector().record(h._ref_t)
    
    h.finitialize(v_rest * mV)
    h.continuerun(25 * ms)
    
    t_list = list(t)
    v_center_list = list(v_center)
    v_end_list = list(v_end)
    
    ax_volt_center.plot(t_list, v_center_list, color=color, linewidth=2,
                       label=f'Ra={Ra} Ω·cm')
    ax_volt_end.plot(t_list, v_end_list, color=color, linewidth=2,
                    label=f'Ra={Ra} Ω·cm')
    
    dv_center = max(v_center_list) - v_rest
    dv_end = max(v_end_list) - v_rest
    delta_v_center_Ra.append(dv_center)
    delta_v_end_Ra.append(dv_end)
    
    print(f'Ra = {Ra} Ω·cm → λ = {lambda_val:.1f} μm')
    print(f'             ΔV_center = {dv_center:.2f} mV, ΔV_end = {dv_end:.2f} mV')

ax_volt_center.set_xlabel('Time (ms)')
ax_volt_center.set_ylabel('Voltage (mV)')
ax_volt_center.set_title('Voltage at Center')
ax_volt_center.legend()
ax_volt_center.grid(True)

ax_volt_end.set_xlabel('Time (ms)')
ax_volt_end.set_ylabel('Voltage (mV)')
ax_volt_end.set_title('Voltage at End')
ax_volt_end.legend()
ax_volt_end.grid(True)

ax_Ra_bar
plt.tight_layout()
plt.show()