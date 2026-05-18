"""
BallStickDemo.py
================

Purpose
-------
Build a minimal "ball-and-stick" neuron in NEURON (soma + single dendrite),
inject a current pulse into the distal end of the dendrite, and record the
resulting membrane potential at the centre of the soma and of the dendrite.
The script then re-runs the simulation across a small grid of stimulus
amplitudes and dendritic spatial discretisations (``nseg``) to illustrate
how spatial resolution affects the simulated voltage trace.

Inputs
------
None as files. Inputs are all hard-coded simulation parameters:
    * Morphology: soma L = diam = 12.6157 um (surface area ~= 500 um^2),
      dendrite L = 200 um, dendrite diam = 1 um.
    * Biophysics: Ra = 100 Ohm*cm; cm = 1 uF/cm^2; soma uses Hodgkin-Huxley
      (gnabar = 0.12 S/cm^2, gkbar = 0.036 S/cm^2, gl = 3e-4 S/cm^2,
      el = -54.3 mV); dendrite is passive (g = 1e-3 S/cm^2, e = -65 mV).
    * Stimulus: IClamp at dend(1.0), delay = 5 ms, dur = 1 ms,
      amp swept over [0.075, 0.150, 0.225, 0.300] nA.
    * Integration: finitialize(-65 mV), continuerun(25 ms).

Outputs
-------
Console only by default (this version has interactive plot blocks commented
out). When the plotting block is enabled the script emits a matplotlib figure
showing soma (solid) and dendrite (dashed) voltage traces for each stimulus
amplitude and each ``nseg`` setting (1 vs 101). The figure is rendered to
the screen via ``plt.show()`` and is not automatically saved.

How the script works
--------------------
1. Load NEURON and the standard run library (``stdrun.hoc``).
2. Define a ``BallAndStick`` class whose constructor builds morphology and
   biophysics in two helper methods.
3. Instantiate one cell.
4. Attach an ``IClamp`` point process to the distal end of the dendrite.
5. Record time and voltage traces with ``n.Vector().record(...)``.
6. Initialise at -65 mV and integrate to 25 ms.
7. Loop over (amplitude, nseg) pairs to study spatial discretisation
   sensitivity; replot on a single matplotlib axes.

Required modules
----------------
    neuron (>= 8.x, exposing the new ``n`` interface and ``neuron.units``),
    neuron.gui (for the optional NEURON shape window),
    matplotlib (>= 3.x),
    Python stdlib: os, sys.

Author / Project: CloudvolNeuron_Test
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import sys

import matplotlib.pyplot as plt
import neuron
from neuron import gui  # registers NEURON's native GUI (Shape windows, etc.)
from neuron import n
from neuron.units import ms, mV, um  # unit-aware constants: 1 * ms == 1.0 ms

print("NEURON version:", neuron.__version__)

# Ensure the script directory is on sys.path so local helpers (e.g.
# MatplotlibVis.NeuronMatplotlibVisualizer) can be imported when this file
# is run from another working directory.
sys.path.insert(0, os.path.dirname(__file__) or ".")
# from MatplotlibVis import NeuronMatplotlibVisualizer  # optional 3D viewer

# NEURON's standard-run library provides finitialize / continuerun and a
# default fixed-step integrator. Required before any of those calls.
n.load_file("stdrun.hoc")


# -----------------------------------------------------------------------------
# Cell definition
# -----------------------------------------------------------------------------
class BallAndStick:
    """A two-compartment ball-and-stick neuron.

    The neuron consists of a cylindrical "soma" (length = diameter so its
    lateral surface area approximates that of a sphere of the same diameter)
    and a thin dendrite attached at soma(1). Hodgkin-Huxley sodium/potassium
    kinetics are inserted in the soma so action potentials can be generated;
    the dendrite is purely passive.

    Parameters
    ----------
    gid : int
        Global identifier for the cell. Useful when many cells are created
        in a network; here we only use it for the ``__repr__``.

    Attributes
    ----------
    soma, dend : neuron.Section
        The two unbranched cables that make up the cell.
    all : list[neuron.Section]
        All sections of the cell, returned by ``soma.wholetree()``. Handy
        when iterating to set passive parameters on every compartment.
    """

    def __init__(self, gid):
        self._gid = gid
        self.setup_morphology()
        self.setup_biophysics()

    # -- morphology -----------------------------------------------------------
    def setup_morphology(self):
        """Create soma and dendrite, connect them, and set geometry.

        Geometry rationale
        ------------------
        * soma.L = soma.diam = 12.6157 um.
            A cylinder with L = d has lateral surface area pi*d*L = pi*d^2,
            which for d = 12.6157 um gives ~500 um^2 -- a value commonly
            used as an electrically equivalent stand-in for a spherical
            soma of similar diameter. This substitution is valid for
            electrophysiology but NOT for volume-dependent processes
            (ion accumulation, diffusion), since the volume differs.
        * dend.L = 200 um, dend.diam = 1 um.
            Representative of a thin distal dendritic segment; long enough
            (relative to the passive space constant lambda ~ few hundred
            um for these parameters) to show attenuation toward the tip.
        """
        self.soma = n.Section("soma", self)
        self.dend = n.Section("dend", self)
        self.dend.connect(self.soma)  # default: dend(0) attaches to soma(1)
        self.all = self.soma.wholetree()  # list of every section in this cell

        self.soma.L = self.soma.diam = 12.6157 * um
        self.dend.L = 200 * um
        self.dend.diam = 1 * um

    # -- biophysics -----------------------------------------------------------
    def setup_biophysics(self):
        """Insert passive parameters in all sections and active channels in
        the soma.

        Biophysics rationale
        --------------------
        * Ra = 100 Ohm*cm (axial / cytoplasmic resistivity). Typical value
          for mammalian neurons (range 70-200).
        * cm = 1 uF/cm^2 (specific membrane capacitance). Canonical value
          for biological membranes.
        * Soma Hodgkin-Huxley (mechanism ``hh``):
              gnabar = 0.12  S/cm^2   sodium maximal conductance
              gkbar  = 0.036 S/cm^2   potassium maximal conductance
              gl     = 3e-4  S/cm^2   leak conductance
              el     = -54.3 mV       leak reversal
          These are the classic squid-giant-axon values from Hodgkin &
          Huxley (1952) and are the defaults shipped with NEURON.
        * Dendrite passive (mechanism ``pas``):
              g = 1e-3 S/cm^2  ->  Rm = 1000 Ohm*cm^2
              e = -65 mV         leak reversal (== resting potential)
        """
        # Apply globally to every section in the cell.
        for sec in self.all:
            sec.Ra = 100    # Ohm * cm
            sec.cm = 1      # uF / cm^2

        # Insert active Hodgkin-Huxley channels into the soma only.
        self.soma.insert("hh")
        for seg in self.soma:
            seg.hh.gnabar = 0.12    # S/cm^2
            seg.hh.gkbar = 0.036    # S/cm^2
            seg.hh.gl = 0.0003      # S/cm^2
            seg.hh.el = -54.3       # mV

        # Insert passive current in the dendrite only.
        self.dend.insert(n.pas)
        for seg in self.dend:
            seg.pas.g = 0.001       # S/cm^2  (Rm = 1000 Ohm*cm^2)
            seg.pas.e = -65 * mV    # mV      (leak reversal == V_rest)

        # Diagnostic: print the units NEURON ascribes to gnabar and the
        # density mechanisms now present in each section.
        print("Units for gnabar_hh:", n.units("gnabar_hh"))
        for sec in n.allsec():
            mechs = ", ".join(sec.psection()["density_mechs"].keys())
            print(f"{sec}: {mechs}")

    def __repr__(self):
        return f"BallAndStick[{self._gid}]"


# -----------------------------------------------------------------------------
# Instantiate the cell
# -----------------------------------------------------------------------------
my_cell = BallAndStick(0)

# -----------------------------------------------------------------------------
# Stimulus and recording
# -----------------------------------------------------------------------------
# IClamp is a NEURON point process delivering a rectangular current pulse to
# the segment it is attached to. We place it at dend(1.0), i.e. the distal
# tip of the dendrite, to study how stimuli applied far from the soma
# propagate back to the soma.
stim = n.IClamp(my_cell.dend(1))
stim.delay = 5 * ms   # ms  - latency before the pulse turns on
stim.dur   = 1 * ms   # ms  - pulse duration
stim.amp   = 0.1      # nA  - pulse amplitude (overwritten in the sweep below)

# Vectors record the named reference variables on every integration step.
soma_v = n.Vector().record(my_cell.soma(0.5)._ref_v)  # mV at soma midpoint
dend_v = n.Vector().record(my_cell.dend(0.5)._ref_v)  # mV at dendrite midpoint
t      = n.Vector().record(n._ref_t)                  # ms simulation time

# -----------------------------------------------------------------------------
# Run a baseline simulation
# -----------------------------------------------------------------------------
n.finitialize(-65 * mV)
n.continuerun(25 * ms)  # integrate for 25 ms of simulated time

# -----------------------------------------------------------------------------
# Parameter sweep: stimulus amplitude x dendritic spatial discretisation
# -----------------------------------------------------------------------------
# Exercise (from the NEURON tutorial):
#   Sweep the stimulus amplitude and re-run with two different values of
#   ``nseg`` on the dendrite (low-resolution baseline = 1, high-resolution
#   reference = 101). Plot soma (solid) and dendrite (dashed) traces so
#   the user can judge by eye when low-resolution and high-resolution
#   curves overlap, indicating that nseg is high enough.
#
#   Because we record at the dendrite midpoint (x = 0.5), nseg should be
#   ODD; otherwise the midpoint coincides with a segment boundary and the
#   recorded voltage is not uniquely defined.
amps = [0.075 * i for i in range(1, 5)]   # nA: [0.075, 0.150, 0.225, 0.300]
colors = ["green", "blue", "red", "black"]

plt.figure(figsize=(12, 8))

for amp, color in zip(amps, colors):
    stim.amp = amp
    print(f"sweep amp = {amp:.3f} nA")
    for nseg_value, width in [(1, 2), (101, 1)]:
        my_cell.dend.nseg = nseg_value
        n.finitialize(-65 * mV)
        n.continuerun(25 * ms)
        # Only label the low-resolution curves so the legend stays compact.
        label = f"amp={amp:.3f} nA" if nseg_value == 1 else None
        plt.plot(list(t), list(soma_v), linewidth=width, color=color, label=label)
        plt.plot(list(t), list(dend_v), linewidth=width, linestyle="--", color=color)

plt.xlim(0, 25)
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Soma (solid) and dendrite (dashed) voltage for amp x nseg sweep")
plt.legend()
plt.grid(True)
plt.show()
