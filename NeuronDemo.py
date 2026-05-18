"""
NeuronDemo.py
=============

Purpose
-------
Introductory NEURON walkthrough: build a single-section "soma-only" neuron,
inject a sustained current via an ``IClamp``, run a 40 ms simulation, and
demonstrate how to record and persist the membrane potential trace using
several file formats (CSV / JSON / Python pickle).

This script accompanies the early sections of the NEURON Python tutorial.
Most of the plotting / persistence demonstrations are present as commented
blocks so that the file remains a self-contained reference for the user to
selectively enable.

Inputs
------
None as files. All parameters are hard-coded:
    * Morphology: soma.L = soma.diam = 20 um.
    * Stimulus  : IClamp at soma(0.5), delay = 0 ms, dur = 15 ms,
                  amp = 0.9 nA (large amplitude so the cell spikes).
    * Integration: finitialize(-65 mV), continuerun(40 ms).

The optional CSV / JSON / pickle read demos accept the corresponding files
(``data.csv`` / ``data.json`` / ``data.p``) produced by the matching write
demos earlier in the same script.

Outputs
-------
* Console: NEURON version, the ``soma.psection()`` dictionary (pretty-
  printed) and the soma length / diameter to confirm geometry settings.
* When the matplotlib block is enabled, a figure of v(t) at soma(0.5).
* When the persistence blocks are enabled:
      data.csv  -- two columns (t [ms], v [mV]); no header.
      data.json -- dict {"t": [...], "v": [...]} with indent = 4.
      data.p    -- pickle of {"t": Vector, "v": Vector}; preserves the
                   NEURON Vector type on reload.

How the script works
--------------------
1. Build a single section ``soma`` with L = diam = 20 um.
2. Print its ``psection()`` dictionary to confirm parameters.
3. Attach an IClamp at soma(0.5) with a 15 ms / 0.9 nA pulse starting at t = 0.
4. Record references to soma(0.5).v and the global simulation time t.
5. Load ``stdrun.hoc``, ``finitialize`` at -65 mV, ``continuerun`` to 40 ms.
6. Optionally serialise (t, v) to CSV / JSON / pickle and plot the loaded
   data with matplotlib or plotnine.

Required modules
----------------
    neuron (with the ``n`` interface and ``neuron.units``),
    matplotlib, pandas, plotnine, numpy,
    Python stdlib: csv, json, pickle, pprint, textwrap, math.

Author / Project: CloudvolNeuron_Test
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import csv
import json
import math
import pickle
import pprint
import textwrap
from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9

import neuron
from neuron import h          # legacy/hoc-style interface (used here for stdrun)
from neuron import n          # modern python-first interface
from neuron.units import ms, mV, um

print("NEURON version:", neuron.__version__)

# -----------------------------------------------------------------------------
# Step 1 - build a single-compartment cell
# -----------------------------------------------------------------------------
# A NEURON Section is the basic morphological building block. For a soma-only
# model it represents an unbranched cylinder approximating a roughly
# spherical cell body.
soma = n.Section("soma")

# n.topology() prints how sections are connected. With only one section the
# output is "|-|       soma(0-1)".
n.topology()

# psection() returns a nested dict of every property NEURON tracks for the
# section (morphology, mechanisms, ion species, etc.).
pprint.pprint(soma.psection())
print()

# Set geometry. With L = diam = 20 um the cylinder has surface area
# pi * d * L = pi * 20 * 20 ~ 1257 um^2, which roughly matches a small
# spherical soma of the same diameter for electrical-only modelling.
soma.L = 20 * um
soma.diam = 20 * um
print("L =", soma.psection()["morphology"]["L"], "um")
print("diam =", soma.psection()["morphology"]["diam"], "um")
print()

# -----------------------------------------------------------------------------
# Step 2 - attach a current clamp
# -----------------------------------------------------------------------------
# IClamp is a "point process": a localised current source attached to a
# specific segment. Here it sits at the centre of the soma.
iclamp = n.IClamp(soma(0.5))

# Pulse parameters. The amplitude is deliberately large so a passive soma
# will charge well above rest within 15 ms.
iclamp.delay = 0       # ms  - turn on immediately at t = 0
iclamp.dur   = 15      # ms  - pulse duration
iclamp.amp   = 0.9     # nA  - pulse amplitude

# Echo psection() now that the point process is attached -- the dictionary
# will list iclamp under "point_processes".
print(soma.psection())
print()

# -----------------------------------------------------------------------------
# Step 3 - set up recording vectors
# -----------------------------------------------------------------------------
# NEURON's Vector.record() takes a *reference* to a variable (prefixed _ref_)
# and stores its value on every integration step.
v = n.Vector().record(soma(0.5)._ref_v)   # mV at soma midpoint
t = n.Vector().record(n._ref_t)           # ms global simulation time

# -----------------------------------------------------------------------------
# Step 4 - run the simulation
# -----------------------------------------------------------------------------
# stdrun.hoc supplies high-level control routines (fadvance, finitialize,
# continuerun) used below. Load it before calling them.
n.load_file("stdrun.hoc")
n.finitialize(-65 * mV)     # set every state variable to its resting value
n.continuerun(40 * ms)      # integrate forward to t = 40 ms

# -----------------------------------------------------------------------------
# Optional: plot v(t)
# -----------------------------------------------------------------------------
# Uncomment to render an interactive matplotlib window of the trace.
# plt.figure()
# plt.plot(list(t), list(v))
# plt.xlabel("t (ms)")
# plt.ylabel("v (mV)")
# plt.show()

# -----------------------------------------------------------------------------
# Optional: persist the trace
# -----------------------------------------------------------------------------
# (A) CSV -- portable, human-readable, but loses NEURON's Vector type.
#     File format: rows of "t,v" with no header.
#     To enable, uncomment below:
# with open("data.csv", "w") as f:
#     csv.writer(f).writerows(zip(t, v))
# with open("data.csv") as f:
#     reader = csv.reader(f)
#     tnew, vnew = zip(*[[float(val) for val in row] for row in reader if row])
#
# (B) JSON -- language-independent; Vectors must be converted to list first.
#     To enable, uncomment below:
# with open("data.json", "w") as f:
#     json.dump({"t": list(t), "v": list(v)}, f, indent=4)
# with open("data.json") as f:
#     data = json.load(f)
# tnew, vnew = data["t"], data["v"]
#
# (C) Pickle -- Python-specific; preserves NEURON Vector objects on reload.
#     To enable, uncomment below:
# with open("data.p", "wb") as f:
#     pickle.dump({"t": t, "v": v}, f)
# with open("data.p", "rb") as f:
#     data = pickle.load(f)
# tnewp, vnewp = data["t"], data["v"]
# print(type(tnewp), tnewp.hname())  # hoc.Vector  -- Vector[<id>]
