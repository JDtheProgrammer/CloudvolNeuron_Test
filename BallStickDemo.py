import neuron 
print(neuron.__version__)
from neuron import n 
from neuron.units import ms, mV, µm 
import matplotlib.pyplot as plt
from neuron import gui # for the NEURON GUI
import os
import sys
# ensure the script directory is on sys.path so local modules (MatplotlibVis.py) can be imported
#ys.path.insert(0, os.path.dirname(__file__) or '.')
#rom MatplotlibVis import NeuronMatplotlibVisualizer

n.load_file("stdrun.hoc")
'A ball-and-stick cell by definition consists of two parts: the soma (ball) '
'and a dendrite (stick). We could define two Sections at the top level as in '
'the previous tutorial, but that wouldn’t give us an easy way to create multiple '
'cells. Instead, let’s define a BallAndStick neuron class. The basic boilerplate '
'for defining any class in Python looks like:'

class BallAndStick:
    def __init__(self):
        self.soma = n.Section("soma", self)
        self.dend = n.Section("dend", self)
'Any variables that describe properties of the cell must get stored as attributes '
'of self. This is why we write self.soma instead of soma. Temporary variables,'
' on the other hand, need not be prefixed with self and will simply stop existing '
'when the initialization function ends.'
'You will also note that we have introduced a new keyword argument for Section, '
'namely the cell attribute. This will always be set to self (i.e. the current cell) '
'to allow each Section to know what cell it belongs to.'
'Recall that we can check the topology via:'

# n.topology() = the way different sections of a neuron (such as soma, dendrites, axon)'
' are connected to each other—essentially, the branching structure or connectivity'
' of the neuron model.'


'''class BallAndStick:
    def __init__(self, gid):
        self._gid = gid # global identifier
        self.soma = n.Section("soma", self) # create a soma section
        self.dend = n.Section("dend", self) # create a dendrite section
        self.dend.connect(self.soma)   # connect the dendrite to the soma
        # (the dendrite begins where the soma ends. We can see that in the topology:)

    def __repr__(self):
        return "BallAndStick[{}]".format(self._gid) # string representation'''

class BallAndStick:
    def __init__(self, gid):
        self._gid = gid
        self.setup_morphology()
        self.setup_biophysics()

    def setup_morphology(self):
        self.soma = n.Section("soma", self)
        self.dend = n.Section("dend", self)
        self.dend.connect(self.soma)
        self.all = self.soma.wholetree()  # <-- NEW wholetree is method of a Section returns a list of all 
        # the sections it is attached to – i.e. the corresponding neuron. 
        self.soma.L = self.soma.diam = 12.6157 * µm
        self.dend.L = 200 * µm
        self.dend.diam = 1 * µm
    
    def setup_biophysics(self): # Biophysical properties
        for sec in self.all:  # <-- NEW self.all is a list of all sections in the cell.
            sec.Ra = 100  # Axial resistance in Ohm * cm                    # <-- NEW
            sec.cm = 1  # Membrane capacitance in micro Farads / cm^2     # <-- NEW
            
            self.soma.insert('hh')  # insert Hodgkin-Huxley kinetics in soma
        for seg in self.soma: # <-- NEW 
            seg.hh.gnabar = (0.12)  # Sodium conductance in S/cm^2
            seg.hh.gkbar = (0.036)  # Potassium conductance in S/cm^2
            seg.hh.gl = (0.0003)    # Leak conductance in S/cm^2
            seg.hh.el = (-54.3)     # Reversal potential in mV
            
     # Insert passive current in the dendrite                       # <-- NEW
        self.dend.insert(n.pas)  # <-- NEW 
        for seg in self.dend:  # <-- NEW
            seg.pas.g = 0.001  # Passive conductance in S/cm2        # <-- NEW
            seg.pas.e = -65 * mV  # Leak reversal potential             # <-- NEW

        print('Checking for units: ',n.units("gnabar_hh")) # check units of gnabar_hh
    
        for sec in n.allsec():
            print("%s: %s" % (sec, ", ".join(sec.psection()["density_mechs"].keys())))
        'psection: it’s great for quickly getting information when interactively exploring a model because '
        'it provides a lot of data in one pass; for the same reason, however, other solutions that only '
        'extract specific desired information are more efficient for automatic explorations.)'

    def __repr__(self):
        return "BallAndStick[{}]".format(self._gid) # string representation

    '''
       __init__: initializer / constructor
       __repr__: string representation
        gid: global identifier
        class BallAndStick: defines a ball-and-stick neuron model with a soma 
        and dendrite.
        '''
    
my_cell = BallAndStick(0) # create an instance of BallAndStick with gid 0
'my_other_cell = BallAndStick(1)' # create another instance with gid 1
'del my_other_cell' # delete the second instance

'n.topology()' # output = |-|       soma(0-1)

'print(my_cell.soma(0.5).area())' # The 0.5 in my_cell.soma(0.5) is a normalized position 
#NEURON represents a section as one or more segments (controlled by nseg). 
# By default nseg = 1, so the single segment at position 0.5 represents the entire 
# section. Calling .area() on that segment returns the area for that segment 
# (and since there's one segment, that's the whole section area).
# If you set soma.nseg > 1, each segment covers only part of the section and 
# soma(0.5).area() will return the area of that midpoint segment (roughly 
# section_area / nseg).
# print(my_cell.soma().area()) prints the total area of the soma section. if some(0)=0
# soma(1)= 0

'''
12.6157 * µm: that number was chosen for the soma length and diameter: it is 
because it makes the surface area (which doesn’t include end faces) approximately
 500 μm2:'''

'''That is, if we’re only interested in electrophysiology modeling, we can 
substitute a cylindrical soma with equal length and diameter for a spherical soma
 with the same diameter, as we’ve done here. (The volume, however, is of course 
 different. So this substitution does not work if we’re modeling diffusion or 
 accumulation of ions.)'''

#n.PlotShape(False).plot(plt) # plot the cell shape without diameters
#ps = n.PlotShape(True) # plot the cell shape with diameters
#ps.plot(plt)
#ps.show(0)

'''We can also make an interactive shape plot in a separate window using 
NEURON’s built-in graphics, via:'from neuron import gui' 
(THIS DOES NOT SEEM TO OPERATE PROPERLY DUE TO WSL)'''

'''Either way, you will notice that this looks like a line instead of a ball 
and stick. Why? It’s because NEURON by default does not display diameters. 
This behavior is useful when we need to see the structure of small dendrites,
 and in NEURON 7.7, it’s the only supported option for Jupyter notebooks with 
 n.PlotShape'''

# robust wait: prefer user input, but fall back to a timed wait if stdin is closed
'''import time, sys
def wait_for_exit(timeout=300):
    try:
        input("If you see the NEURON window, press Enter here to exit...\n")
    except EOFError:
        # no stdin (IDE/non-interactive); wait until timeout or until process is killed
        print(f"stdin not available — waiting {timeout} seconds for manual close (Ctrl-C to abort).")
        try:
            t0 = time.time()
            while time.time() - t0 < timeout:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass

wait_for_exit(timeout=300)'''
# -----------------------------------------------------------------------------
# Simulation: 
# -----------------------------------------------------------------------------

# stimulus
stim = n.IClamp(my_cell.dend(1)) # create a current clamp at the end of the dendrite
stim.get_segment()
print(", ".join(item for item in dir(stim) if not item.startswith("__"))) # print attributes of stim
stim.delay = 5 * ms # delay before the stimulus starts
stim.dur = 1 * ms # duration of the stimulus
stim.amp = 0.1 # amplitude of the stimulus in nA

# recording 
soma_v = n.Vector().record(my_cell.soma(0.5)._ref_v) # record voltage at the middle of the soma
t = n.Vector().record(n._ref_t) # record time
dend_v = n.Vector().record(my_cell.dend(0.5)._ref_v) # record voltage at the middle of the dendrite

# run simulation

n.finitialize(-65 * mV)
n.continuerun(25 * mV)

# ---------------------------------------------------------------------
# Visualize
# ---------------------------------------------------------------------

'''t_values = list(t) # or
v_values = list(soma_v) # convert the NEURON Vectors to standard Python lists 
# for compatibility with Matplotlib
d_values = list(dend_v) # ^^^^^^
plt.figure(figsize=(8, 4))
plt.plot(t, soma_v, color='blue')
plt.title("Potential vs Time at Soma")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.grid(True)
plt.show()

amps = [0.075 * i for i in range(1, 5)]
colors = ["green", "blue", "red", "black"]

plt.figure(figsize=(8, 4))

for amp, color in zip(amps, colors):
    stim.amp = amp
    soma_v = n.Vector().record(my_cell.soma(0.5)._ref_v)
    t = n.Vector().record(n._ref_t)
    n.finitialize(-65 * mV)
    n.continuerun(25 * ms)
    plt.plot(list(t), list(soma_v), color=color, linewidth=2, label=f"amp={amp:.3f} nA")
    plt.plot(list(t), list(dend_v), linewidth=2, linestyle='--', color=color)
    # plt.plot(t.to_python(), soma_v.to_python(), color=color, linewidth=2, label=f"amp={amp:.3f} nA")
    #^: The to_python() method converts NEURON Vectors to standard Python lists for 
    # compatibility with Matplotlib. 
plt.xlim(80,130)
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Soma Voltage for Different Stimulus Amplitudes")
plt.legend()
plt.grid(True)
plt.show()'''

# high resolution nseg = 101 case
'''
amps = [0.075 * i for i in range(1, 5)]
colors = ["green", "blue", "red", "black"]

plt.figure(figsize=(12, 8))

for amp, color in zip(amps, colors):
    stim.amp = amp
    print(amp)
    for nsegValue, width in [(1, 2), (101, 1)]: # nseg 1 width 2, nseg 101 width 1.
        my_cell.dend.nseg = nsegValue # 
        n.finitialize(-65)
        n.continuerun(25)
        label = f"amp={amp:.3f} nA" if nsegValue == 1 else None # only label nseg=1 curves for readability
        plt.plot(list(t), list(soma_v), linewidth=width, color=color, label=label)
        plt.plot(list(t), list(dend_v), linewidth=2, linestyle='--', color=color)
    # plt.plot(t.to_python(), soma_v.to_python(), color=color, linewidth=2, label=f"amp={amp:.3f} nA")
    #^: The to_python() method converts NEURON Vectors to standard Python lists for 
    # compatibility with Matplotlib. 
plt.xlim(0,25)
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Soma Voltage for Different Stimulus Amplitudes")
plt.legend()
plt.grid(True)
plt.show()

2 soma curves (thick solid for nseg=1, thin solid for nseg=101)
2 dendrite curves (thick dashed for nseg=1, thin dashed for nseg=101)
Total: 4 colors × 4 curves/color = 16 curves
'''
# ----------------------------------------------------------------------------
#                                Exercise
# -----------------------------------------------------------------------------
'''Modify the above example, increasing the low resolution case from nseg=1 to find a low value of nseg that gives 
negligible error relative to the high-resolution nseg=101 case. Hint: a non-quantitative way to proceed would be to 
find values such that all the thin and thick curves of a given color overlap. (Note that since we’re plotting from 
the center of the dendrite, nseg should be odd, otherwise the center will fall on the segment boundaries and be 
ill-defined.)'''
amps = [0.075 * i for i in range(1, 5)]
colors = ["green", "blue", "red", "black"]

plt.figure(figsize=(12, 8))

for amp, color in zip(amps, colors):
    stim.amp = amp
    print(amp)
    # for j in range(1, 101)
    for nsegValue, width in [(1, 2), (101, 1)]: # nseg 1 width 2, nseg 101 width 1.
        my_cell.dend.nseg = nsegValue # 
        n.finitialize(-65)
        n.continuerun(25)
        label = f"amp={amp:.3f} nA" if nsegValue == 1 else None # only label nseg=1 curves for readability
        plt.plot(list(t), list(soma_v), linewidth=width, color=color, label=label)
        plt.plot(list(t), list(dend_v), linewidth=width, linestyle='--', color=color)
    # plt.plot(t.to_python(), soma_v.to_python(), color=color, linewidth=2, label=f"amp={amp:.3f} nA")
    #^: The to_python() method converts NEURON Vectors to standard Python lists for 
    # compatibility with Matplotlib. 
plt.xlim(0,25)
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Soma Voltage for Different Stimulus Amplitudes")
plt.legend()
plt.grid(True)
plt.show()


#ps = n.PlotShape(True)  # NEURON’s own 3D shape window

'viz = NeuronMatplotlibVisualizer()'
'viz.wrap_and_show(ps)   # prepare combined visualization'

# show both GUIs
'viz.show_both(ps, 0)'

# keep process alive until user closes GUI
'''print("\nPress Ctrl-C or close the windows to end.")
try:
    import time
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    print("\nExiting...")'''