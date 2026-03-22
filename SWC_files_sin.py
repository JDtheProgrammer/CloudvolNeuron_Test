#=======================================================================================================================================================================
#          Manipulating SWC files (one file at a time)
#=======================================================================================================================================================================
from ast import main
from cloudvolume import CloudVolume
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('Agg') # Use 'Agg' backend for headless environments (no display)
import vtk
from mpl_toolkits.mplot3d import Axes3D
import neuron 
from neuron import h, gui
from neuron.units import ms, mV, um
import os # os stands for operating system, and it's a built-in Python module that provides a way to interact with the underlying operating system. It allows you to perform various tasks such as file and directory manipulation, environment variable access, and more. 
import pprint # pprint stands for "pretty-print" and is a built-in Python module that provides a way to print data structures in a more readable and organized format.
import sys  # sys is a built-in Python module that provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter. It allows you to manipulate the Python runtime environment, access command-line arguments, and perform various system-related tasks.
import pprint
from neuron import n

def enablePrint():
    sys.stdout = sys.__stdout__

def disablePrint():
    sys.stdout = open(os.devnull, 'w')

print(f'VTK version: {vtk.VTK_VERSION}')

try:
    import cloudvolume
    print("CloudVolume is installed.")
except ImportError:
    print("CloudVolume is not installed. Please install it using 'pip install cloud-volume'.")
    # You can also add code here to exit the script or attempt installation
    # import sys
    # sys.exit(1) 
h.load_file("stdrun.hoc")    
h.load_file('import3d.hoc')

# code to source only 1 SWC file.
swc_path = '/home/jd/NeuronProject/CloudvolNeuron_Test/SWC_files/76182_reRoot_reSample_5000.swc'  
print("File exists?", os.path.exists(swc_path))
print("Loading SWC file:", swc_path)

reader = h.Import3d_SWC_read() # This line creates an instance of the Import3d_SWC_read class, which is a built-in class in NEURON that is used to read SWC files. The reader object will be used to load the SWC file and extract the morphological data to create a cell model in NEURON.
reader.input(swc_path) # This line calls the input() method of the reader object, passing the path to the SWC file as an argument. The input() method reads the SWC file and prepares the data for instantiation. It processes the morphological information contained in the SWC file, such as the coordinates of the points, their types, and their connectivity, so that it can be used to create a cell model in NEURON.

importer = h.Import3d_GUI(reader, 1)
# Create a plain Python object to hold sections
class Cell:
    def __init__(self):
        self.sl = h.SectionList()   # real HOC SectionList

cell = Cell()
importer.instantiate(cell) # This line calls the instantiate() method of the importer object, passing the cell object as an argument. The instantiate() method takes the morphological data that was read from the SWC file and uses it to create a cell model in NEURON. It generates the sections and their connectivity based on the information in the SWC file and populates the cell.sl SectionList with the created sections. After this line is executed, the cell object will contain a SectionList that represents the morphology of the neuron as defined in the SWC file.
#cell.sl = h.SectionList() # This line reinitializes the cell.sl attribute to a new, empty SectionList. This is done to ensure that the SectionList is empty before populating it with the sections created from the SWC file. By creating a new SectionList, we can avoid any potential issues that might arise from having pre-existing sections in the list, and it allows us to start fresh with the sections generated from the SWC file.
'''for sec in h.allsec():
    cell.sl.append(sec)'''

# Give NEURON a chance to process the new topology
#h.define_shape() # the underscore in define_shape() is important, as it indicates that this is a method of the h object in NEURON. The define_shape() method is used to compute the 3D coordinates of the sections and their connectivity based on the morphological data that was loaded from the SWC file. It processes the sections and their connections to create a 3D representation of the neuron, which can then be visualized using NEURON's built-in graphics capabilities.
shape = h.Shape(cell.sl)
shape.exec_menu('3D Rotate')
shape.show(0)
shape.flush()
h.doNotify()


# ─────────────────────────────────────────────
# Step 1: Parse SWC — separate soma vs non-soma nodes
# ─────────────────────────────────────────────
nodes = {}
with open(swc_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('#') or line == '':
            continue
        parts     = line.split()
        node_id   = int(parts[0])
        node_type = int(parts[1])
        x         = float(parts[2])
        y         = float(parts[3])
        z         = float(parts[4])
        radius    = float(parts[5])
        parent_id = int(parts[6])
        nodes[node_id] = {
            'node_id'  : node_id,
            'type'     : node_type,
            'x'        : x, 'y': y, 'z': z,
            'radius'   : radius,
            'parent_id': parent_id
        }

# Soma = node 1 + all nodes whose parent_id == 1
soma_node_ids   = {nid for nid, d in nodes.items() if nid == 1 or d['parent_id'] == 1}
non_soma_node_ids = {nid for nid in nodes if nid not in soma_node_ids}

soma_coords     = np.array([[nodes[nid]['x'], nodes[nid]['y'], nodes[nid]['z']]
                             for nid in soma_node_ids])
non_soma_coords = np.array([[nodes[nid]['x'], nodes[nid]['y'], nodes[nid]['z']]
                             for nid in non_soma_node_ids])

print(f"Soma nodes    : {len(soma_node_ids)}")
print(f"Non-soma nodes: {len(non_soma_node_ids)}")

# ------------------------------
# Step 2: Classify sections (soma vs non-soma)
# ------------------------------

# 1️⃣ Find root section(s) — these are the soma sections
root_sections = [sec for sec in h.allsec() if sec.parentseg() is None]

# 2️⃣ Traverse the tree to mark all sections connected to root as soma
soma_sections = []
non_soma_sections = []

# For each section, we check its 3D coordinates (using the n3d points) and see if they are close to any of the soma node coordinates.
for sec in h.allsec():
    n3d = sec.n3d() # n3d() returns the number of 3D points in the section.
    if n3d == 0: # If there are no 3D points, we can't classify it based on coordinates.
        non_soma_sections.append(sec) # so we classify it as non-soma by default 
        continue # continue causes the loop to skip the rest of the code for this section and move on to the next one.
    mid_idx = n3d // 2 # Takes the middle point of the section to represent its location.
    x, y, z = sec.x3d(mid_idx), sec.y3d(mid_idx), sec.z3d(mid_idx) # x3d(), y3d(), and z3d() are methods that return the x, y, and z coordinates of the 3D points in the section. 

    if any(np.linalg.norm(np.array([x, y, z]) - coord) < 10.0 for coord in soma_coords): # coord is a not pre-existing varible created by the python for-loop-like generator expression. It iterates over each coordinate in soma_coords and checks if the distance between the section's midpoint (x, y, z) and the soma coordinate is less than 10.0 units. If any of the soma coordinates are within this distance threshold, we classify the section as a soma section.
        soma_sections.append(sec) 
    else:
        non_soma_sections.append(sec)

# 4️⃣ Print counts
print(f"\nMatched soma sections     : {len(soma_sections)}")
print(f"Matched non-soma sections : {len(non_soma_sections)}")

# Create a nickname map for cleaner printing and legends
section_nicknames = {}

# Label the soma
for i, sec in enumerate(soma_sections):
    section_nicknames[sec.name()] = f"Soma_{i}"

# Label everything else as Non-Soma
for i, sec in enumerate(non_soma_sections):
    section_nicknames[sec.name()] = f"Non-Soma_{i}"

# ─────────────────────────────────────────────
# Step 3: Apply parameters to soma sections
# ─────────────────────────────────────────────
print("\n── Soma sections ──")
for sec in soma_sections:
    sec.Ra = 150.0          # axial resistance     (ohm·cm)
    sec.cm = 1.0            # membrane capacitance (uF/cm²)
    sec.insert('pas')

    for seg in sec:
        seg.pas.g = 0.00005  # conductance  (S/cm²) = 20 kohm·cm²
        seg.pas.e = -65.0   # reversal potential (mV)

    print(f"  {sec.name():<30} Ra={sec.Ra} | cm={sec.cm} | "
          f"g_pas={sec(0.5).pas.g} | e_pas={sec(0.5).pas.e}")

# ─────────────────────────────────────────────
# Step 4: Apply parameters to all other sections
# ─────────────────────────────────────────────
print("\n── Non-soma sections ──")
for sec in non_soma_sections:
    sec.Ra = 150.0          # dendrites/axon typically higher Ra
    sec.cm = 1.0            # membrane capacitance (uF/cm²)    
    sec.insert('pas')

    # Get the clean name from our map
    clean_name = section_nicknames.get(sec.name(), sec.name())

    for seg in sec:
        seg.pas.g = 0.00005  # conductance  (S/cm²) = 20 kohm·cm²
        seg.pas.e = -65.0

    print(f" {clean_name:<20} Ra={sec.Ra} | cm={sec.cm} | "
          f"g_pas={sec(0.5).pas.g} | e_pas={sec(0.5).pas.e}")

# ─────────────────────────────────────────────
# Step 6: Print full properties of first soma section
# ─────────────────────────────────────────────
if soma_sections:
    print("\nSoma's full properties:\n")
    pprint.pprint(soma_sections[0].psection())

# ─────────────────────────────────────────────────────────────────
# STEP 7: Find terminal dendritic segment (leaf node)
# ─────────────────────────────────────────────────────────────────
# A "leaf" section has no children
all_sections = list(h.allsec())

def get_children(sec):
    return list(sec.children())

leaf_sections = [sec for sec in all_sections if len(get_children(sec)) == 0]
" The first sec in the bracket is the section being checked, and the second sec is the one being com-"
"-pared against. The get_children(sec) function returns a list of child sections for the given sec-"
"-tion sec. The len(get_children(sec)) == 0 condition checks if the length of the list of child sec-"
"-tions is zero, which means that the section has no children and is therefore a leaf section."
" If this condition is true, the section sec is included in the leaf_sections list."

# Pick the first leaf that's a non-soma section
stim_section = None # Initialize stim_section to None before the loop. 
for sec in leaf_sections:
    if sec in non_soma_sections:
        stim_section = sec # Stim as in stimulation, since this is where we'll inject current later.
        break

if stim_section is None:
    stim_section = leaf_sections[0]  # fallback

print(f"\nStimulus site: {stim_section.name()} — injected at seg(0.5)")

# ─────────────────────────────────────────────────────────────────
# STEP 8: Simulation runner with multi-segment recording
# ─────────────────────────────────────────────────────────────────

import matplotlib.gridspec as gridspec # for more flexible subplot layouts

T_STOP   = 5000.0   # ms (5 s)
DT       = 0.025    # ms DT stands for the time step of the simulation.
I_AMP    = 0.0075   # nA

def run_sim(stim_dur_ms, i_amp=I_AMP): # before: (stim_dur_ms, i_amp=I_AMP cm_val=None, g_pas_val=None, Ra_val=None)
    """
    Run one simulation. Returns (t_vec, v_dict) where v_dict maps section name -> v vector.
    Optionally override cm, g_pas, Ra for parametric sweeps.
    """
    # HOC object attributes
    h.dt     = DT
    h.tstop  = T_STOP
    h.v_init = -65.0

    '''Override biophysics if requested. Inother words, if cm_val is not None, set sec.cm = cm_val
      for all sections. If g_pas_val is not None, set seg.pas.g = g_pas_val for all segments in all 
      sections. If Ra_val is not None, set sec.Ra = Ra_val for all sections.'''
    """ for sec in h.allsec():
        if cm_val  is not None: sec.cm  = cm_val
        if Ra_val  is not None: sec.Ra  = Ra_val
        if g_pas_val is not None:
            for seg in sec:
                try:    seg.pas.g = g_pas_val
                except: pass"""

    # Stimulus
    stim = h.IClamp(stim_section(0.5))
    stim.delay = 100.0        # ms — start after 100 ms, '''
    '''why is it delaying by 100 ms? This allows the neuron to reach its resting state before the stimulus is applied,
     providing a clear baseline for observing the effects of the stimulus. By starting the stimulus after 100 ms, 
     we can ensure that any changes in membrane voltage are due to the stimulus rather than transient effects 
     from initializing the simulation.'''
    

    stim.dur   = stim_dur_ms
    stim.amp   = i_amp        # nA

    # Record time
    t_vec = h.Vector().record(h._ref_t)

    # Record voltage in EVERY segment
    v_recs = {}
    for sec in h.allsec():
        for seg in sec:
            key = f"{sec.name()}({seg.x:.3f})" #sec.name return the name of the section, and seg.x gives the normalized position along the section
            v_recs[key] = h.Vector().record(seg._ref_v)

    h.finitialize(-65.0)
    h.run()

    # Convert to numpy
    t_np = np.array(t_vec)
    v_np = {k: np.array(v) for k, v in v_recs.items()}

    # Restore defaults
    for sec in h.allsec():
        sec.cm = 1.0
        sec.Ra = 150.0
        for seg in sec:
            try:    seg.pas.g = 0.00005
            except: pass # the pass in the except causes the except block to do nothing and allows the loop to continue without interruption.

    return t_np, v_np, stim  # return stim to keep it alive during run

# ─────────────────────────────────────────────────────────────────
# STEP 9: Run baseline sims (1 ms and 1 s durations)
# ─────────────────────────────────────────────────────────────────
print("\nRunning 1 ms stimulus simulation...")
t_1ms, v_1ms, _ = run_sim(stim_dur_ms=1.0)

print("Running 1 s stimulus simulation...")
t_1s,  v_1s,  _ = run_sim(stim_dur_ms=1000.0)

stim_site_key = f"{stim_section.name()}(0.500)"  # midpoint key

# ─────────────────────────────────────────────────────────────────
# STEP 10: Parametric sweeps (record peak V at stim site)
# ─────────────────────────────────────────────────────────────────

# --- Current sweep ---
#i_vals = np.linspace(0.001, 0.02, 10)  # nA, i__vals as in injected current values to sweep over, from 0.001 nA to 0.02 nA, with 10 evenly spaced values in that range.
''' why sweep over if the intended injected current is 0.0075 nA? Sweeping over a range of current values allows us 
to understand how the neuron's voltage response changes with different levels of stimulation. By testing a range of currents, 
we can identify thresholds for action potential generation, observe subthreshold responses, and characterize the overall excitability of the neuron. 
This can provide insights into the neuron's behavior under various conditions and help us understand its functional properties.'''

"""
peak_v_i_1ms, peak_v_i_1s = [], []
for i in i_vals:
    _, vd, _ = run_sim(1.0,    i_amp=i)
    peak_v_i_1ms.append(np.max(vd[stim_site_key]))
    _, vd, _ = run_sim(1000.0, i_amp=i)
    peak_v_i_1s.append(np.max(vd[stim_site_key]))"""

# --- Capacitance sweep ---
"""cm_vals = np.linspace(0.5, 3.0, 8)

peak_v_cm_1ms, peak_v_cm_1s = [], []
for cm in cm_vals:
    _, vd, _ = run_sim(1.0,    cm_val=cm)
    peak_v_cm_1ms.append(np.max(vd[stim_site_key]))
    _, vd, _ = run_sim(1000.0, cm_val=cm)
    peak_v_cm_1s.append(np.max(vd[stim_site_key]))"""

# --- Membrane resistance sweep (via g_pas) ---
# Rm = 1/g_pas  (kohm·cm²)
"""g_vals = np.linspace(0.00002, 0.0002, 8)
Rm_vals = 1.0 / g_vals / 1000.0  # MΩ·cm² (just for axis label)

peak_v_rm_1ms, peak_v_rm_1s = [], []
for g in g_vals:
    _, vd, _ = run_sim(1.0,    g_pas_val=g)
    peak_v_rm_1ms.append(np.max(vd[stim_site_key]))
    _, vd, _ = run_sim(1000.0, g_pas_val=g)
    peak_v_rm_1s.append(np.max(vd[stim_site_key]))"""

# --- Axial resistance sweep ---
"""Ra_vals = np.linspace(50, 500, 8)

peak_v_Ra_1ms, peak_v_Ra_1s = [], []
for Ra in Ra_vals:
    _, vd, _ = run_sim(1.0,    Ra_val=Ra)
    peak_v_Ra_1ms.append(np.max(vd[stim_site_key]))
    _, vd, _ = run_sim(1000.0, Ra_val=Ra)
    peak_v_Ra_1s.append(np.max(vd[stim_site_key]))"""

# ─────────────────────────────────────────────────────────────────
# STEP 11: PLOTTING — 2 columns (1 ms | 1 s), 5 rows
# ─────────────────────────────────────────────────────────────────

# Select a few representative segments to show on V-vs-t plot
# (showing all would be thousands of lines — pick ~6 spread across sections)
sample_keys = list(v_1ms.keys()) #[::max(1, len(v_1ms)//6)], the :: syntax is used for slicing the list of keys in v_1ms. The expression len(v_1ms)//6 
#calculates the step size for slicing, which is determined by dividing the total number of keys in v_1ms by 6. 
#This means that we want to select approximately 6 evenly spaced keys from the list. The max(1, len(v_1ms)//6) ensures 
#that the step size is at least 1, so that we don't end up with an empty list if there are fewer than 6 keys.

fig, axes = plt.subplots(2 , 1, figsize=(12, 8)) # before: 5 rows, 2 columns. (in respective order)
fig.suptitle("NEURON Simulation — Single-Neuron Voltage Analysis\n"
             f"Stim site: {stim_section.name()}(0.5)  |  I = {I_AMP} nA",
             fontsize=13, fontweight='bold', y=0.98)

colors_seg = plt.cm.viridis(np.linspace(0, 1, len(sample_keys)))

# ── Row 0: V vs Time ──────────────────────────────────────────────
for ax, t_np, v_np, title in [
    (axes[0], t_1ms, v_1ms, "V vs Time  (stim dur = 1 ms)"),
    (axes[1], t_1s,  v_1s,  "V vs Time  (stim dur = 1 s)"),
]:
    # Inside your plotting loop:
    for k, c in zip(sample_keys, colors_seg):
        # 1. Clean the key to get the base section name
        base_name = k.split('(')[0]
        
        # 2. Get the nickname (e.g., NonSoma_4)
        display_lbl = section_nicknames.get(base_name, base_name)
        
        ax.plot(t_np / 1000.0, v_np[k], color=c, lw=0.8, label=display_lbl)

    # 3. Always show the STIM SITE in red and bold
    # We use stim_section.name() to find the right key dynamically
    dynamic_stim_key = f"{stim_section.name()}(0.500)"
    if dynamic_stim_key in v_np:
        ax.plot(t_np / 1000.0, v_np[dynamic_stim_key],
                color='red', lw=1.8, label='STIM SITE', zorder=10)

    ax.axhline(-65, color='gray', lw=0.6, ls='--', label='V_rest')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Membrane Voltage (mV)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=6, ncol=4, loc='upper right') #ncol is the # of columns in the legend
    ax.grid(True, alpha=0.3)

# ── Row 1: V vs Injected Current ──────────────────────────────────
"""for ax, peak, title in [
    (axes[1,0], peak_v_i_1ms, "Peak V vs Injected Current  (1 ms)"),
    (axes[1,1], peak_v_i_1s,  "Peak V vs Injected Current  (1 s)"),
]:
    ax.plot(i_vals * 1000, peak, 'o-', color='steelblue', lw=1.8)
    ax.axhline(-65, color='gray', lw=0.6, ls='--')
    ax.set_xlabel("Injected Current (pA)")
    ax.set_ylabel("Peak Voltage (mV)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)"""

# ── Row 2: V vs Capacitance ───────────────────────────────────────
"""for ax, peak, title in [
    (axes[2,0], peak_v_cm_1ms, "Peak V vs Membrane Capacitance  (1 ms)"),
    (axes[2,1], peak_v_cm_1s,  "Peak V vs Membrane Capacitance  (1 s)"),
]:
    ax.plot(cm_vals, peak, 's-', color='darkorange', lw=1.8)
    ax.axhline(-65, color='gray', lw=0.6, ls='--')
    ax.set_xlabel("cm (µF/cm²)")
    ax.set_ylabel("Peak Voltage (mV)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)"""

# ── Row 3: V vs Membrane Resistance ──────────────────────────────
"""for ax, peak, title in [
    (axes[3,0], peak_v_rm_1ms, "Peak V vs Membrane Resistance  (1 ms)"),
    (axes[3,1], peak_v_rm_1s,  "Peak V vs Membrane Resistance  (1 s)"),
]:
    ax.plot(1.0/g_vals * 1e-3, peak, 'D-', color='mediumseagreen', lw=1.8)
    ax.axhline(-65, color='gray', lw=0.6, ls='--')
    ax.set_xlabel("Rm = 1/g_pas  (kΩ·cm²)")
    ax.set_ylabel("Peak Voltage (mV)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)"""

# ── Row 4: V vs Axial Resistance ─────────────────────────────────
"""for ax, peak, title in [
    (axes[4,0], peak_v_Ra_1ms, "Peak V vs Axial Resistance  (1 ms)"),
    (axes[4,1], peak_v_Ra_1s,  "Peak V vs Axial Resistance  (1 s)"),
]:
    ax.plot(Ra_vals, peak, '^-', color='crimson', lw=1.8)
    ax.axhline(-65, color='gray', lw=0.6, ls='--')
    ax.set_xlabel("Ra (Ω·cm)")
    ax.set_ylabel("Peak Voltage (mV)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)"""

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("neuron_voltage_analysis.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to neuron_voltage_analysis.png")

input("Press Enter to exit...")

