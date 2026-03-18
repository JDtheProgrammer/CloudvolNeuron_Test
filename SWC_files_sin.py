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
swc_path = '/home/aksay_lab/NeuronProject/CloudvolNeuron_Test/SWC_files/76182_reRoot_reSample_5000.swc'  
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
for sec in h.allsec():
    cell.sl.append(sec)

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

# ─────────────────────────────────────────────
# Step 2: Match every NEURON section to soma or non-soma
# via 3D coordinate proximity
# ─────────────────────────────────────────────
soma_sections     = []
non_soma_sections = []
unmatched         = []

for sec in h.allsec():
    n3d     = int(h.n3d(sec=sec))
    matched = False

    for i in range(n3d):
        pt = np.array([h.x3d(i, sec=sec),
                       h.y3d(i, sec=sec),
                       h.z3d(i, sec=sec)])

        soma_dist     = np.min(np.linalg.norm(soma_coords     - pt, axis=1))
        non_soma_dist = np.min(np.linalg.norm(non_soma_coords - pt, axis=1))

        if soma_dist < 1e-3:
            soma_sections.append(sec)
            matched = True
            break
        elif non_soma_dist < 1e-3:
            non_soma_sections.append(sec)
            matched = True
            break

    if not matched:
        unmatched.append(sec)

print(f"\nMatched soma sections     : {len(soma_sections)}")
print(f"Matched non-soma sections : {len(non_soma_sections)}")
print(f"Unmatched sections        : {len(unmatched)}")

# ─────────────────────────────────────────────
# Step 3: Apply parameters to soma sections
# ─────────────────────────────────────────────
print("\n── Soma sections ──")
for sec in soma_sections:
    sec.Ra = 100.0          # axial resistance     (ohm·cm)
    sec.cm = 1.0            # membrane capacitance (uF/cm²)
    sec.insert('pas')
    for seg in sec:
        seg.pas.g = 0.0001  # conductance  (S/cm²)
        seg.pas.e = -65.0   # reversal potential (mV)

    print(f"  {sec.name():<30} Ra={sec.Ra} | cm={sec.cm} | "
          f"g_pas={sec(0.5).pas.g} | e_pas={sec(0.5).pas.e}")

# ─────────────────────────────────────────────
# Step 4: Apply parameters to all other sections
# ─────────────────────────────────────────────
print("\n── Non-soma sections ──")
for sec in non_soma_sections:
    sec.Ra = 150.0          # dendrites/axon typically higher Ra
    sec.cm = 1.0
    sec.insert('pas')
    for seg in sec:
        seg.pas.g = 0.0001
        seg.pas.e = -65.0

    print(f"  {sec.name():<30} Ra={sec.Ra} | cm={sec.cm} | "
          f"g_pas={sec(0.5).pas.g} | e_pas={sec(0.5).pas.e}")

# ─────────────────────────────────────────────
# Step 5: Fallback — apply default Ra to unmatched
# so no section is left without parameters
# ─────────────────────────────────────────────
if unmatched:
    print(f"\n── Unmatched sections (default params) ──")
    for sec in unmatched:
        sec.Ra = 150.0
        sec.cm = 1.0
        sec.insert('pas')
        for seg in sec:
            seg.pas.g = 0.0001
            seg.pas.e = -65.0
        print(f"  {sec.name()} — default params applied")

# ─────────────────────────────────────────────
# Step 6: Print full properties of first soma section
# ─────────────────────────────────────────────
if soma_sections:
    print("\nSoma's full properties:\n")
    pprint.pprint(soma_sections[0].psection())
input("Press Enter to exit...")

