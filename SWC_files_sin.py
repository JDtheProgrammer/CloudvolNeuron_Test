# =========================
# Imports
# =========================
# Import this package/function because a later step depends on it.
import re
# Import this package/function because a later step depends on it.
import json
# Import this package/function because a later step depends on it.
import numpy as np
# Import this package/function because a later step depends on it.
import matplotlib.pyplot as plt
# Import this package/function because a later step depends on it.
import matplotlib.cm as mcm
# Import this package/function because a later step depends on it.
import matplotlib.colors as mcolors
# Import this package/function because a later step depends on it.
import matplotlib.colorbar as mcbar
# Import this package/function because a later step depends on it.
import vtk
# Import this package/function because a later step depends on it.
import os
# Import this package/function because a later step depends on it.
import sys

# Import this package/function because a later step depends on it.
from scipy.stats import spearmanr
# Import this package/function because a later step depends on it.
from sklearn.cluster import KMeans
# Import this package/function because a later step depends on it.
from sklearn.preprocessing import StandardScaler
# Import this package/function because a later step depends on it.
from sklearn.metrics import silhouette_score
# Import this package/function because a later step depends on it.
from collections import defaultdict

# Import this package/function because a later step depends on it.
from neuron import h, gui
# Import this package/function because a later step depends on it.
from neuron.units import ms, mV, um

# Print a diagnostic message/value so the run can be checked while executing.
print(f'VTK version: {vtk.VTK_VERSION}')

# =================================================================================================================
# POSTER SIZING PARAMETERS — edit these to scale the entire pipeline at once (Poster PDF in JD's downloads folder)
#
#           These constants control text size, line thickness, marker size, and figure size globally.
#               Change them here once and the visual updates propagate through the whole script.
# =================================================================================================================

# ----- Font sizes -----
# Set title font size so all subplot titles scale consistently.
FONT_TITLE    = 24   # subplot titles
# Set axis-label font size so units remain readable on poster figures.
FONT_LABEL    = 20   # x/y axis labels
# Set tick-label font size so axis numbers are readable.
FONT_TICK     = 18   # axis tick labels
# Set legend font size so labels remain visible without crowding plots.
FONT_LEGEND   = 17   # legend text
# Set colorbar font size so scale labels/ticks match the figure size.
FONT_CBAR     = 17   # colorbar label + ticks
# Set annotation font size for text placed inside plots.
FONT_ANNOT    = 16   # annotations inside plots
# Set figure-level title font size for the main caption/title.
FONT_SUPTITLE = 26   # figure-level title

# ----- Line widths -----
# Store `LW_MORPH` so later simulation, analysis, or plotting code can reuse it.
LW_MORPH      = 3.2  # morphology branch line width
# Store `LW_MORPH_OUT` so later simulation, analysis, or plotting code can reuse it.
LW_MORPH_OUT  = 4.4  # black outline behind morphology lines
# Store `LW_TRACE` so later simulation, analysis, or plotting code can reuse it.
LW_TRACE      = 3.4  # voltage trace thickness
# Store `LW_SKEL_SEL` so later simulation, analysis, or plotting code can reuse it.
LW_SKEL_SEL   = 5.0  # highlighted selected arbor thickness
# Store `LW_SKEL_SOMA` so later simulation, analysis, or plotting code can reuse it.
LW_SKEL_SOMA  = 8.0  # soma-adjacent arbor thickness
# Store `LW_SKEL_GREY` so later simulation, analysis, or plotting code can reuse it.
LW_SKEL_GREY  = 2.2  # non-selected grey arbor thickness
# Store `LW_SKEL_AXON` so later simulation, analysis, or plotting code can reuse it.
LW_SKEL_AXON  = 3.4  # axon thickness

# ----- Marker sizes -----
# Store `MS_SOMA` so later simulation, analysis, or plotting code can reuse it.
MS_SOMA       = 260  # soma marker size
# Store `MS_STIM` so later simulation, analysis, or plotting code can reuse it.
MS_STIM       = 430  # stimulus site star size
# Store `MS_DOT_BRANCH` so later simulation, analysis, or plotting code can reuse it.
MS_DOT_BRANCH = 80   # branch-order scatter dots
# Store `MS_DOT_DIST` so later simulation, analysis, or plotting code can reuse it.
MS_DOT_DIST   = 85   # TTM/FWHM scatter dots
# Store `MS_DOT_CLUST` so later simulation, analysis, or plotting code can reuse it.
MS_DOT_CLUST  = 95   # cluster scatter dots

# ----- Figure sizes -----
# Store `FIG_GRADIENT` so later simulation, analysis, or plotting code can reuse it.
FIG_GRADIENT  = (11, 19)    # morphology gradient figures
# Store `FIG_COMPOSITE` so later simulation, analysis, or plotting code can reuse it.
FIG_COMPOSITE = (23, 12.5)  # tree + current + voltage composite
# Store `FIG_VOLTAGE` so later simulation, analysis, or plotting code can reuse it.
FIG_VOLTAGE   = (16, 12)    # clustered voltage figure
# Store `FIG_BRANCH` so later simulation, analysis, or plotting code can reuse it.
FIG_BRANCH    = (21, 14)    # branch-order figure
# Store `FIG_DISTANCE` so later simulation, analysis, or plotting code can reuse it.
FIG_DISTANCE  = (14, 21)    # TTM/FWHM figure

# ----- Colorbar sizing -----
# Store `CBAR_FRACTION` so later simulation, analysis, or plotting code can reuse it.
CBAR_FRACTION = 0.050
# Store `CBAR_THICK` so later simulation, analysis, or plotting code can reuse it.
CBAR_THICK    = 22

# Apply global matplotlib defaults so anything not explicitly set also scales up
# Apply a plotting/style command to format the current figure.
plt.rcParams.update({
    # Execute this statement as part of the current analysis step.
    'font.size':        FONT_TICK,
    # Execute this statement as part of the current analysis step.
    'axes.titlesize':   FONT_TITLE,
    # Execute this statement as part of the current analysis step.
    'axes.labelsize':   FONT_LABEL,
    # Execute this statement as part of the current analysis step.
    'xtick.labelsize':  FONT_TICK,
    # Execute this statement as part of the current analysis step.
    'ytick.labelsize':  FONT_TICK,
    # Execute this statement as part of the current analysis step.
    'legend.fontsize':  FONT_LEGEND,
    # Execute this statement as part of the current analysis step.
    'figure.titlesize': FONT_SUPTITLE,
    # Execute this statement as part of the current analysis step.
    'lines.linewidth':  LW_TRACE,
})


# ==================================================================================================================
#                                    File loading and morphology parsing     
# ==================================================================================================================

# Load NEURON tools for SWC import and 3D visualization
# Load NEURON/HOC support code needed for simulation and morphology import.
h.load_file("stdrun.hoc")
# Load NEURON/HOC support code needed for simulation and morphology import.
h.load_file("import3d.hoc")

# Load SWC file
# Store the SWC file path so the importer and manual parser read the same morphology.
swc_path = '/home/aksay_lab/NeuronProject/CloudvolNeuron_Test/SWC_files/76182_reRoot_reSample_5000.swc'

# Print a diagnostic message/value so the run can be checked while executing.
print("File exists?", os.path.exists(swc_path))
# Print a diagnostic message/value so the run can be checked while executing.
print("Loading SWC file:", swc_path)

# Store `reader` so later simulation, analysis, or plotting code can reuse it.
reader = h.Import3d_SWC_read()
# Give the SWC file path to NEURON’s SWC reader.
reader.input(swc_path)

# Create the NEURON importer that converts the SWC tree into NEURON sections.
importer = h.Import3d_GUI(reader, 1) # h. is the NEURON hoc module; Import3d_GUI is the class that handles SWC import

# Create a Cell object to hold the morphology; the importer will populate it with sections based on the SWC file.
# Define this class to hold imported NEURON sections in a simple object.
class Cell:  
    # Define `__init__` to isolate this repeated calculation/plotting step.
    def __init__(self):
        # Store `self.sl` so later simulation, analysis, or plotting code can reuse it.
        self.sl = h.SectionList()

# The importer will call cell.sl.append(sec) for each section it creates from the SWC file.
# Store `cell` so later simulation, analysis, or plotting code can reuse it.
cell = Cell()
# Build the NEURON section objects from the imported SWC morphology.
importer.instantiate(cell)

# ── Scale NEURON's internal 3D coordinates from nm to um ─────────────────────
# Define the nanometer-to-micrometer conversion because NEURON expects morphology dimensions in µm.
NM_TO_UM = 1000.0  # This is because many SWC files use nanometres, but NEURON typically uses microns for morphology coordinates.
# Iterate through every relevant item so none of the morphology/simulation data is skipped.
for sec in h.allsec(): # h. allsec() returns an iterator over all sections in the NEURON model
    # Store `n3d` so later simulation, analysis, or plotting code can reuse it.
    n3d = sec.n3d() # n3d is the number of 3D points in the section; if it's 0, we skip scaling because there are no points to scale
    # Test this condition so the script handles this case safely/correctly.
    if n3d == 0:
        # Skip this item because it is invalid or not needed for the current calculation.
        continue
    # Store `pts` so later simulation, analysis, or plotting code can reuse it.
    pts = []
    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for i in range(n3d):
        # Append this result to a list so it is included in later processing.
        pts.append((sec.x3d(i)    / NM_TO_UM,
                    # Call this function/method to execute the next operation in the pipeline.
                    sec.y3d(i)    / NM_TO_UM,
                    # Call this function/method to execute the next operation in the pipeline.
                    sec.z3d(i)    / NM_TO_UM,
                    # Call this function/method to execute the next operation in the pipeline.
                    sec.diam3d(i) / NM_TO_UM))
    # Clear old 3D points before writing the scaled coordinates back into the section.
    h.pt3dclear(sec=sec)
    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for x, y, z, d in pts:
        # Add one scaled 3D morphology point back into the NEURON section.
        h.pt3dadd(x, y, z, d, sec=sec)

# =========================
# 3D visualization
# =========================
# Store `shape` so later simulation, analysis, or plotting code can reuse it.
shape = h.Shape(cell.sl)
# Call this function/method to execute the next operation in the pipeline.
shape.exec_menu('3D Rotate')
# Call this function/method to execute the next operation in the pipeline.
shape.show(0)
# Call this function/method to execute the next operation in the pipeline.
shape.flush()
# Call this function/method to execute the next operation in the pipeline.
h.doNotify()


# Parse SWC nodes manually
# ===================================================================================================
# FIX #13: SWC parsing moved BEFORE the nseg-fixing loop so node_order/nodes are available
#          for any downstream code that might reference them immediately after Step 1.

# Create a dictionary that will store each SWC node by its numeric id.
nodes      = {} # node_id → {node_id, type, x, y, z, radius, parent_id}
# Create an ordered list so later traversals can follow the original SWC order.
node_order = [] # to preserve original SWC node order for any algorithms that might rely on it (e.g. branch order)

# FIX #10: NM_TO_UM already defined above; no redefinition inside this loop
# Open the SWC file for reading while automatically closing it afterward.
with open(swc_path, 'r') as f: # open the SWC file for reading
    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for line in f: # iterate over each line in the file
        # Store `line` so later simulation, analysis, or plotting code can reuse it.
        line = line.strip()
        # Test this condition so the script handles this case safely/correctly.
        if line.startswith('#') or line == '':
            # Skip this item because it is invalid or not needed for the current calculation.
            continue
        # Store `parts` so later simulation, analysis, or plotting code can reuse it.
        parts     = line.split()
        # Store `node_id` so later simulation, analysis, or plotting code can reuse it.
        node_id   = int(parts[0])
        # Store `node_type` so later simulation, analysis, or plotting code can reuse it.
        node_type = int(parts[1])
        # Store `x` so later simulation, analysis, or plotting code can reuse it.
        x         = float(parts[2]) / NM_TO_UM
        # Store `y` so later simulation, analysis, or plotting code can reuse it.
        y         = float(parts[3]) / NM_TO_UM
        # Store `z` so later simulation, analysis, or plotting code can reuse it.
        z         = float(parts[4]) / NM_TO_UM
        # Store `radius` so later simulation, analysis, or plotting code can reuse it.
        radius    = float(parts[5]) / NM_TO_UM
        # Store `parent_id` so later simulation, analysis, or plotting code can reuse it.
        parent_id = int(parts[6])
        # Store `nodes[node_id]` so later simulation, analysis, or plotting code can reuse it.
        nodes[node_id] = {
            # Execute this statement as part of the current analysis step.
            'node_id': node_id, 'type': node_type,
            # Execute this statement as part of the current analysis step.
            'x': x, 'y': y, 'z': z,
            # Execute this statement as part of the current analysis step.
            'radius': radius, 'parent_id': parent_id
        }
        # Append this result to a list so it is included in later processing.
        node_order.append(node_id)

# Print a diagnostic message/value so the run can be checked while executing.
print(f"Expected SWC edges: {len(nodes) - 1}")

# FIX #13: nseg loop moved here, AFTER nodes is populated (logical ordering)
# Store `total_fixed_segments` so later simulation, analysis, or plotting code can reuse it.
total_fixed_segments = 0
# Iterate through every relevant item so none of the morphology/simulation data is skipped.
for sec in h.allsec():
    # Store `n3d` so later simulation, analysis, or plotting code can reuse it.
    n3d = sec.n3d()
    # Test this condition so the script handles this case safely/correctly.
    if n3d > 1:
        # Set the number of computational segments so segment locations align with 3D morphology points.
        sec.nseg = n3d - 1
    # Use this fallback when the earlier condition is not true.
    else:
        # Set the number of computational segments so segment locations align with 3D morphology points.
        sec.nseg = 1
    # Set the number of computational segments so segment locations align with 3D morphology points.
    total_fixed_segments += int(sec.nseg)

# Print a diagnostic message/value so the run can be checked while executing.
print("\nAfter fixing nseg:")
# Print a diagnostic message/value so the run can be checked while executing.
print("Total NEURON segments:", total_fixed_segments)

# Print bounding box
# Store `all_x` so later simulation, analysis, or plotting code can reuse it.
all_x = [nodes[n]['x'] for n in nodes]
# Store `all_y` so later simulation, analysis, or plotting code can reuse it.
all_y = [nodes[n]['y'] for n in nodes]
# Store `all_z` so later simulation, analysis, or plotting code can reuse it.
all_z = [nodes[n]['z'] for n in nodes]
# Print a diagnostic message/value so the run can be checked while executing.
print(f"\nSWC coordinate bounding box:")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"  x: {min(all_x):.1f} to {max(all_x):.1f}")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"  y: {min(all_y):.1f} to {max(all_y):.1f}")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"  z: {min(all_z):.1f} to {max(all_z):.1f}")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"  If these values are > 10,000 the coordinates are likely in nanometres.")

# Store `edge_lengths` so later simulation, analysis, or plotting code can reuse it.
edge_lengths = []
# Iterate through every relevant item so none of the morphology/simulation data is skipped.
for nid, d in nodes.items():
    # Store `parent` so later simulation, analysis, or plotting code can reuse it.
    parent = d['parent_id']
    # Test this condition so the script handles this case safely/correctly.
    if parent == -1:
        # Skip this item because it is invalid or not needed for the current calculation.
        continue
    # Store `n` so later simulation, analysis, or plotting code can reuse it.
    n  = nodes[nid]
    # Store `p` so later simulation, analysis, or plotting code can reuse it.
    p  = nodes[parent]
    # Store `dx` so later simulation, analysis, or plotting code can reuse it.
    dx = n['x'] - p['x']
    # Store `dy` so later simulation, analysis, or plotting code can reuse it.
    dy = n['y'] - p['y']
    # Store `dz` so later simulation, analysis, or plotting code can reuse it.
    dz = n['z'] - p['z']
    # Append this result to a list so it is included in later processing.
    edge_lengths.append(np.sqrt(dx**2 + dy**2 + dz**2))

# Convert data into a NumPy array for vectorized numerical operations.
edge_lengths = np.array(edge_lengths)
# Print a diagnostic message/value so the run can be checked while executing.
print(f"\nSWC edge length statistics:")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"  Min    : {edge_lengths.min():.1f}")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"  Max    : {edge_lengths.max():.1f}")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"  Mean   : {edge_lengths.mean():.1f}")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"  Median : {np.median(edge_lengths):.1f}")

# =====================================================================
# BUILD TRUE SECTIONS FROM SWC GRAPH
# =====================================================================
# Map each parent node to its child nodes so the SWC tree can be traversed.
children = defaultdict(list)
# Iterate through every relevant item so none of the morphology/simulation data is skipped.
for nid, d in nodes.items():
    # Store `parent` so later simulation, analysis, or plotting code can reuse it.
    parent = d['parent_id']
    # Test this condition so the script handles this case safely/correctly.
    if parent != -1:
        # Append this result to a list so it is included in later processing.
        children[parent].append(nid)

# Define `is_branch` to isolate this repeated calculation/plotting step.
def is_branch(nid): return len(children[nid]) > 1
# Define `is_leaf` to isolate this repeated calculation/plotting step.
def is_leaf(nid):   return len(children[nid]) == 0
# Define `is_root` to isolate this repeated calculation/plotting step.
def is_root(nid):   return nodes[nid]['parent_id'] == -1

# Store reconstructed true sections as node paths between branches/leaves.
sections      = []
# Store `visited_edges` so later simulation, analysis, or plotting code can reuse it.
visited_edges = set()

# Iterate through every relevant item so none of the morphology/simulation data is skipped.
for nid in nodes:
    # Store `parent` so later simulation, analysis, or plotting code can reuse it.
    parent = nodes[nid]['parent_id']
    # Test this condition so the script handles this case safely/correctly.
    if is_root(nid) or is_branch(nid) or (parent != -1 and is_branch(parent)):
        # Iterate through every relevant item so none of the morphology/simulation data is skipped.
        for child in children[nid]:
            # Store `path` so later simulation, analysis, or plotting code can reuse it.
            path = [nid, child]
            # Add this item to a set for fast membership checking without duplicates.
            visited_edges.add((nid, child))
            # Store `current` so later simulation, analysis, or plotting code can reuse it.
            current = child
            # Keep traversing/searching until the morphology path reaches its stopping condition.
            while not is_branch(current) and not is_leaf(current):
                # Store `next_node` so later simulation, analysis, or plotting code can reuse it.
                next_node = children[current][0]
                # Test this condition so the script handles this case safely/correctly.
                if (current, next_node) in visited_edges:
                    # Exit the loop because the traversal/calculation is complete for this path.
                    break
                # Append this result to a list so it is included in later processing.
                path.append(next_node)
                # Add this item to a set for fast membership checking without duplicates.
                visited_edges.add((current, next_node))
                # Store `current` so later simulation, analysis, or plotting code can reuse it.
                current = next_node
            # Append this result to a list so it is included in later processing.
            sections.append(path)

# Print a diagnostic message/value so the run can be checked while executing.
print(f"\nTOTAL TRUE SECTIONS: {len(sections)}")

# ===================================================================================================
# STEP 2 — Identify soma nodes
# ===================================================================================================
# Identify SWC nodes treated as soma so soma and dendrite/axon regions can be separated.
soma_node_ids = {
    # Execute this statement as part of the current analysis step.
    nid for nid, d in nodes.items()
    # Test this condition so the script handles this case safely/correctly.
    if nid == 1 or d['parent_id'] == 1
}
# Store all nodes outside the soma for dendrite/axon analysis.
non_soma_node_ids = set(nodes.keys()) - soma_node_ids
# Convert data into a NumPy array for vectorized numerical operations.
soma_coords = np.array([[nodes[n]['x'], nodes[n]['y'], nodes[n]['z']] for n in soma_node_ids])

# Print a diagnostic message/value so the run can be checked while executing.
print(f"Soma nodes: {len(soma_node_ids)}")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"Non-soma nodes: {len(non_soma_node_ids)}")

# ===================================================================================================
# STEP 3 — Classify SEGMENTS
# ===================================================================================================
# Set the distance cutoff used to classify NEURON segments as soma-adjacent.
SOMA_THRESHOLD = 1.5

# Initialize the list that will hold NEURON segments classified as soma.
soma_segments     = []
# Initialize the list that will hold NEURON segments outside the soma.
non_soma_segments = []
# Create labels that make segment ids readable in plots/tables.
segment_labels    = {}

# Store `soma_count` so later simulation, analysis, or plotting code can reuse it.
soma_count     = 0
# Store `non_soma_count` so later simulation, analysis, or plotting code can reuse it.
non_soma_count = 0

# Iterate through every relevant item so none of the morphology/simulation data is skipped.
for sec in h.allsec():
    # Store `n3d_pts` so later simulation, analysis, or plotting code can reuse it.
    n3d_pts = sec.n3d()
    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for seg in sec:
        # Store `is_soma` so later simulation, analysis, or plotting code can reuse it.
        is_soma = False
        # Test this condition so the script handles this case safely/correctly.
        if n3d_pts > 0:
            # Store `total_arc` so later simulation, analysis, or plotting code can reuse it.
            total_arc = sec.arc3d(n3d_pts - 1)
            # Test this condition so the script handles this case safely/correctly.
            if total_arc > 0:
                # Convert data into a NumPy array for vectorized numerical operations.
                arc_fracs = np.array([sec.arc3d(i) / total_arc for i in range(n3d_pts)])
                # Find the closest index/value match.
                idx = int(np.argmin(np.abs(arc_fracs - seg.x)))
                # Convert data into a NumPy array for vectorized numerical operations.
                seg_xyz = np.array([sec.x3d(idx), sec.y3d(idx), sec.z3d(idx)])
                # Iterate through every relevant item so none of the morphology/simulation data is skipped.
                for coord in soma_coords:
                    # Test this condition so the script handles this case safely/correctly.
                    if np.linalg.norm(seg_xyz - coord) < SOMA_THRESHOLD:
                        # Store `is_soma` so later simulation, analysis, or plotting code can reuse it.
                        is_soma = True
                        # Exit the loop because the traversal/calculation is complete for this path.
                        break
        # Store `key` so later simulation, analysis, or plotting code can reuse it.
        key = f"{sec.name()}({seg.x:.3f})"
        # Test this condition so the script handles this case safely/correctly.
        if is_soma:
            # Append this result to a list so it is included in later processing.
            soma_segments.append((sec, seg))
            # Store `segment_labels[key]` so later simulation, analysis, or plotting code can reuse it.
            segment_labels[key] = f"Soma_{soma_count}"
            # Store `soma_count +` so later simulation, analysis, or plotting code can reuse it.
            soma_count += 1
        # Use this fallback when the earlier condition is not true.
        else:
            # Append this result to a list so it is included in later processing.
            non_soma_segments.append((sec, seg))
            # Store `segment_labels[key]` so later simulation, analysis, or plotting code can reuse it.
            segment_labels[key] = f"NonSoma_{non_soma_count}"
            # Store `non_soma_count +` so later simulation, analysis, or plotting code can reuse it.
            non_soma_count += 1

# Print a diagnostic message/value so the run can be checked while executing.
print(f"\nTotal segments: {soma_count + non_soma_count}")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"Soma segments: {soma_count}")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"Non-soma segments: {non_soma_count}")

# ===================================================================================================
# STEP 4 — Apply passive properties
# ===================================================================================================
# Iterate through every relevant item so none of the morphology/simulation data is skipped.
for sec in h.allsec():
    # Set axial resistance, which controls current flow along each section.
    sec.Ra = 150.0
    # Set membrane capacitance, which affects how fast voltage changes over time.
    sec.cm  = 1.0
    # Insert passive leak channels so each segment has passive membrane behavior.
    sec.insert('pas')
    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for seg in sec:
        # Set passive leak conductance, which determines membrane resistance.
        seg.pas.g = 0.00005
        # Set passive reversal/resting potential for the leak mechanism.
        seg.pas.e = -65.0

# ===================================================================================================
# STEP 5 — Find stimulation site (non-soma leaf)
# ===================================================================================================
# Store `soma_secs` so later simulation, analysis, or plotting code can reuse it.
soma_secs = {sec for sec, _ in soma_segments}
# Test this condition so the script handles this case safely/correctly.
if not soma_secs:
    # Store `soma_secs` so later simulation, analysis, or plotting code can reuse it.
    soma_secs = {list(h.allsec())[0]}

# Store `sec_ref_map` so later simulation, analysis, or plotting code can reuse it.
sec_ref_map = {sec: h.SectionRef(sec=sec) for sec in h.allsec()}

# Define `path_distance_from_soma` to isolate this repeated calculation/plotting step.
def path_distance_from_soma(sec, soma_secs, dx=0.5):
    # Test this condition so the script handles this case safely/correctly.
    if sec in soma_secs:
        # Return the final value(s) produced by this helper function.
        return 0.0
    # Store `dist` so later simulation, analysis, or plotting code can reuse it.
    dist = sec.L * dx
    # Store `sr` so later simulation, analysis, or plotting code can reuse it.
    sr = sec_ref_map[sec]
    # Keep traversing/searching until the morphology path reaches its stopping condition.
    while sr.has_parent():
        # Store `parent` so later simulation, analysis, or plotting code can reuse it.
        parent = sr.parent().sec
        # Test this condition so the script handles this case safely/correctly.
        if parent in soma_secs:
            # Store `dist +` so later simulation, analysis, or plotting code can reuse it.
            dist += parent.L * 0.5
            # Exit the loop because the traversal/calculation is complete for this path.
            break
        # Use this fallback when the earlier condition is not true.
        else:
            # Store `dist +` so later simulation, analysis, or plotting code can reuse it.
            dist += parent.L
            # Store `sr` so later simulation, analysis, or plotting code can reuse it.
            sr = sec_ref_map[parent]
    # Return the final value(s) produced by this helper function.
    return dist

# Define `get_children` to isolate this repeated calculation/plotting step.
def get_children(sec):
    # Return the final value(s) produced by this helper function.
    return list(sec.children())

# Store `all_sections` so later simulation, analysis, or plotting code can reuse it.
all_sections  = list(h.allsec())
# Store `leaf_sections` so later simulation, analysis, or plotting code can reuse it.
leaf_sections = [sec for sec in all_sections if len(get_children(sec)) == 0]
# Store `non_soma_sections` so later simulation, analysis, or plotting code can reuse it.
non_soma_sections = {sec for sec, _ in non_soma_segments}

# ===================================================================================================
# AXON IDENTIFICATION — topology + geometry heuristic
# ===================================================================================================
# The SWC file has all nodes labelled type 1 (soma/undefined), so SWC type codes cannot be used.
#
# Biological rationale:
#   The axon is the single longest unbranched cable emerging from the soma.  It may have a
#   few collateral branches, but its trunk is characteristically:
#     (a) the longest continuous unbranched path from the soma, AND
#     (b) unusually thin (small mean diameter) compared to thick apical dendrites.
#
# Algorithm — "longest unbranched subtree" heuristic:
#   1. For every child section of the soma, perform a DFS / path-following walk that
#      accumulates cable length WITHOUT crossing any bifurcation point.
#      At each bifurcation the walk terminates; we record the total unbranched cable length
#      of that subtree's main trunk.
#   2. The subtree with the longest unbranched trunk AND whose mean diameter is below the
#      median diameter of all non-soma sections is labelled the axon.
#      (Criterion (b) breaks ties in morphologies where one dendrite happens to be long
#      and straight before it first bifurcates.)
#   3. All sections reachable from that root section (the entire axonal tree, including any
#      collaterals) are added to axon_sections.
#
# This correctly identifies the axon even when type codes are absent.
# ===================================================================================================

# Define `_section_mean_diam` to isolate this repeated calculation/plotting step.
def _section_mean_diam(sec):
    """Mean diameter (um) over all pt3d points of a section."""
    # Store `n` so later simulation, analysis, or plotting code can reuse it.
    n = sec.n3d()
    # Test this condition so the script handles this case safely/correctly.
    if n == 0:
        # Return the final value(s) produced by this helper function.
        return sec.diam   # fallback to uniform diameter
    # Return the final value(s) produced by this helper function.
    return float(np.mean([sec.diam3d(i) for i in range(n)]))

# Define `_unbranched_trunk_length` to isolate this repeated calculation/plotting step.
def _unbranched_trunk_length(root_sec):
    """
    # Execute this statement as part of the current analysis step.
    Walk from root_sec toward distal tips, always following the SINGLE child
    # Execute this statement as part of the current analysis step.
    at each step.  Stop as soon as we hit a bifurcation (2+ children) or a leaf.
    # Execute this statement as part of the current analysis step.
    Return total cable length of this unbranched trunk.
    """
    # Store `total` so later simulation, analysis, or plotting code can reuse it.
    total = root_sec.L
    # Store `sec` so later simulation, analysis, or plotting code can reuse it.
    sec   = root_sec
    # Keep traversing/searching until the morphology path reaches its stopping condition.
    while True:
        # Store `kids` so later simulation, analysis, or plotting code can reuse it.
        kids = list(sec.children())
        # Test this condition so the script handles this case safely/correctly.
        if len(kids) != 1:   # bifurcation or leaf → stop
            # Exit the loop because the traversal/calculation is complete for this path.
            break
        # Store `sec` so later simulation, analysis, or plotting code can reuse it.
        sec    = kids[0]
        # Store `total +` so later simulation, analysis, or plotting code can reuse it.
        total += sec.L
    # Return the final value(s) produced by this helper function.
    return total

# Define `_all_descendant_sections` to isolate this repeated calculation/plotting step.
def _all_descendant_sections(root_sec):
    """Return the set of all sections reachable from root_sec (inclusive) by DFS."""
    # Store `result` so later simulation, analysis, or plotting code can reuse it.
    result = set()
    # Store `stack` so later simulation, analysis, or plotting code can reuse it.
    stack  = [root_sec]
    # Keep traversing/searching until the morphology path reaches its stopping condition.
    while stack:
        # Store `s` so later simulation, analysis, or plotting code can reuse it.
        s = stack.pop()
        # Test this condition so the script handles this case safely/correctly.
        if s in result:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue
        # Add this item to a set for fast membership checking without duplicates.
        result.add(s)
        # Extend the list with multiple connected items during traversal.
        stack.extend(list(s.children()))
    # Return the final value(s) produced by this helper function.
    return result

# Candidate axon roots = direct children of soma sections (non-soma themselves)
# Store `soma_children` so later simulation, analysis, or plotting code can reuse it.
soma_children = []
# Iterate through every relevant item so none of the morphology/simulation data is skipped.
for s in soma_secs:
    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for child in list(s.children()):
        # Test this condition so the script handles this case safely/correctly.
        if child not in soma_secs:
            # Append this result to a list so it is included in later processing.
            soma_children.append(child)

# Compute median diameter across all non-soma sections (used as thinness threshold)
# Store `all_non_soma_sec_list` so later simulation, analysis, or plotting code can reuse it.
all_non_soma_sec_list = list(non_soma_sections)
# Store `median_diam` so later simulation, analysis, or plotting code can reuse it.
median_diam = float(np.median([_section_mean_diam(s) for s in all_non_soma_sec_list])) \
              if all_non_soma_sec_list else 1.0

# Print a diagnostic message/value so the run can be checked while executing.
print(f"\nMedian non-soma section diameter: {median_diam:.3f} um")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"Soma direct children (axon candidates): {len(soma_children)}")

# Score each candidate: primary key = unbranched trunk length (longer = more axon-like),
# secondary key = thin diameter (below median = more axon-like)
# Store `best_axon_root` so later simulation, analysis, or plotting code can reuse it.
best_axon_root  = None
# Store `best_trunk_len` so later simulation, analysis, or plotting code can reuse it.
best_trunk_len  = -1.0

# Iterate through every relevant item so none of the morphology/simulation data is skipped.
for cand in soma_children:
    # Store `trunk_len` so later simulation, analysis, or plotting code can reuse it.
    trunk_len  = _unbranched_trunk_length(cand)
    # Store `mean_diam` so later simulation, analysis, or plotting code can reuse it.
    mean_diam  = _section_mean_diam(cand)
    # Store `is_thin` so later simulation, analysis, or plotting code can reuse it.
    is_thin    = mean_diam < median_diam
    # Prefer the longest unbranched trunk; among ties prefer the thinner one.
    # We give a 20 % bonus to thin candidates so thinness can break close ties
    # but a grossly shorter thin section won't beat a clearly longer thick one.
    # Store `effective_len` so later simulation, analysis, or plotting code can reuse it.
    effective_len = trunk_len * (1.20 if is_thin else 1.0)
    # Print a diagnostic message/value so the run can be checked while executing.
    print(f"  Candidate {cand.name():30s}  trunk={trunk_len:8.1f} um  "
          # Store `f"diam` so later simulation, analysis, or plotting code can reuse it.
          f"diam={mean_diam:.3f} um  thin={is_thin}  eff={effective_len:.1f}")
    # Test this condition so the script handles this case safely/correctly.
    if effective_len > best_trunk_len:
        # Store `best_trunk_len` so later simulation, analysis, or plotting code can reuse it.
        best_trunk_len = effective_len
        # Store `best_axon_root` so later simulation, analysis, or plotting code can reuse it.
        best_axon_root = cand

# Collect the entire axonal tree (trunk + all collaterals branching off it)
# Test this condition so the script handles this case safely/correctly.
if best_axon_root is not None:
    # Store `axon_sections` so later simulation, analysis, or plotting code can reuse it.
    axon_sections = _all_descendant_sections(best_axon_root)
# Use this fallback when the earlier condition is not true.
else:
    # Store `axon_sections` so later simulation, analysis, or plotting code can reuse it.
    axon_sections = set()

# Print a diagnostic message/value so the run can be checked while executing.
print(f"\nAxon root section: {best_axon_root.name() if best_axon_root else 'None'}")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"Total axon sections: {len(axon_sections)}")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"Axon unbranched trunk length: {_unbranched_trunk_length(best_axon_root):.1f} um"
      # Test this condition so the script handles this case safely/correctly.
      if best_axon_root else "")

# Build a per-segment axon flag dict for fast lookup later
# Key: segment key string → True/False
# Store `seg_is_axon` so later simulation, analysis, or plotting code can reuse it.
seg_is_axon = {}
# Iterate through every relevant item so none of the morphology/simulation data is skipped.
for sec in h.allsec():
    # Store `is_ax` so later simulation, analysis, or plotting code can reuse it.
    is_ax = sec in axon_sections
    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for seg in sec:
        # Store `key` so later simulation, analysis, or plotting code can reuse it.
        key = f"{sec.name()}({seg.x:.3f})"
        # Store `seg_is_axon[key]` so later simulation, analysis, or plotting code can reuse it.
        seg_is_axon[key] = is_ax

# ── Stim site: most distal NON-SOMA, NON-AXON (dendritic) leaf ───────────────
# ── Stim site: most distal NON-SOMA, NON-AXON (dendritic) leaf ───────────────
# We want the stim site at the deepest branch order (most distal in topological
# sense) AND the longest path distance within that order.
# Strategy:
#   1. Build branch-order map NOW (before node_to_segment exists) using SWC tree only.
#   2. Among dendritic leaves, find those at the maximum branch order.
#   3. Among those, pick the one with the greatest path-length distance from soma.
#
# Note: node_to_segment is built later (it needs the simulation first), so here we
# compute a quick SWC-node-level branch order and map leaf SECTIONS to their order
# by finding the SWC node nearest to each section's distal tip.

# Define `_swc_branch_order_map` to isolate this repeated calculation/plotting step.
def _swc_branch_order_map(nodes, node_order, children):
    """BFS branch order: root=1, increments at each bifurcation."""
    # Import this package/function because a later step depends on it.
    from collections import deque
    # Store `root` so later simulation, analysis, or plotting code can reuse it.
    root = next(nid for nid in node_order if nodes[nid]['parent_id'] == -1)
    # Store `order` so later simulation, analysis, or plotting code can reuse it.
    order = {}
    # Store `q` so later simulation, analysis, or plotting code can reuse it.
    q = deque([(root, 1)])
    # Keep traversing/searching until the morphology path reaches its stopping condition.
    while q:
        # Store `nid, lvl` so later simulation, analysis, or plotting code can reuse it.
        nid, lvl = q.popleft()
        # Store `order[nid]` so later simulation, analysis, or plotting code can reuse it.
        order[nid] = lvl
        # Store `kids` so later simulation, analysis, or plotting code can reuse it.
        kids = children[nid]
        # Store `nxt` so later simulation, analysis, or plotting code can reuse it.
        nxt  = lvl + 1 if len(kids) >= 2 else lvl
        # Iterate through every relevant item so none of the morphology/simulation data is skipped.
        for child in kids:
            # Test this condition so the script handles this case safely/correctly.
            if child not in order:
                # Append this result to a list so it is included in later processing.
                q.append((child, nxt))
    # Return the final value(s) produced by this helper function.
    return order

# Store `_swc_order` so later simulation, analysis, or plotting code can reuse it.
_swc_order = _swc_branch_order_map(nodes, node_order, children)

# Define `_section_distal_swc_order` to isolate this repeated calculation/plotting step.
def _section_distal_swc_order(sec, swc_order, nodes):
    """
    # Execute this statement as part of the current analysis step.
    Return the branch order of the SWC node geometrically nearest to the
    # Execute this statement as part of the current analysis step.
    distal tip (last pt3d point) of a NEURON section.
    """
    # Store `n3d_n` so later simulation, analysis, or plotting code can reuse it.
    n3d_n = sec.n3d()
    # Test this condition so the script handles this case safely/correctly.
    if n3d_n == 0:
        # Return the final value(s) produced by this helper function.
        return 1
    # Store `dx, dy, dz` so later simulation, analysis, or plotting code can reuse it.
    dx, dy, dz = sec.x3d(n3d_n - 1), sec.y3d(n3d_n - 1), sec.z3d(n3d_n - 1)
    # Store `best_dist` so later simulation, analysis, or plotting code can reuse it.
    best_dist = float('inf')
    # Store `best_order` so later simulation, analysis, or plotting code can reuse it.
    best_order = 1
    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for nid, nd in nodes.items():
        # Store `d` so later simulation, analysis, or plotting code can reuse it.
        d = (nd['x'] - dx)**2 + (nd['y'] - dy)**2 + (nd['z'] - dz)**2
        # Test this condition so the script handles this case safely/correctly.
        if d < best_dist:
            # Store `best_dist` so later simulation, analysis, or plotting code can reuse it.
            best_dist  = d
            # Store `best_order` so later simulation, analysis, or plotting code can reuse it.
            best_order = swc_order.get(nid, 1)
    # Return the final value(s) produced by this helper function.
    return best_order

# Store `dendritic_leaves` so later simulation, analysis, or plotting code can reuse it.
dendritic_leaves = [sec for sec in leaf_sections
                    # Test this condition so the script handles this case safely/correctly.
                    if sec in non_soma_sections and sec not in axon_sections]
# Test this condition so the script handles this case safely/correctly.
if not dendritic_leaves:
    # Store `dendritic_leaves` so later simulation, analysis, or plotting code can reuse it.
    dendritic_leaves = [sec for sec in leaf_sections if sec not in axon_sections]
# Test this condition so the script handles this case safely/correctly.
if not dendritic_leaves:
    # Store `dendritic_leaves` so later simulation, analysis, or plotting code can reuse it.
    dendritic_leaves = leaf_sections   # last-resort fallback

# Annotate each dendritic leaf with (path_distance, branch_order)
# Store `leaf_info` so later simulation, analysis, or plotting code can reuse it.
leaf_info = [
    # Execute this statement as part of the current analysis step.
    (path_distance_from_soma(sec, soma_secs),
     # Call this function/method to execute the next operation in the pipeline.
     _section_distal_swc_order(sec, _swc_order, nodes),
     # Execute this statement as part of the current analysis step.
     sec)
    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for sec in dendritic_leaves
]

# Find the maximum branch order among all dendritic leaves
# Store `max_dendritic_order` so later simulation, analysis, or plotting code can reuse it.
max_dendritic_order = max(bo for _, bo, _ in leaf_info)
# Print a diagnostic message/value so the run can be checked while executing.
print(f"\nMax dendritic branch order among leaves: {max_dendritic_order}")

# Keep only leaves AT the maximum branch order, then pick longest path distance
# Store `deepest_leaves` so later simulation, analysis, or plotting code can reuse it.
deepest_leaves = [(pd, bo, sec) for pd, bo, sec in leaf_info if bo == max_dendritic_order]
# Test this condition so the script handles this case safely/correctly.
if not deepest_leaves:
    # Store `deepest_leaves` so later simulation, analysis, or plotting code can reuse it.
    deepest_leaves = leaf_info   # fallback: shouldn't happen

# Sort these values so ranking/order-based selection is deterministic.
deepest_leaves.sort(key=lambda x: x[0])   # sort ascending by path distance
# Store `max_dist, _, stim_section` so later simulation, analysis, or plotting code can reuse it.
max_dist, _, stim_section = deepest_leaves[-1]  # most distal among deepest

# Full ranked list for the diagnostic printout (all non-soma leaves)
# Sort these values so ranking/order-based selection is deterministic.
all_leaf_distances = sorted(
    # Execute this statement as part of the current analysis step.
    [(path_distance_from_soma(sec, soma_secs), sec) for sec in
     # Execute this statement as part of the current analysis step.
     ([sec for sec in leaf_sections if sec in non_soma_sections] or leaf_sections)],
    # Store `key` so later simulation, analysis, or plotting code can reuse it.
    key=lambda x: x[0]
)

# Store `n3d` so later simulation, analysis, or plotting code can reuse it.
n3d = stim_section.n3d()
# Store `distal_idx` so later simulation, analysis, or plotting code can reuse it.
distal_idx  = n3d - 1
# Convert data into a NumPy array for vectorized numerical operations.
distal_xyz  = np.array([stim_section.x3d(distal_idx),
                        # Call this function/method to execute the next operation in the pipeline.
                        stim_section.y3d(distal_idx),
                        # Call this function/method to execute the next operation in the pipeline.
                        stim_section.z3d(distal_idx)])
# Store `total_arc` so later simulation, analysis, or plotting code can reuse it.
total_arc   = stim_section.arc3d(n3d - 1)
# Convert data into a NumPy array for vectorized numerical operations.
arc_fracs   = np.array([stim_section.arc3d(i) / total_arc for i in range(n3d)])

# Store `best_seg` so later simulation, analysis, or plotting code can reuse it.
best_seg      = None
# Store `best_dist_seg` so later simulation, analysis, or plotting code can reuse it.
best_dist_seg = float('inf')
# Iterate through every relevant item so none of the morphology/simulation data is skipped.
for seg in stim_section:
    # Find the closest index/value match.
    idx     = int(np.argmin(np.abs(arc_fracs - seg.x)))
    # Convert data into a NumPy array for vectorized numerical operations.
    seg_xyz = np.array([stim_section.x3d(idx),
                        # Call this function/method to execute the next operation in the pipeline.
                        stim_section.y3d(idx),
                        # Call this function/method to execute the next operation in the pipeline.
                        stim_section.z3d(idx)])
    # Compute straight-line distance between two coordinate points.
    d = np.linalg.norm(seg_xyz - distal_xyz)
    # Test this condition so the script handles this case safely/correctly.
    if d < best_dist_seg:
        # Store `best_dist_seg` so later simulation, analysis, or plotting code can reuse it.
        best_dist_seg = d
        # Store `best_seg` so later simulation, analysis, or plotting code can reuse it.
        best_seg      = seg

# Store the NEURON segment closest to the distal tip of the selected stimulation section.
stim_seg = best_seg
# Store the normalized location along the stimulation section.
stim_x   = float(stim_seg.x)
# Store the stimulated segment key so all metrics can compare against it.
stim_key = f"{stim_section.name()}({stim_x:.3f})"

# Store `is_stim_axon` so later simulation, analysis, or plotting code can reuse it.
is_stim_axon = stim_section in axon_sections
# Print a diagnostic message/value so the run can be checked while executing.
print(f"\nStim site : {stim_section.name()}({stim_x:.3f})")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"Path dist : {max_dist:.2f} um from soma")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"Is axon?  : {is_stim_axon}  ← should be False")

# Print ALL non-soma leaves ranked by path distance; flag axon and chosen stim site
# Print a diagnostic message/value so the run can be checked while executing.
print("\nAll non-soma leaf sections ranked by path distance:")
# Iterate through every relevant item so none of the morphology/simulation data is skipped.
for rank, (dist, sec) in enumerate(reversed(all_leaf_distances)):
    # Store `axon_tag` so later simulation, analysis, or plotting code can reuse it.
    axon_tag = " [AXON]"   if sec in axon_sections else ""
    # Store `stim_tag` so later simulation, analysis, or plotting code can reuse it.
    stim_tag = " <-- STIM" if sec is stim_section  else ""
    # Print a diagnostic message/value so the run can be checked while executing.
    print(f"  rank {rank+1:>2}  dist={dist:>10.1f} um  {sec.name()}{axon_tag}{stim_tag}")

# ===================================================================================================
# STEP 6 — Simulation function
# ===================================================================================================
# Set total simulation duration in ms.
T_STOP = 5000.0
# Set NEURON time step in ms for numerical integration.
DT     = 0.025
# Set current-clamp amplitude in nA.
I_AMP  = 0.002

# Define `run_sim` to isolate this repeated calculation/plotting step.
def run_sim(stim_dur):
    # Store `h.dt` so later simulation, analysis, or plotting code can reuse it.
    h.dt     = DT
    # Store `h.tstop` so later simulation, analysis, or plotting code can reuse it.
    h.tstop  = T_STOP
    # Store `h.v_init` so later simulation, analysis, or plotting code can reuse it.
    h.v_init = -65

    # Place a current clamp at the selected stimulation segment.
    stim       = h.IClamp(stim_section(stim_x))
    # Set when the current injection begins.
    stim.delay = 100
    # Set how long the current injection lasts.
    stim.dur   = stim_dur
    # Set the current injection amplitude.
    stim.amp   = I_AMP

    # Record this NEURON variable over time for later analysis.
    t_vec = h.Vector().record(h._ref_t)

    # Store `v_recs` so later simulation, analysis, or plotting code can reuse it.
    v_recs = {}
    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for sec in h.allsec():
        # Iterate through every relevant item so none of the morphology/simulation data is skipped.
        for seg in sec:
            # Store `key` so later simulation, analysis, or plotting code can reuse it.
            key = f"{sec.name()}({seg.x:.3f})"
            # Record this NEURON variable over time for later analysis.
            v_recs[key] = h.Vector().record(seg._ref_v)

    # Initialize all segment voltages before starting the simulation.
    h.finitialize(-65)
    # Run the NEURON simulation using the configured time step and stop time.
    h.run()

    # Convert data into a NumPy array for vectorized numerical operations.
    t = np.array(t_vec)
    # Convert data into a NumPy array for vectorized numerical operations.
    v = {k: np.array(v_recs[k]) for k in v_recs}
    # Return the final value(s) produced by this helper function.
    return t, v

# ===================================================================================================
# STEP 7 — Run simulations
# ===================================================================================================
# Print a diagnostic message/value so the run can be checked while executing.
print("\nRunning simulations...")
# Store `t1, v1` so later simulation, analysis, or plotting code can reuse it.
t1, v1 = run_sim(1.0)
# Store `t2, v2` so later simulation, analysis, or plotting code can reuse it.
t2, v2 = run_sim(1000.0)

# =====================================================================
# Build ordered segment key lists
# =====================================================================
# Store `soma_segment_keys` so later simulation, analysis, or plotting code can reuse it.
soma_segment_keys     = [f"{sec.name()}({seg.x:.3f})" for sec, seg in soma_segments]
# Store `non_soma_segment_keys` so later simulation, analysis, or plotting code can reuse it.
non_soma_segment_keys = [f"{sec.name()}({seg.x:.3f})" for sec, seg in non_soma_segments]

# Store `soma_index` so later simulation, analysis, or plotting code can reuse it.
soma_index     = {k: i + 1 for i, k in enumerate(soma_segment_keys)}
# Store `non_soma_index` so later simulation, analysis, or plotting code can reuse it.
non_soma_index = {k: i + 1 for i, k in enumerate(non_soma_segment_keys)}

# Iterate through every relevant item so none of the morphology/simulation data is skipped.
for k in soma_segment_keys:
    # Store `segment_labels[k]` so later simulation, analysis, or plotting code can reuse it.
    segment_labels[k] = f"Soma segment {soma_index[k]}"
# Iterate through every relevant item so none of the morphology/simulation data is skipped.
for k in non_soma_segment_keys:
    # Store `segment_labels[k]` so later simulation, analysis, or plotting code can reuse it.
    segment_labels[k] = f"Non-soma segment {non_soma_index[k]}"

# Store `all_segment_keys` so later simulation, analysis, or plotting code can reuse it.
all_segment_keys = soma_segment_keys + non_soma_segment_keys

# ===================================================================================================
# STEP 8 — PEAK-RATIO METRIC
# ===================================================================================================

# Define `compute_peak_ratio` to isolate this repeated calculation/plotting step.
def compute_peak_ratio(v_dict, reference_key, all_keys, t, t_start=100.0, t_end=300.0):
    # Store `mask` so later simulation, analysis, or plotting code can reuse it.
    mask     = (t >= t_start) & (t <= t_end)
    # Store `ref_full` so later simulation, analysis, or plotting code can reuse it.
    ref_full = v_dict[reference_key]
    # Compute the mean used as a baseline or summary value.
    ref_base = np.mean(ref_full[t < t_start])
    # Compute the maximum used for peak detection or normalization.
    ref_peak = np.max(ref_full[mask] - ref_base)

    # Store `ratio_values` so later simulation, analysis, or plotting code can reuse it.
    ratio_values = {}
    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for key in all_keys:
        # Test this condition so the script handles this case safely/correctly.
        if key not in v_dict or ref_peak <= 0:
            # Store `ratio_values[key]` so later simulation, analysis, or plotting code can reuse it.
            ratio_values[key] = 0.0
            # Skip this item because it is invalid or not needed for the current calculation.
            continue
        # Store `x_full` so later simulation, analysis, or plotting code can reuse it.
        x_full = v_dict[key]
        # Compute the mean used as a baseline or summary value.
        x_base = np.mean(x_full[t < t_start])
        # Compute the maximum used for peak detection or normalization.
        x_peak = np.max(x_full[mask] - x_base)
        # Restrict the ratio to the valid 0-to-1 range.
        ratio_values[key] = float(np.clip(x_peak / ref_peak, 0.0, 1.0))
    # Return the final value(s) produced by this helper function.
    return ratio_values


# Define `compute_time_to_max_and_fwhm` to isolate this repeated calculation/plotting step.
def compute_time_to_max_and_fwhm(v_dict, all_keys, t, t_start=100.0, t_end=300.0):
    # Store `mask` so later simulation, analysis, or plotting code can reuse it.
    mask = (t >= t_start) & (t <= t_end)
    # Store `time_to_max` so later simulation, analysis, or plotting code can reuse it.
    time_to_max = {}
    # Store `fwhm` so later simulation, analysis, or plotting code can reuse it.
    fwhm = {}

    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for key in all_keys:
        # Test this condition so the script handles this case safely/correctly.
        if key not in v_dict:
            # Store `time_to_max[key]` so later simulation, analysis, or plotting code can reuse it.
            time_to_max[key] = np.nan
            # Store `fwhm[key]` so later simulation, analysis, or plotting code can reuse it.
            fwhm[key]        = np.nan
            # Skip this item because it is invalid or not needed for the current calculation.
            continue

        # Store `trace_full` so later simulation, analysis, or plotting code can reuse it.
        trace_full = v_dict[key]
        # Compute the mean used as a baseline or summary value.
        baseline   = np.mean(trace_full[t < t_start])
        # Store `trace` so later simulation, analysis, or plotting code can reuse it.
        trace      = trace_full[mask] - baseline
        # Store `tt` so later simulation, analysis, or plotting code can reuse it.
        tt         = t[mask]

        # Test this condition so the script handles this case safely/correctly.
        if len(trace) == 0:
            # Store `time_to_max[key]` so later simulation, analysis, or plotting code can reuse it.
            time_to_max[key] = np.nan
            # Store `fwhm[key]` so later simulation, analysis, or plotting code can reuse it.
            fwhm[key]        = np.nan
            # Skip this item because it is invalid or not needed for the current calculation.
            continue

        # Find where the trace reaches its maximum response.
        peak_idx = int(np.argmax(trace))
        # Store `peak_val` so later simulation, analysis, or plotting code can reuse it.
        peak_val = float(trace[peak_idx])

        # Test this condition so the script handles this case safely/correctly.
        if peak_val <= 0:
            # Store `time_to_max[key]` so later simulation, analysis, or plotting code can reuse it.
            time_to_max[key] = np.nan
            # Store `fwhm[key]` so later simulation, analysis, or plotting code can reuse it.
            fwhm[key]        = np.nan
            # Skip this item because it is invalid or not needed for the current calculation.
            continue

        # Store `time_to_max[key]` so later simulation, analysis, or plotting code can reuse it.
        time_to_max[key] = float(tt[peak_idx] - t_start)

        # Store `half_val` so later simulation, analysis, or plotting code can reuse it.
        half_val = 0.5 * peak_val
        # Store `above` so later simulation, analysis, or plotting code can reuse it.
        above    = np.where(trace >= half_val)[0]

        # Test this condition so the script handles this case safely/correctly.
        if len(above) == 0:
            # Store `fwhm[key]` so later simulation, analysis, or plotting code can reuse it.
            fwhm[key] = np.nan
        # Use this fallback when the earlier condition is not true.
        else:
            # Store `fwhm[key]` so later simulation, analysis, or plotting code can reuse it.
            fwhm[key] = float(tt[above[-1]] - tt[above[0]])

    # Return the final value(s) produced by this helper function.
    return time_to_max, fwhm


# FIX #2 / #3: plot_ttm_fwhm_scatter and filter_segments_by_shape defined ONCE here.
# The duplicate definitions that appeared in Step 8d and again at the top of Step 9
# have been removed. Only one canonical copy of each function exists below.
# =====================================================================
# Compute space constant (lambda) from attenuation vs distance
# =====================================================================
# Define `compute_space_constant` to isolate this repeated calculation/plotting step.
def compute_space_constant(corr_values, distances, keys):
    """
    # Execute this statement as part of the current analysis step.
    Estimate space constant λ using:
        # Store `log(V/V0)` so later simulation, analysis, or plotting code can reuse it.
        log(V/V0) = -x / λ
    """

    # Import this package/function because a later step depends on it.
    import numpy as np

    # Store `xs` so later simulation, analysis, or plotting code can reuse it.
    xs = []
    # Store `ys` so later simulation, analysis, or plotting code can reuse it.
    ys = []

    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for k in keys:
        # Test this condition so the script handles this case safely/correctly.
        if k not in corr_values or k not in distances:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue

        # Store `r` so later simulation, analysis, or plotting code can reuse it.
        r = corr_values[k]
        # Store `d` so later simulation, analysis, or plotting code can reuse it.
        d = distances[k]

        # Skip invalid values
        # Test this condition so the script handles this case safely/correctly.
        if r <= 0 or r >= 1.0:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue

        # Append this result to a list so it is included in later processing.
        xs.append(d)
        # Append this result to a list so it is included in later processing.
        ys.append(np.log(r))

    # Convert data into a NumPy array for vectorized numerical operations.
    xs = np.array(xs)
    # Convert data into a NumPy array for vectorized numerical operations.
    ys = np.array(ys)

    # Test this condition so the script handles this case safely/correctly.
    if len(xs) < 5:
        # Print a diagnostic message/value so the run can be checked while executing.
        print("Not enough points to estimate lambda.")
        # Return the final value(s) produced by this helper function.
        return None

    # Linear fit
    # Fit a line to estimate the slope used for the space constant calculation.
    slope, intercept = np.polyfit(xs, ys, 1)

    # λ = -1 / slope
    # Store `lambda_um` so later simulation, analysis, or plotting code can reuse it.
    lambda_um = -1.0 / slope

    # Return the final value(s) produced by this helper function.
    return lambda_um

# Define `plot_ttm_fwhm_vs_distance` to isolate this repeated calculation/plotting step.
def plot_ttm_fwhm_vs_distance(axes, ttm, fwhm, dist_map, seg_key_to_level,
                               # Execute this statement as part of the current analysis step.
                               all_keys, run_label,
                               # Store `filtered_keys` so later simulation, analysis, or plotting code can reuse it.
                               filtered_keys=None,
                               # Store `ttm_max` so later simulation, analysis, or plotting code can reuse it.
                               ttm_max=None, fwhm_max=None, fwhm_min=None):
    """
    # Execute this statement as part of the current analysis step.
    Three scatter plots (one per row of `axes`):
      # Execute this statement as part of the current analysis step.
      Row 0 : Time-to-max (ms)  vs distance from stim site (µm)
      # Execute this statement as part of the current analysis step.
      Row 1 : FWHM (ms)         vs distance from stim site (µm)
      # Store `Row 2 : TTM vs FWHM — filter diagnostic (passed` so later simulation, analysis, or plotting code can reuse it.
      Row 2 : TTM vs FWHM — filter diagnostic (passed = coloured, failed = grey)

    # Execute this statement as part of the current analysis step.
    Rows 0–1: colour encodes branch order (plasma colormap).
    # Execute this statement as part of the current analysis step.
    Row 2: shows the filter thresholds as dashed lines so it is clear
           # Execute this statement as part of the current analysis step.
           which segments survived into clustering.
    """
    # ── Collect valid data points ────────────────────────────────────────────
    # Store `dists, ttms, fwhms, orders, keys_valid` so later simulation, analysis, or plotting code can reuse it.
    dists, ttms, fwhms, orders, keys_valid = [], [], [], [], []
    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for key in all_keys:
        # Store `d` so later simulation, analysis, or plotting code can reuse it.
        d  = dist_map.get(key, np.nan)
        # Store `tm` so later simulation, analysis, or plotting code can reuse it.
        tm = ttm.get(key, np.nan)
        # Store `fw` so later simulation, analysis, or plotting code can reuse it.
        fw = fwhm.get(key, np.nan)
        # Store `bo` so later simulation, analysis, or plotting code can reuse it.
        bo = seg_key_to_level.get(key, 1)
        # Test this condition so the script handles this case safely/correctly.
        if not (np.isfinite(d) and np.isfinite(tm) and np.isfinite(fw)):
            # Skip this item because it is invalid or not needed for the current calculation.
            continue
        # Append this result to a list so it is included in later processing.
        dists.append(d);  ttms.append(tm);  fwhms.append(fw)
        # Append this result to a list so it is included in later processing.
        orders.append(bo); keys_valid.append(key)

    # Convert data into a NumPy array for vectorized numerical operations.
    dists  = np.array(dists)
    # Convert data into a NumPy array for vectorized numerical operations.
    ttms   = np.array(ttms)
    # Convert data into a NumPy array for vectorized numerical operations.
    fwhms  = np.array(fwhms)
    # Convert data into a NumPy array for vectorized numerical operations.
    orders = np.array(orders)

    # Do nothing here; this placeholder keeps the syntax valid.
    passed = np.array([k in filtered_keys for k in keys_valid]) \
             if filtered_keys is not None else np.ones(len(keys_valid), dtype=bool)

    # Store `max_order` so later simulation, analysis, or plotting code can reuse it.
    max_order = int(orders.max()) if len(orders) else 1
    # Store `cmap` so later simulation, analysis, or plotting code can reuse it.
    cmap      = plt.cm.plasma
    # Store `norm` so later simulation, analysis, or plotting code can reuse it.
    norm      = mcolors.Normalize(vmin=1, vmax=max_order)

    # ── Row 0: TTM vs distance ────────────────────────────────────────────────
    # Store `ax0` so later simulation, analysis, or plotting code can reuse it.
    ax0 = axes[0]
    # Draw points marking segment metrics or key anatomical locations.
    sc0 = ax0.scatter(dists, ttms, c=orders, cmap=cmap, norm=norm,
                      # Store `s` so later simulation, analysis, or plotting code can reuse it.
                      s=MS_DOT_DIST, edgecolors='black', linewidths=0.4, alpha=0.85)
    # Set the panel title so the stimulus/metric being plotted is clear.
    ax0.set_title(f"Time to max vs distance from stim — {run_label}",
                  # Store `fontsize` so later simulation, analysis, or plotting code can reuse it.
                  fontsize=FONT_TITLE, fontweight='bold')
    # Label the horizontal axis with the correct variable and units.
    ax0.set_xlabel("Distance from stim site (µm)", fontsize=FONT_LABEL)
    # Label the vertical axis with the correct variable and units.
    ax0.set_ylabel("Time to max (ms)", fontsize=FONT_LABEL)
    # Store `ax0.tick_params(labelsize` so later simulation, analysis, or plotting code can reuse it.
    ax0.tick_params(labelsize=FONT_TICK)
    # Add a light grid to make values easier to read from the plot.
    ax0.grid(True, alpha=0.22)
    # Hide unnecessary plot borders for a cleaner figure style.
    ax0.spines['top'].set_visible(False)
    # Hide unnecessary plot borders for a cleaner figure style.
    ax0.spines['right'].set_visible(False)
    # Add a colorbar so the colormap’s numeric meaning is visible.
    cbar0 = plt.colorbar(sc0, ax=ax0, pad=0.02)
    # Apply a plotting/style command to format the current figure.
    cbar0.set_label("Branch order", fontsize=FONT_CBAR)
    # Apply a plotting/style command to format the current figure.
    cbar0.set_ticks(range(1, max_order + 1))
    # Apply a plotting/style command to format the current figure.
    cbar0.ax.tick_params(labelsize=FONT_CBAR)

    # ── Row 1: FWHM vs distance ───────────────────────────────────────────────
    # Store `ax1` so later simulation, analysis, or plotting code can reuse it.
    ax1 = axes[1]
    # Draw points marking segment metrics or key anatomical locations.
    sc1 = ax1.scatter(dists, fwhms, c=orders, cmap=cmap, norm=norm,
                      # Store `s` so later simulation, analysis, or plotting code can reuse it.
                      s=MS_DOT_DIST, edgecolors='black', linewidths=0.4, alpha=0.85)
    # Set the panel title so the stimulus/metric being plotted is clear.
    ax1.set_title(f"FWHM vs distance from stim — {run_label}",
                  # Store `fontsize` so later simulation, analysis, or plotting code can reuse it.
                  fontsize=FONT_TITLE, fontweight='bold')
    # Label the horizontal axis with the correct variable and units.
    ax1.set_xlabel("Distance from stim site (µm)", fontsize=FONT_LABEL)
    # Label the vertical axis with the correct variable and units.
    ax1.set_ylabel("FWHM (ms)", fontsize=FONT_LABEL)
    # Store `ax1.tick_params(labelsize` so later simulation, analysis, or plotting code can reuse it.
    ax1.tick_params(labelsize=FONT_TICK)
    # Add a light grid to make values easier to read from the plot.
    ax1.grid(True, alpha=0.22)
    # Hide unnecessary plot borders for a cleaner figure style.
    ax1.spines['top'].set_visible(False)
    # Hide unnecessary plot borders for a cleaner figure style.
    ax1.spines['right'].set_visible(False)
    # Add a colorbar so the colormap’s numeric meaning is visible.
    cbar1 = plt.colorbar(sc1, ax=ax1, pad=0.02)
    # Apply a plotting/style command to format the current figure.
    cbar1.set_label("Branch order", fontsize=FONT_CBAR)
    # Apply a plotting/style command to format the current figure.
    cbar1.set_ticks(range(1, max_order + 1))
    # Apply a plotting/style command to format the current figure.
    cbar1.ax.tick_params(labelsize=FONT_CBAR)

    # ── Row 2: TTM vs FWHM — filter diagnostic ───────────────────────────────
    # Store `ax2` so later simulation, analysis, or plotting code can reuse it.
    ax2 = axes[2]
    # Test this condition so the script handles this case safely/correctly.
    if (~passed).any():
        # Draw points marking segment metrics or key anatomical locations.
        ax2.scatter(ttms[~passed], fwhms[~passed],
                    # Store `color` so later simulation, analysis, or plotting code can reuse it.
                    color='#cccccc', s=MS_DOT_DIST - 10, edgecolors='none',
                    # Store `alpha` so later simulation, analysis, or plotting code can reuse it.
                    alpha=0.6, label='Filtered out', zorder=2)
    # Test this condition so the script handles this case safely/correctly.
    if passed.any():
        # Draw points marking segment metrics or key anatomical locations.
        sc2 = ax2.scatter(ttms[passed], fwhms[passed],
                          # Store `c` so later simulation, analysis, or plotting code can reuse it.
                          c=orders[passed], cmap=cmap, norm=norm,
                          # Store `s` so later simulation, analysis, or plotting code can reuse it.
                          s=MS_DOT_DIST, edgecolors='black', linewidths=0.4,
                          # Store `alpha` so later simulation, analysis, or plotting code can reuse it.
                          alpha=0.88, label='Passed filter', zorder=3)
        # Add a colorbar so the colormap’s numeric meaning is visible.
        cbar2 = plt.colorbar(sc2, ax=ax2, pad=0.02)
        # Apply a plotting/style command to format the current figure.
        cbar2.set_label("Branch order", fontsize=FONT_CBAR)
        # Apply a plotting/style command to format the current figure.
        cbar2.set_ticks(range(1, max_order + 1))
        # Apply a plotting/style command to format the current figure.
        cbar2.ax.tick_params(labelsize=FONT_CBAR)

    # Test this condition so the script handles this case safely/correctly.
    if ttm_max is not None:
        # Draw a vertical reference line for stimulus timing or filter thresholds.
        ax2.axvline(ttm_max, color='steelblue', linestyle='--', lw=2.0,
                    # Store `label` so later simulation, analysis, or plotting code can reuse it.
                    label=f"TTM max = {ttm_max:.0f} ms")
    # Test this condition so the script handles this case safely/correctly.
    if fwhm_max is not None:
        # Draw a horizontal reference line for baseline or filter thresholds.
        ax2.axhline(fwhm_max, color='darkorange', linestyle='--', lw=2.0,
                    # Store `label` so later simulation, analysis, or plotting code can reuse it.
                    label=f"FWHM max = {fwhm_max:.0f} ms")
    # Test this condition so the script handles this case safely/correctly.
    if fwhm_min is not None:
        # Draw a horizontal reference line for baseline or filter thresholds.
        ax2.axhline(fwhm_min, color='darkgreen', linestyle='--', lw=2.0,
                    # Store `label` so later simulation, analysis, or plotting code can reuse it.
                    label=f"FWHM min = {fwhm_min:.0f} ms")

    # Set the panel title so the stimulus/metric being plotted is clear.
    ax2.set_title(f"TTM vs FWHM — filter diagnostic — {run_label}",
                  # Store `fontsize` so later simulation, analysis, or plotting code can reuse it.
                  fontsize=FONT_TITLE, fontweight='bold')
    # Label the horizontal axis with the correct variable and units.
    ax2.set_xlabel("Time to max (ms)", fontsize=FONT_LABEL)
    # Label the vertical axis with the correct variable and units.
    ax2.set_ylabel("FWHM (ms)", fontsize=FONT_LABEL)
    # Store `ax2.tick_params(labelsize` so later simulation, analysis, or plotting code can reuse it.
    ax2.tick_params(labelsize=FONT_TICK)
    # Add a light grid to make values easier to read from the plot.
    ax2.grid(True, alpha=0.22)
    # Hide unnecessary plot borders for a cleaner figure style.
    ax2.spines['top'].set_visible(False)
    # Hide unnecessary plot borders for a cleaner figure style.
    ax2.spines['right'].set_visible(False)
    # Add a legend so the plotted colors/markers can be interpreted.
    ax2.legend(fontsize=FONT_LEGEND, loc='best')


# Define `filter_segments_by_shape` to isolate this repeated calculation/plotting step.
def filter_segments_by_shape(att_values, ttm, fwhm, all_keys,
                              # Store `ttm_max` so later simulation, analysis, or plotting code can reuse it.
                              ttm_max=None, fwhm_max=None, fwhm_min=None):
    """Keep only segments whose peak-shape satisfies TTM/FWHM criteria."""
    # Store `filtered_keys` so later simulation, analysis, or plotting code can reuse it.
    filtered_keys = []
    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for key in all_keys:
        # Test this condition so the script handles this case safely/correctly.
        if key not in att_values:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue
        # Store `ttm_val` so later simulation, analysis, or plotting code can reuse it.
        ttm_val  = ttm.get(key, np.nan)
        # Store `fwhm_val` so later simulation, analysis, or plotting code can reuse it.
        fwhm_val = fwhm.get(key, np.nan)
        # Test this condition so the script handles this case safely/correctly.
        if np.isnan(ttm_val) or np.isnan(fwhm_val):
            # Skip this item because it is invalid or not needed for the current calculation.
            continue
        # Test this condition so the script handles this case safely/correctly.
        if ttm_max  is not None and ttm_val  > ttm_max:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue
        # Test this condition so the script handles this case safely/correctly.
        if fwhm_max is not None and fwhm_val > fwhm_max:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue
        # Test this condition so the script handles this case safely/correctly.
        if fwhm_min is not None and fwhm_val < fwhm_min:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue
        # Append this result to a list so it is included in later processing.
        filtered_keys.append(key)
    # Return the final value(s) produced by this helper function.
    return filtered_keys


# Compute metrics
# Compute/store peak-ratio attenuation values for the burst stimulus.
corr1 = compute_peak_ratio(v_dict=v1, reference_key=stim_key,
                            # Store `all_keys` so later simulation, analysis, or plotting code can reuse it.
                            all_keys=all_segment_keys, t=t1,
                            # Store `t_start` so later simulation, analysis, or plotting code can reuse it.
                            t_start=100.0, t_end=250.0)
# Compute/store peak-ratio attenuation values for the sustained stimulus.
corr2 = compute_peak_ratio(v_dict=v2, reference_key=stim_key,
                            # Store `all_keys` so later simulation, analysis, or plotting code can reuse it.
                            all_keys=all_segment_keys, t=t2,
                            # Store `t_start` so later simulation, analysis, or plotting code can reuse it.
                            t_start=100.0, t_end=1100.0)

# Store `ttm1, fwhm1` so later simulation, analysis, or plotting code can reuse it.
ttm1, fwhm1 = compute_time_to_max_and_fwhm(v1, all_segment_keys, t1,
                                             # Store `t_start` so later simulation, analysis, or plotting code can reuse it.
                                             t_start=100.0, t_end=250.0)
# Store `ttm2, fwhm2` so later simulation, analysis, or plotting code can reuse it.
ttm2, fwhm2 = compute_time_to_max_and_fwhm(v2, all_segment_keys, t2,
                                             # Store `t_start` so later simulation, analysis, or plotting code can reuse it.
                                             t_start=100.0, t_end=1100.0)

# Keep burst segments whose response shape passes the chosen TTM/FWHM filter.
filtered_keys1 = filter_segments_by_shape(corr1, ttm1, fwhm1, all_segment_keys,
                                           # Store `ttm_max` so later simulation, analysis, or plotting code can reuse it.
                                           ttm_max=15.0, fwhm_max=25.0)
# Keep sustained segments whose response shape passes the chosen TTM/FWHM filter.
filtered_keys2 = filter_segments_by_shape(corr2, ttm2, fwhm2, all_segment_keys,
                                           # Store `ttm_max` so later simulation, analysis, or plotting code can reuse it.
                                           ttm_max=1000.0, fwhm_min=5.0)

# Test this condition so the script handles this case safely/correctly.
if len(filtered_keys2) == 0:
    # Print a diagnostic message/value so the run can be checked while executing.
    print("WARNING: 1 s shape filter removed all segments. Falling back to all segments.")
    # Keep sustained segments whose response shape passes the chosen TTM/FWHM filter.
    filtered_keys2 = list(all_segment_keys)

# Print a diagnostic message/value so the run can be checked while executing.
print(f"Filtered 1 ms segments: {len(filtered_keys1)}")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"Filtered 1 s segments:  {len(filtered_keys2)}")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"\nPeak ratio computed for {len(corr1)} segments (burst run)")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"Peak ratio computed for {len(corr2)} segments (sustained run)")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"  Burst     range: [{min(corr1.values()):.3f}, {max(corr1.values()):.3f}]")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"  Sustained range: [{min(corr2.values()):.3f}, {max(corr2.values()):.3f}]")

# ─── Map every SWC node to its nearest NEURON segment ────────────────────────

# Define `build_node_to_segment_map` to isolate this repeated calculation/plotting step.
def build_node_to_segment_map():
    """For every SWC node, find the NEURON segment whose 3D position is closest."""
    # Store `node_to_seg` so later simulation, analysis, or plotting code can reuse it.
    node_to_seg = {}
    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for nid in node_order:
        # Store `node` so later simulation, analysis, or plotting code can reuse it.
        node     = nodes[nid]
        # Convert data into a NumPy array for vectorized numerical operations.
        node_xyz = np.array([node['x'], node['y'], node['z']])
        # Store `best_key` so later simulation, analysis, or plotting code can reuse it.
        best_key  = None
        # Store `best_dist` so later simulation, analysis, or plotting code can reuse it.
        best_dist = float('inf')
        # Iterate through every relevant item so none of the morphology/simulation data is skipped.
        for sec in h.allsec():
            # Store `n3d_n` so later simulation, analysis, or plotting code can reuse it.
            n3d_n = sec.n3d()
            # Test this condition so the script handles this case safely/correctly.
            if n3d_n == 0:
                # Skip this item because it is invalid or not needed for the current calculation.
                continue
            # Store `ta` so later simulation, analysis, or plotting code can reuse it.
            ta = sec.arc3d(n3d_n - 1)
            # Test this condition so the script handles this case safely/correctly.
            if ta == 0:
                # Skip this item because it is invalid or not needed for the current calculation.
                continue
            # Convert data into a NumPy array for vectorized numerical operations.
            af = np.array([sec.arc3d(i) / ta for i in range(n3d_n)])
            # Iterate through every relevant item so none of the morphology/simulation data is skipped.
            for seg in sec:
                # Store `key` so later simulation, analysis, or plotting code can reuse it.
                key     = f"{sec.name()}({seg.x:.3f})"
                # Find the closest index/value match.
                idx     = int(np.argmin(np.abs(af - seg.x)))
                # Convert data into a NumPy array for vectorized numerical operations.
                seg_xyz = np.array([sec.x3d(idx), sec.y3d(idx), sec.z3d(idx)])
                # Compute straight-line distance between two coordinate points.
                dist    = np.linalg.norm(node_xyz - seg_xyz)
                # Test this condition so the script handles this case safely/correctly.
                if dist < best_dist:
                    # Store `best_dist` so later simulation, analysis, or plotting code can reuse it.
                    best_dist = dist
                    # Store `best_key` so later simulation, analysis, or plotting code can reuse it.
                    best_key  = key
        # Store `node_to_seg[nid]` so later simulation, analysis, or plotting code can reuse it.
        node_to_seg[nid] = best_key
    # Return the final value(s) produced by this helper function.
    return node_to_seg


# Define `choose_representative_soma_key` to isolate this repeated calculation/plotting step.
def choose_representative_soma_key(soma_segments, nodes):
    # Convert data into a NumPy array for vectorized numerical operations.
    soma_root_xyz = np.array([nodes[1]['x'], nodes[1]['y'], nodes[1]['z']])
    # Store `best_key` so later simulation, analysis, or plotting code can reuse it.
    best_key  = None
    # Store `best_dist` so later simulation, analysis, or plotting code can reuse it.
    best_dist = float('inf')
    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for sec, seg in soma_segments:
        # Store `key` so later simulation, analysis, or plotting code can reuse it.
        key   = f"{sec.name()}({seg.x:.3f})"
        # Store `n3d_n` so later simulation, analysis, or plotting code can reuse it.
        n3d_n = sec.n3d()
        # Test this condition so the script handles this case safely/correctly.
        if n3d_n == 0:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue
        # Store `ta` so later simulation, analysis, or plotting code can reuse it.
        ta = sec.arc3d(n3d_n - 1)
        # Test this condition so the script handles this case safely/correctly.
        if ta == 0:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue
        # Convert data into a NumPy array for vectorized numerical operations.
        af  = np.array([sec.arc3d(i) / ta for i in range(n3d_n)])
        # Find the closest index/value match.
        idx = int(np.argmin(np.abs(af - seg.x)))
        # Convert data into a NumPy array for vectorized numerical operations.
        seg_xyz = np.array([sec.x3d(idx), sec.y3d(idx), sec.z3d(idx)])
        # Compute straight-line distance between two coordinate points.
        d = np.linalg.norm(seg_xyz - soma_root_xyz)
        # Test this condition so the script handles this case safely/correctly.
        if d < best_dist:
            # Store `best_dist` so later simulation, analysis, or plotting code can reuse it.
            best_dist = d
            # Store `best_key` so later simulation, analysis, or plotting code can reuse it.
            best_key  = key
    # Return the final value(s) produced by this helper function.
    return best_key


# Print a diagnostic message/value so the run can be checked while executing.
print("\nBuilding node-to-segment map for morphology colouring...")
# Map every SWC node to the nearest NEURON segment for coloring morphology edges.
node_to_segment = build_node_to_segment_map()


# Define `build_segment_distance_from_stim` to isolate this repeated calculation/plotting step.
def build_segment_distance_from_stim():
    # Store `seg_xyz` so later simulation, analysis, or plotting code can reuse it.
    seg_xyz = {}
    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for sec in h.allsec():
        # Store `n3d_n` so later simulation, analysis, or plotting code can reuse it.
        n3d_n = sec.n3d()
        # Test this condition so the script handles this case safely/correctly.
        if n3d_n == 0:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue
        # Store `ta` so later simulation, analysis, or plotting code can reuse it.
        ta = sec.arc3d(n3d_n - 1)
        # Test this condition so the script handles this case safely/correctly.
        if ta == 0:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue
        # Convert data into a NumPy array for vectorized numerical operations.
        af = np.array([sec.arc3d(i) / ta for i in range(n3d_n)])
        # Iterate through every relevant item so none of the morphology/simulation data is skipped.
        for seg in sec:
            # Store `key` so later simulation, analysis, or plotting code can reuse it.
            key     = f"{sec.name()}({seg.x:.3f})"
            # Find the closest index/value match.
            idx     = int(np.argmin(np.abs(af - seg.x)))
            # Convert data into a NumPy array for vectorized numerical operations.
            seg_xyz[key] = np.array([sec.x3d(idx), sec.y3d(idx), sec.z3d(idx)])

    # Test this condition so the script handles this case safely/correctly.
    if stim_key not in seg_xyz:
        # Stop execution with a clear error message because required data is missing.
        raise ValueError(f"Stim segment {stim_key} not found in segment coordinate map.")

    # Store `stim_xyz` so later simulation, analysis, or plotting code can reuse it.
    stim_xyz = seg_xyz[stim_key]
    # Return the final value(s) produced by this helper function.
    return {key: float(np.linalg.norm(xyz - stim_xyz)) for key, xyz in seg_xyz.items()}


# Store Euclidean distance from the stimulation segment to every segment.
segment_distance_from_stim = build_segment_distance_from_stim()

# ===================================================================================================
# RESULT 1/2 — MORPHOLOGY GRADIENT PLOT
# ===================================================================================================

# Define `plot_morphology_gradient` to isolate this repeated calculation/plotting step.
def plot_morphology_gradient(nodes, node_order, node_to_segment, corr_values,
                              # Store `title` so later simulation, analysis, or plotting code can reuse it.
                              title="Morphology — peak-ratio gradient"):
    # Create the figure/axes where the next plot will be drawn.
    fig, ax = plt.subplots(figsize=FIG_GRADIENT)

    # Store `norm` so later simulation, analysis, or plotting code can reuse it.
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    # Store `cmap` so later simulation, analysis, or plotting code can reuse it.
    cmap = plt.cm.Reds

    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for nid in node_order:
        # Store `node` so later simulation, analysis, or plotting code can reuse it.
        node   = nodes[nid]
        # Store `parent` so later simulation, analysis, or plotting code can reuse it.
        parent = node['parent_id']
        # Test this condition so the script handles this case safely/correctly.
        if parent == -1:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue

        # Store `seg_key` so later simulation, analysis, or plotting code can reuse it.
        seg_key    = node_to_segment.get(nid)
        # Store `metric_val` so later simulation, analysis, or plotting code can reuse it.
        metric_val = corr_values.get(seg_key, None)
        # Test this condition so the script handles this case safely/correctly.
        if metric_val is None and seg_key and ').' in seg_key:
            # Store `seg_key` so later simulation, analysis, or plotting code can reuse it.
            seg_key    = seg_key.split(').', 1)[1]
            # Store `metric_val` so later simulation, analysis, or plotting code can reuse it.
            metric_val = corr_values.get(seg_key, None)

        # Store `x0, y0` so later simulation, analysis, or plotting code can reuse it.
        x0, y0 = nodes[parent]['x'], nodes[parent]['y']
        # Store `x1, y1` so later simulation, analysis, or plotting code can reuse it.
        x1, y1 = node['x'],          node['y']

        # Draw a line showing morphology edges, current waveform, or voltage trace data.
        ax.plot([x0, x1], [y0, y1], color='black', lw=LW_MORPH_OUT, alpha=0.7,
                # Store `solid_capstyle` so later simulation, analysis, or plotting code can reuse it.
                solid_capstyle='round', zorder=1)

        # Store `color` so later simulation, analysis, or plotting code can reuse it.
        color = cmap(norm(metric_val)) if metric_val is not None else '#c8c8c8'
        # Draw a line showing morphology edges, current waveform, or voltage trace data.
        ax.plot([x0, x1], [y0, y1], color=color, lw=LW_MORPH,
                # Store `solid_capstyle` so later simulation, analysis, or plotting code can reuse it.
                solid_capstyle='round', zorder=2)

    # Store `soma_xs` so later simulation, analysis, or plotting code can reuse it.
    soma_xs = [nodes[n]['x'] for n in nodes if n == 1 or nodes[n]['parent_id'] == 1]
    # Store `soma_ys` so later simulation, analysis, or plotting code can reuse it.
    soma_ys = [nodes[n]['y'] for n in nodes if n == 1 or nodes[n]['parent_id'] == 1]
    # Compute the mean used as a baseline or summary value.
    ax.scatter(np.mean(soma_xs), np.mean(soma_ys),
               # Store `s` so later simulation, analysis, or plotting code can reuse it.
               s=MS_SOMA, marker='o', color='gold',
               # Store `edgecolors` so later simulation, analysis, or plotting code can reuse it.
               edgecolors='black', linewidths=1.5, zorder=5, label='Soma')
    # Draw points marking segment metrics or key anatomical locations.
    ax.scatter(distal_xyz[0], distal_xyz[1],
               # Store `s` so later simulation, analysis, or plotting code can reuse it.
               s=MS_STIM, marker='*', color='black',
               # Store `edgecolors` so later simulation, analysis, or plotting code can reuse it.
               edgecolors='white', linewidths=1.0, zorder=6, label='Stim site')

    # Store `sm` so later simulation, analysis, or plotting code can reuse it.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # Call this function/method to execute the next operation in the pipeline.
    sm.set_array([])
    # Add a colorbar so the colormap’s numeric meaning is visible.
    cbar = plt.colorbar(sm, ax=ax, fraction=CBAR_FRACTION, pad=0.04)
    # Apply a plotting/style command to format the current figure.
    cbar.set_label("Peak ratio vs stim site", fontsize=FONT_CBAR)
    # Apply a plotting/style command to format the current figure.
    cbar.ax.tick_params(labelsize=FONT_CBAR)

    # Apply a plotting/style command to format the current figure.
    ax.set_aspect('equal')
    # Set the panel title so the stimulus/metric being plotted is clear.
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold')
    # Label the horizontal axis with the correct variable and units.
    ax.set_xlabel("x (µm)", fontsize=FONT_LABEL)
    # Label the vertical axis with the correct variable and units.
    ax.set_ylabel("y (µm)", fontsize=FONT_LABEL)
    # Apply a plotting/style command to format the current figure.
    ax.tick_params(labelsize=FONT_TICK)
    # More transparent legend so morphology underneath is easier to see.
    # Add a legend so the plotted colors/markers can be interpreted.
    ax.legend(
    # Store `fontsize` so later simulation, analysis, or plotting code can reuse it.
    fontsize=FONT_LEGEND,
    # Store `loc` so later simulation, analysis, or plotting code can reuse it.
    loc='upper right',
    # Store `framealpha` so later simulation, analysis, or plotting code can reuse it.
    framealpha=0.55,
    # Store `facecolor` so later simulation, analysis, or plotting code can reuse it.
    facecolor='white',
    # Store `edgecolor` so later simulation, analysis, or plotting code can reuse it.
    edgecolor='black'
)
    # Hide unnecessary plot borders for a cleaner figure style.
    ax.spines['top'].set_visible(False)
    # Hide unnecessary plot borders for a cleaner figure style.
    ax.spines['right'].set_visible(False)
    # Adjust spacing so labels and panels do not overlap.
    plt.tight_layout()
    # Save the completed figure file for poster/report use.
    plt.savefig("neuron_morphology_gradient.png", dpi=150, bbox_inches='tight')
    # Print a diagnostic message/value so the run can be checked while executing.
    print("Saved: neuron_morphology_gradient.png")
    # Display the completed figure after saving it.
    plt.show()


# Call this function/method to execute the next operation in the pipeline.
plot_morphology_gradient(nodes, node_order, node_to_segment, corr1,
                         # Store `title` so later simulation, analysis, or plotting code can reuse it.
                         title="Morphology — peak-ratio gradient (Burst 1 ms)")
# Call this function/method to execute the next operation in the pipeline.
plot_morphology_gradient(nodes, node_order, node_to_segment, corr2,
                         # Store `title` so later simulation, analysis, or plotting code can reuse it.
                         title="Morphology — peak-ratio gradient (Sustained 1 s)")

# ===================================================================================================
# STEP 8b — SELECT 5 REPRESENTATIVE TRACES
# ===================================================================================================

# Print a diagnostic message/value so the run can be checked while executing.
print("\n--- SOMA DEBUG ---")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"soma_segment_keys[0] = {soma_segment_keys[0] if soma_segment_keys else 'NONE'}")
# Print a diagnostic message/value so the run can be checked while executing.
print("node_to_segment for nodes 1-5:")
# Iterate through every relevant item so none of the morphology/simulation data is skipped.
for nid in [1, 2, 3, 4, 5]:
    # Test this condition so the script handles this case safely/correctly.
    if nid in node_to_segment:
        # Print a diagnostic message/value so the run can be checked while executing.
        print(f"  node {nid} -> {node_to_segment[nid]}")
# Print a diagnostic message/value so the run can be checked while executing.
print("---")


# Define `select_five_traces` to isolate this repeated calculation/plotting step.
def select_five_traces(corr_values, stim_key, soma_segment_keys,
                       # Execute this statement as part of the current analysis step.
                       non_soma_segment_keys, soma_index, non_soma_index,
                       # Execute this statement as part of the current analysis step.
                       soma_segments, nodes):
    # Exclude axon segments — traces should represent dendritic compartments only
    # Store `candidates` so later simulation, analysis, or plotting code can reuse it.
    candidates  = [k for k in non_soma_segment_keys
                   # Test this condition so the script handles this case safely/correctly.
                   if k != stim_key and not seg_is_axon.get(k, False)]
    # Test this condition so the script handles this case safely/correctly.
    if not candidates:   # fallback if everything is somehow axon-labelled
        # Store `candidates` so later simulation, analysis, or plotting code can reuse it.
        candidates = [k for k in non_soma_segment_keys if k != stim_key]
    # Convert data into a NumPy array for vectorized numerical operations.
    r_vals      = np.array([corr_values.get(k, 0.0) for k in candidates])
    # Store `sorted_idx` so later simulation, analysis, or plotting code can reuse it.
    sorted_idx  = np.argsort(r_vals)
    # Store `lowest_key` so later simulation, analysis, or plotting code can reuse it.
    lowest_key  = candidates[sorted_idx[0]]
    # Store `median_key` so later simulation, analysis, or plotting code can reuse it.
    median_key  = candidates[sorted_idx[len(sorted_idx) // 2]]
    # Store `highest_key` so later simulation, analysis, or plotting code can reuse it.
    highest_key = candidates[sorted_idx[-1]]
    # Store `soma_key` so later simulation, analysis, or plotting code can reuse it.
    soma_key    = choose_representative_soma_key(soma_segments, nodes) if soma_segments else None

    # Store `keys` so later simulation, analysis, or plotting code can reuse it.
    keys = [stim_key, soma_key, lowest_key, median_key, highest_key]

    # Define `seg_label` to isolate this repeated calculation/plotting step.
    def seg_label(key):
        # Test this condition so the script handles this case safely/correctly.
        if key == stim_key:
            # Test this condition so the script handles this case safely/correctly.
            if key in soma_index:
                # Return the final value(s) produced by this helper function.
                return f"Stim (S{soma_index[key]})"
            # Check this alternate case only after the previous condition failed.
            elif key in non_soma_index:
                # Return the final value(s) produced by this helper function.
                return f"Stim ({non_soma_index[key]})"
            # Return the final value(s) produced by this helper function.
            return "Stim"
        # Test this condition so the script handles this case safely/correctly.
        if key is None:
            # Return the final value(s) produced by this helper function.
            return "Soma"
        # Test this condition so the script handles this case safely/correctly.
        if key in soma_index:
            # Return the final value(s) produced by this helper function.
            return f"Soma (S{soma_index[key]})"
        # Check this alternate case only after the previous condition failed.
        elif key in non_soma_index:
            # Return the final value(s) produced by this helper function.
            return f"Segment {non_soma_index[key]}"
        # Return the final value(s) produced by this helper function.
        return "Segment"

    # Store `labels` so later simulation, analysis, or plotting code can reuse it.
    labels = [seg_label(stim_key), seg_label(soma_key),
              # Call this function/method to execute the next operation in the pipeline.
              seg_label(lowest_key), seg_label(median_key), seg_label(highest_key)]
    # Store `colors` so later simulation, analysis, or plotting code can reuse it.
    colors = ['black', 'crimson', 'steelblue', 'darkorange', 'mediumseagreen']
    # Return the final value(s) produced by this helper function.
    return keys, labels, colors


# Execute this statement as part of the current analysis step.
keys1, labels1, trace_colors = select_five_traces(
    # Execute this statement as part of the current analysis step.
    corr1, stim_key, soma_segment_keys, non_soma_segment_keys,
    # Execute this statement as part of the current analysis step.
    soma_index, non_soma_index, soma_segments, nodes)

# Execute this statement as part of the current analysis step.
keys2, labels2, _ = select_five_traces(
    # Execute this statement as part of the current analysis step.
    corr2, stim_key, soma_segment_keys, non_soma_segment_keys,
    # Execute this statement as part of the current analysis step.
    soma_index, non_soma_index, soma_segments, nodes)


# Define `plot_five_traces` to isolate this repeated calculation/plotting step.
def plot_five_traces(ax, t, v, keys, labels, colors, title,
                     # Store `stim_delay_ms` so later simulation, analysis, or plotting code can reuse it.
                     stim_delay_ms=100, stim_dur_ms=1.0, xlim_s=None):
    # Store `t_s` so later simulation, analysis, or plotting code can reuse it.
    t_s   = t / 1000.0
    # Store `x_end` so later simulation, analysis, or plotting code can reuse it.
    x_end = xlim_s if xlim_s is not None else t_s[-1]

    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for key, label, color in zip(keys, labels, colors):
        # Test this condition so the script handles this case safely/correctly.
        if key is None or key not in v:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue
        # Store `mask` so later simulation, analysis, or plotting code can reuse it.
        mask = t_s <= x_end
        # Draw a line showing morphology edges, current waveform, or voltage trace data.
        ax.plot(t_s[mask], v[key][mask], color=color, lw=LW_TRACE, label=label, zorder=3)

    # Draw a vertical reference line for stimulus timing or filter thresholds.
    ax.axvline(stim_delay_ms / 1000.0, color='dimgrey', lw=1.5,
               # Store `linestyle` so later simulation, analysis, or plotting code can reuse it.
               linestyle='--', alpha=0.7, label='Stim onset')
    # Draw a horizontal reference line for baseline or filter thresholds.
    ax.axhline(-65, linestyle=':', color='k', alpha=0.25, lw=1.2)

    # Limit the visible axis range to focus on the important part of the data.
    ax.set_xlim(0, x_end)
    # Set the panel title so the stimulus/metric being plotted is clear.
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold')
    # Label the horizontal axis with the correct variable and units.
    ax.set_xlabel("Time (s)", fontsize=FONT_LABEL)
    # Label the vertical axis with the correct variable and units.
    ax.set_ylabel("Voltage (mV)", fontsize=FONT_LABEL)
    # Apply a plotting/style command to format the current figure.
    ax.tick_params(labelsize=FONT_TICK)
    # More transparent legend so traces are less obstructed.
    # Add a legend so the plotted colors/markers can be interpreted.
    ax.legend(
        # Store `fontsize` so later simulation, analysis, or plotting code can reuse it.
        fontsize=FONT_LEGEND,
        # Store `loc` so later simulation, analysis, or plotting code can reuse it.
        loc='upper right',
        # Store `framealpha` so later simulation, analysis, or plotting code can reuse it.
        framealpha=0.55,
        # Store `facecolor` so later simulation, analysis, or plotting code can reuse it.
        facecolor='white',
        # Store `edgecolor` so later simulation, analysis, or plotting code can reuse it.
        edgecolor='black'
    )
    # Hide unnecessary plot borders for a cleaner figure style.
    ax.spines['top'].set_visible(False)
    # Hide unnecessary plot borders for a cleaner figure style.
    ax.spines['right'].set_visible(False)
    # Add a light grid to make values easier to read from the plot.
    ax.grid(alpha=0.18, lw=0.5)


# ===================================================================================================
# STEP 8c — SKELETON COLOUR-CODED BY SELECTED TRACES + VOLTAGE + CURRENT PANELS
# ===================================================================================================

# Print a diagnostic message/value so the run can be checked while executing.
print("\n" + "=" * 70)
# Print a diagnostic message/value so the run can be checked while executing.
print("  SOMA SEGMENTS")
# Print a diagnostic message/value so the run can be checked while executing.
print("=" * 70)
# Iterate through every relevant item so none of the morphology/simulation data is skipped.
for k in soma_segment_keys:
    # Print a diagnostic message/value so the run can be checked while executing.
    print(f"  {soma_index[k]:>4}  {k}")

# Print a diagnostic message/value so the run can be checked while executing.
print("\n" + "=" * 70)
# Print a diagnostic message/value so the run can be checked while executing.
print("  NON-SOMA SEGMENTS  (first 30 shown)")
# Print a diagnostic message/value so the run can be checked while executing.
print("=" * 70)
# Iterate through every relevant item so none of the morphology/simulation data is skipped.
for k in non_soma_segment_keys[:30]:
    # Print a diagnostic message/value so the run can be checked while executing.
    print(f"  {non_soma_index[k]:>4}  {k}")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"\n  ... ({len(non_soma_segment_keys)} non-soma segments total)")

# Print a diagnostic message/value so the run can be checked while executing.
print("\n" + "=" * 110)
# Print a diagnostic message/value so the run can be checked while executing.
print(f"  {'#':>4}  {'Class':<9}  {'Segment key':<50}  "
      # Execute this statement as part of the current analysis step.
      f"{'x (um)':>10}  {'y (um)':>10}  {'z (um)':>10}  {'diam (um)':>10}")
# Print a diagnostic message/value so the run can be checked while executing.
print("=" * 110)


# Define `seg_centre_xyz_diam` to isolate this repeated calculation/plotting step.
def seg_centre_xyz_diam(sec, seg):
    # Store `n3d_n` so later simulation, analysis, or plotting code can reuse it.
    n3d_n = sec.n3d()
    # Test this condition so the script handles this case safely/correctly.
    if n3d_n == 0:
        # Return the final value(s) produced by this helper function.
        return 0.0, 0.0, 0.0, 0.0
    # Store `ta` so later simulation, analysis, or plotting code can reuse it.
    ta = sec.arc3d(n3d_n - 1)
    # Test this condition so the script handles this case safely/correctly.
    if ta == 0:
        # Return the final value(s) produced by this helper function.
        return 0.0, 0.0, 0.0, 0.0
    # Convert data into a NumPy array for vectorized numerical operations.
    af  = np.array([sec.arc3d(i) / ta for i in range(n3d_n)])
    # Find the closest index/value match.
    idx = int(np.argmin(np.abs(af - seg.x)))
    # Return the final value(s) produced by this helper function.
    return sec.x3d(idx), sec.y3d(idx), sec.z3d(idx), sec.diam3d(idx)


# Iterate through every relevant item so none of the morphology/simulation data is skipped.
for sec, seg in soma_segments:
    # Store `key` so later simulation, analysis, or plotting code can reuse it.
    key         = f"{sec.name()}({seg.x:.3f})"
    # Store `idx` so later simulation, analysis, or plotting code can reuse it.
    idx         = soma_index[key]
    # Store `x, y, z, d` so later simulation, analysis, or plotting code can reuse it.
    x, y, z, d = seg_centre_xyz_diam(sec, seg)
    # Print a diagnostic message/value so the run can be checked while executing.
    print(f"  {idx:>4}  {'SOMA':<9}  {key:<50}  {x:>10.1f}  {y:>10.1f}  {z:>10.1f}  {d:>10.3f}")

# Iterate through every relevant item so none of the morphology/simulation data is skipped.
for sec, seg in non_soma_segments:
    # Store `key` so later simulation, analysis, or plotting code can reuse it.
    key         = f"{sec.name()}({seg.x:.3f})"
    # Store `idx` so later simulation, analysis, or plotting code can reuse it.
    idx         = non_soma_index[key]
    # Store `x, y, z, d` so later simulation, analysis, or plotting code can reuse it.
    x, y, z, d = seg_centre_xyz_diam(sec, seg)
    # Print a diagnostic message/value so the run can be checked while executing.
    print(f"  {idx:>4}  {'NON-SOMA':<9}  {key:<50}  {x:>10.1f}  {y:>10.1f}  {z:>10.1f}  {d:>10.3f}")

# Print a diagnostic message/value so the run can be checked while executing.
print("=" * 110)
# Print a diagnostic message/value so the run can be checked while executing.
print(f"  Total soma: {len(soma_segments)}   Total non-soma: {len(non_soma_segments)}")
# Print a diagnostic message/value so the run can be checked while executing.
print("=" * 110)


# Define `build_selected_colour_map` to isolate this repeated calculation/plotting step.
def build_selected_colour_map(keys, colors):
    # Return the final value(s) produced by this helper function.
    return {k: c for k, c in zip(keys, colors) if k is not None}


# Store `colour_map1` so later simulation, analysis, or plotting code can reuse it.
colour_map1        = build_selected_colour_map(keys1, trace_colors)
# Store `colour_map2` so later simulation, analysis, or plotting code can reuse it.
colour_map2        = build_selected_colour_map(keys2, trace_colors)
# Store `colour_map_combined` so later simulation, analysis, or plotting code can reuse it.
colour_map_combined = {**colour_map1, **colour_map2}

# Store `soma_key_check` so later simulation, analysis, or plotting code can reuse it.
soma_key_check = soma_segment_keys[0] if soma_segment_keys else None
# Test this condition so the script handles this case safely/correctly.
if soma_key_check and soma_key_check not in colour_map_combined:
    # Store `colour_map_combined[soma_key_check]` so later simulation, analysis, or plotting code can reuse it.
    colour_map_combined[soma_key_check] = 'crimson'
    # Print a diagnostic message/value so the run can be checked while executing.
    print(f"  WARNING: soma key was missing from colour map — force-added: {soma_key_check}")


# Define `draw_skeleton_selected` to isolate this repeated calculation/plotting step.
def draw_skeleton_selected(ax, nodes, node_order, node_to_segment,
                            # Execute this statement as part of the current analysis step.
                            selected_colour_map, soma_node_ids,
                            # Store `stim_key, distal_xyz, title` so later simulation, analysis, or plotting code can reuse it.
                            stim_key, distal_xyz, title="Morphology skeleton"):
    # Store `GREY` so later simulation, analysis, or plotting code can reuse it.
    GREY = '#777777'

    # Store `selected_seg_xyz` so later simulation, analysis, or plotting code can reuse it.
    selected_seg_xyz = {}
    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for sec in h.allsec():
        # Store `n3d_n` so later simulation, analysis, or plotting code can reuse it.
        n3d_n = sec.n3d()
        # Test this condition so the script handles this case safely/correctly.
        if n3d_n == 0:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue
        # Store `ta` so later simulation, analysis, or plotting code can reuse it.
        ta = sec.arc3d(n3d_n - 1)
        # Test this condition so the script handles this case safely/correctly.
        if ta == 0:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue
        # Convert data into a NumPy array for vectorized numerical operations.
        af = np.array([sec.arc3d(i) / ta for i in range(n3d_n)])
        # Iterate through every relevant item so none of the morphology/simulation data is skipped.
        for seg in sec:
            # Store `key` so later simulation, analysis, or plotting code can reuse it.
            key = f"{sec.name()}({seg.x:.3f})"
            # Test this condition so the script handles this case safely/correctly.
            if key in selected_colour_map:
                # Find the closest index/value match.
                idx = int(np.argmin(np.abs(af - seg.x)))
                # Convert data into a NumPy array for vectorized numerical operations.
                selected_seg_xyz[key] = np.array([sec.x3d(idx),
                                                   # Call this function/method to execute the next operation in the pipeline.
                                                   sec.y3d(idx),
                                                   # Call this function/method to execute the next operation in the pipeline.
                                                   sec.z3d(idx)])

    # Store `edge_colours` so later simulation, analysis, or plotting code can reuse it.
    edge_colours   = {}
    # Store `keys_with_edge` so later simulation, analysis, or plotting code can reuse it.
    keys_with_edge = set()

    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for nid in node_order:
        # Store `node` so later simulation, analysis, or plotting code can reuse it.
        node   = nodes[nid]
        # Store `parent` so later simulation, analysis, or plotting code can reuse it.
        parent = node['parent_id']
        # Test this condition so the script handles this case safely/correctly.
        if parent == -1:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue
        # Store `seg_key` so later simulation, analysis, or plotting code can reuse it.
        seg_key = node_to_segment.get(nid)
        # Store `color` so later simulation, analysis, or plotting code can reuse it.
        color   = selected_colour_map.get(seg_key, GREY)
        # Store `edge_colours[(parent, nid)]` so later simulation, analysis, or plotting code can reuse it.
        edge_colours[(parent, nid)] = color
        # Test this condition so the script handles this case safely/correctly.
        if seg_key in selected_colour_map:
            # Add this item to a set for fast membership checking without duplicates.
            keys_with_edge.add(seg_key)

    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for key, seg_xyz in selected_seg_xyz.items():
        # Test this condition so the script handles this case safely/correctly.
        if key in keys_with_edge:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue
        # Store `best_nid` so later simulation, analysis, or plotting code can reuse it.
        best_nid  = None
        # Store `best_dist` so later simulation, analysis, or plotting code can reuse it.
        best_dist = float('inf')
        # Iterate through every relevant item so none of the morphology/simulation data is skipped.
        for nid in node_order:
            # Store `nd` so later simulation, analysis, or plotting code can reuse it.
            nd = nodes[nid]
            # Convert data into a NumPy array for vectorized numerical operations.
            d  = np.linalg.norm(seg_xyz - np.array([nd['x'], nd['y'], nd['z']]))
            # Test this condition so the script handles this case safely/correctly.
            if d < best_dist:
                # Store `best_dist` so later simulation, analysis, or plotting code can reuse it.
                best_dist = d
                # Store `best_nid` so later simulation, analysis, or plotting code can reuse it.
                best_nid  = nid
        # Test this condition so the script handles this case safely/correctly.
        if best_nid is not None:
            # Store `parent` so later simulation, analysis, or plotting code can reuse it.
            parent = nodes[best_nid]['parent_id']
            # Test this condition so the script handles this case safely/correctly.
            if parent != -1:
                # Store `edge_colours[(parent, best_nid)]` so later simulation, analysis, or plotting code can reuse it.
                edge_colours[(parent, best_nid)] = selected_colour_map[key]
            # Use this fallback when the earlier condition is not true.
            else:
                # Iterate through every relevant item so none of the morphology/simulation data is skipped.
                for child_nid in node_order:
                    # Test this condition so the script handles this case safely/correctly.
                    if nodes[child_nid]['parent_id'] == best_nid:
                        # Store `edge_colours[(best_nid, child_nid)]` so later simulation, analysis, or plotting code can reuse it.
                        edge_colours[(best_nid, child_nid)] = selected_colour_map[key]
                        # Exit the loop because the traversal/calculation is complete for this path.
                        break

    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for nid in node_order:
        # Store `node` so later simulation, analysis, or plotting code can reuse it.
        node   = nodes[nid]
        # Store `parent` so later simulation, analysis, or plotting code can reuse it.
        parent = node['parent_id']
        # Test this condition so the script handles this case safely/correctly.
        if parent == -1:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue
        # Store `color` so later simulation, analysis, or plotting code can reuse it.
        color = edge_colours.get((parent, nid), GREY)
        # Test this condition so the script handles this case safely/correctly.
        if color != GREY:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue
        # Store `x0, y0` so later simulation, analysis, or plotting code can reuse it.
        x0, y0 = nodes[parent]['x'], nodes[parent]['y']
        # Store `x1, y1` so later simulation, analysis, or plotting code can reuse it.
        x1, y1 = node['x'],          node['y']
        # Axon edges drawn in cyan even when not one of the 5 selected segments.
        # Uses seg_is_axon (topology-based) because SWC type codes are all 1.
        # Store `seg_key_nid` so later simulation, analysis, or plotting code can reuse it.
        seg_key_nid    = node_to_segment.get(nid)
        # Store `seg_key_parent` so later simulation, analysis, or plotting code can reuse it.
        seg_key_parent = node_to_segment.get(parent)
        # Store `is_axon_edge` so later simulation, analysis, or plotting code can reuse it.
        is_axon_edge   = (seg_is_axon.get(seg_key_nid,    False) or
                          # Call this function/method to execute the next operation in the pipeline.
                          seg_is_axon.get(seg_key_parent, False))
        # Store `draw_color` so later simulation, analysis, or plotting code can reuse it.
        draw_color   = 'cyan' if is_axon_edge else GREY
        # Store `lw` so later simulation, analysis, or plotting code can reuse it.
        lw           = LW_SKEL_AXON if is_axon_edge else LW_SKEL_GREY
        # Draw a line showing morphology edges, current waveform, or voltage trace data.
        ax.plot([x0, x1], [y0, y1], color=draw_color, lw=lw,
                # Store `solid_capstyle` so later simulation, analysis, or plotting code can reuse it.
                solid_capstyle='round', zorder=2)

    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for nid in node_order:
        # Store `node` so later simulation, analysis, or plotting code can reuse it.
        node   = nodes[nid]
        # Store `parent` so later simulation, analysis, or plotting code can reuse it.
        parent = node['parent_id']
        # Test this condition so the script handles this case safely/correctly.
        if parent == -1:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue
        # Store `color` so later simulation, analysis, or plotting code can reuse it.
        color = edge_colours.get((parent, nid), GREY)
        # Test this condition so the script handles this case safely/correctly.
        if color == GREY:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue
        # Store `x0, y0` so later simulation, analysis, or plotting code can reuse it.
        x0, y0 = nodes[parent]['x'], nodes[parent]['y']
        # Store `x1, y1` so later simulation, analysis, or plotting code can reuse it.
        x1, y1 = node['x'],          node['y']
        # Store `lw` so later simulation, analysis, or plotting code can reuse it.
        lw = LW_SKEL_SOMA if (nid in soma_node_ids or parent in soma_node_ids) else LW_SKEL_SEL
        # Draw a line showing morphology edges, current waveform, or voltage trace data.
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw,
                # Store `solid_capstyle` so later simulation, analysis, or plotting code can reuse it.
                solid_capstyle='round', zorder=5)

    # Store `stim_node` so later simulation, analysis, or plotting code can reuse it.
    stim_node      = None
    # Store `stim_node_dist` so later simulation, analysis, or plotting code can reuse it.
    stim_node_dist = float('inf')
    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for nid, seg_key in node_to_segment.items():
        # Test this condition so the script handles this case safely/correctly.
        if seg_key == stim_key:
            # Convert data into a NumPy array for vectorized numerical operations.
            candidate_xyz = np.array([nodes[nid]['x'], nodes[nid]['y'], nodes[nid]['z']])
            # Compute straight-line distance between two coordinate points.
            d = np.linalg.norm(candidate_xyz - distal_xyz)
            # Test this condition so the script handles this case safely/correctly.
            if d < stim_node_dist:
                # Store `stim_node_dist` so later simulation, analysis, or plotting code can reuse it.
                stim_node_dist = d
                # Store `stim_node` so later simulation, analysis, or plotting code can reuse it.
                stim_node      = nid

    # Test this condition so the script handles this case safely/correctly.
    if stim_node is not None:
        # Draw points marking segment metrics or key anatomical locations.
        ax.scatter(nodes[stim_node]['x'], nodes[stim_node]['y'],
                   # Store `s` so later simulation, analysis, or plotting code can reuse it.
                   s=MS_STIM, marker='*', color='black', zorder=6,
                   # Store `label` so later simulation, analysis, or plotting code can reuse it.
                   label='Stim site', edgecolors='white', linewidths=1.5)

    # Store `legend_handles` so later simulation, analysis, or plotting code can reuse it.
    legend_handles = []
    # Store `seen_colors` so later simulation, analysis, or plotting code can reuse it.
    seen_colors    = set()
    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for key, label, color in zip(keys1, labels1, trace_colors):
        # Test this condition so the script handles this case safely/correctly.
        if color in seen_colors:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue
        # Add this item to a set for fast membership checking without duplicates.
        seen_colors.add(color)
        # Append this result to a list so it is included in later processing.
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=LW_SKEL_SEL, label=label))
    # Append this result to a list so it is included in later processing.
    legend_handles.append(
        # Apply a plotting/style command to format the current figure.
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
                   # Store `markersize` so later simulation, analysis, or plotting code can reuse it.
                   markersize=18, label='Stim site'))
    # Append this result to a list so it is included in later processing.
    legend_handles.append(
        # Apply a plotting/style command to format the current figure.
        plt.Line2D([0], [0], color='cyan', lw=LW_SKEL_AXON, label='Axon (detected)'))

    # Apply a plotting/style command to format the current figure.
    ax.set_aspect('equal')
    # Set the panel title so the stimulus/metric being plotted is clear.
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold')
    # Label the horizontal axis with the correct variable and units.
    ax.set_xlabel("x (µm)", fontsize=FONT_LABEL)
    # Label the vertical axis with the correct variable and units.
    ax.set_ylabel("y (µm)", fontsize=FONT_LABEL)
    # Apply a plotting/style command to format the current figure.
    ax.tick_params(labelsize=FONT_TICK)
    # Put the morphology/tree legend in the bottom-left corner
    # and make the legend box more transparent so data behind it stays visible.
    # Add a legend so the plotted colors/markers can be interpreted.
    ax.legend(
        # Store `handles` so later simulation, analysis, or plotting code can reuse it.
        handles=legend_handles,
        # Store `fontsize` so later simulation, analysis, or plotting code can reuse it.
        fontsize=FONT_LEGEND,
        # Store `loc` so later simulation, analysis, or plotting code can reuse it.
        loc='lower right',
        # Store `framealpha` so later simulation, analysis, or plotting code can reuse it.
        framealpha=0.55,
        # Store `facecolor` so later simulation, analysis, or plotting code can reuse it.
        facecolor='white',
        # Store `edgecolor` so later simulation, analysis, or plotting code can reuse it.
        edgecolor='black'
    )
    # Hide unnecessary plot borders for a cleaner figure style.
    ax.spines['top'].set_visible(False)
    # Hide unnecessary plot borders for a cleaner figure style.
    ax.spines['right'].set_visible(False)


# Define `plot_current_waveform` to isolate this repeated calculation/plotting step.
def plot_current_waveform(ax, t, stim_delay_ms, stim_dur_ms, amp_nA, title):
    # Store `t_s` so later simulation, analysis, or plotting code can reuse it.
    t_s             = t / 1000.0
    # Store `i_wave` so later simulation, analysis, or plotting code can reuse it.
    i_wave          = np.zeros_like(t)
    # Store `on_mask` so later simulation, analysis, or plotting code can reuse it.
    on_mask         = (t >= stim_delay_ms) & (t <= stim_delay_ms + stim_dur_ms)
    # Store `i_wave[on_mask]` so later simulation, analysis, or plotting code can reuse it.
    i_wave[on_mask] = amp_nA

    # Fill the area under the current waveform to make injection timing obvious.
    ax.fill_between(t_s, i_wave, step='post',
                    # Store `color` so later simulation, analysis, or plotting code can reuse it.
                    color='steelblue', alpha=0.35, label='Current (nA)')
    # Draw a line showing morphology edges, current waveform, or voltage trace data.
    ax.plot(t_s, i_wave, color='steelblue', lw=1.5, drawstyle='steps-post')

    # Draw a vertical reference line for stimulus timing or filter thresholds.
    ax.axvline(stim_delay_ms / 1000.0, color='dimgrey',
               # Store `lw` so later simulation, analysis, or plotting code can reuse it.
               lw=1.0, linestyle='--', alpha=0.7, label='Stim onset')
    # Draw a vertical reference line for stimulus timing or filter thresholds.
    ax.axvline((stim_delay_ms + stim_dur_ms) / 1000.0, color='salmon',
               # Store `lw` so later simulation, analysis, or plotting code can reuse it.
               lw=1.0, linestyle='--', alpha=0.7, label='Stim offset')

    # Limit the visible axis range to focus on the important part of the data.
    ax.set_ylim(-amp_nA * 0.15, amp_nA * 1.30)
    # FIX #11: removed the internal ax.set_xlim(0, t_s[-1]) that was immediately
    # overridden by the caller; the caller's xlim is the authoritative one.

    # Set the panel title so the stimulus/metric being plotted is clear.
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold')
    # Label the horizontal axis with the correct variable and units.
    ax.set_xlabel("Time (s)", fontsize=FONT_LABEL)
    # Label the vertical axis with the correct variable and units.
    ax.set_ylabel("Current (nA)", fontsize=FONT_LABEL)
    # Apply a plotting/style command to format the current figure.
    ax.tick_params(labelsize=FONT_TICK)
    # More transparent legend so the current waveform remains visible behind it.
    # Add a legend so the plotted colors/markers can be interpreted.
    ax.legend(
        # Store `fontsize` so later simulation, analysis, or plotting code can reuse it.
        fontsize=FONT_LEGEND,
        # Store `loc` so later simulation, analysis, or plotting code can reuse it.
        loc='upper right',
        # Store `framealpha` so later simulation, analysis, or plotting code can reuse it.
        framealpha=0.55,
        # Store `facecolor` so later simulation, analysis, or plotting code can reuse it.
        facecolor='white',
        # Store `edgecolor` so later simulation, analysis, or plotting code can reuse it.
        edgecolor='black'
    )
    # Hide unnecessary plot borders for a cleaner figure style.
    ax.spines['top'].set_visible(False)
    # Hide unnecessary plot borders for a cleaner figure style.
    ax.spines['right'].set_visible(False)
    # Add a light grid to make values easier to read from the plot.
    ax.grid(alpha=0.18, lw=0.5)


# Create the figure/axes where the next plot will be drawn.
fig = plt.figure(figsize=FIG_COMPOSITE)
# Make the tree, current plots, and voltage plots sit closer together.
# - Slightly widen the tree panel
# - Reduce horizontal and vertical spacing between panels
# Allocate subplot space so related panels appear in one organized figure.
gs  = fig.add_gridspec(
    # Store `nrows` so later simulation, analysis, or plotting code can reuse it.
    nrows=2, ncols=3,
    # Store `width_ratios` so later simulation, analysis, or plotting code can reuse it.
    width_ratios=[1.55, 1, 1],
    # Store `height_ratios` so later simulation, analysis, or plotting code can reuse it.
    height_ratios=[1, 2.2],
    # Store `hspace` so later simulation, analysis, or plotting code can reuse it.
    hspace=0.22,
    # Store `wspace` so later simulation, analysis, or plotting code can reuse it.
    wspace=0.16
)

# Allocate subplot space so related panels appear in one organized figure.
ax_morph = fig.add_subplot(gs[:, 0])
# Allocate subplot space so related panels appear in one organized figure.
ax_curr1 = fig.add_subplot(gs[0, 1])
# Allocate subplot space so related panels appear in one organized figure.
ax_curr2 = fig.add_subplot(gs[0, 2])
# Allocate subplot space so related panels appear in one organized figure.
ax_volt1 = fig.add_subplot(gs[1, 1])
# Allocate subplot space so related panels appear in one organized figure.
ax_volt2 = fig.add_subplot(gs[1, 2])

# Call this function/method to execute the next operation in the pipeline.
draw_skeleton_selected(
    # Execute this statement as part of the current analysis step.
    ax_morph, nodes, node_order, node_to_segment,
    # Store `selected_colour_map` so later simulation, analysis, or plotting code can reuse it.
    selected_colour_map = colour_map_combined,
    # Identify SWC nodes treated as soma so soma and dendrite/axon regions can be separated.
    soma_node_ids       = soma_node_ids,
    # Store the stimulated segment key so all metrics can compare against it.
    stim_key            = stim_key,
    # Store `distal_xyz` so later simulation, analysis, or plotting code can reuse it.
    distal_xyz          = distal_xyz,
    # Store `title` so later simulation, analysis, or plotting code can reuse it.
    title               = "Morphology skeleton\n(highlighted: 5 selected segments)"
)

# Store `plot_current_waveform(ax_curr1, t1, stim_delay_ms` so later simulation, analysis, or plotting code can reuse it.
plot_current_waveform(ax_curr1, t1, stim_delay_ms=100, stim_dur_ms=1.0,
                      # Store `amp_nA` so later simulation, analysis, or plotting code can reuse it.
                      amp_nA=I_AMP, title="Current injection — Burst (1 ms)")
# Limit the visible axis range to focus on the important part of the data.
ax_curr1.set_xlim(0.08, 0.15)

# Store `plot_current_waveform(ax_curr2, t2, stim_delay_ms` so later simulation, analysis, or plotting code can reuse it.
plot_current_waveform(ax_curr2, t2, stim_delay_ms=100, stim_dur_ms=1000.0,
                      # Store `amp_nA` so later simulation, analysis, or plotting code can reuse it.
                      amp_nA=I_AMP, title="Current injection — Sustained (1 s)")
# Limit the visible axis range to focus on the important part of the data.
ax_curr2.set_xlim(0, 1.5)

# Store `plot_five_traces(ax_volt1, t1, v1, keys` so later simulation, analysis, or plotting code can reuse it.
plot_five_traces(ax_volt1, t1, v1, keys=keys1, labels=labels1, colors=trace_colors,
                 # Store `title` so later simulation, analysis, or plotting code can reuse it.
                 title="Voltage responses — Burst (1 ms)",
                 # Store `stim_delay_ms` so later simulation, analysis, or plotting code can reuse it.
                 stim_delay_ms=100, stim_dur_ms=1.0)
# Limit the visible axis range to focus on the important part of the data.
ax_volt1.set_xlim(0.08, 0.15)

# Store `plot_five_traces(ax_volt2, t2, v2, keys` so later simulation, analysis, or plotting code can reuse it.
plot_five_traces(ax_volt2, t2, v2, keys=keys2, labels=labels2, colors=trace_colors,
                 # Store `title` so later simulation, analysis, or plotting code can reuse it.
                 title="Voltage responses — Sustained (1 s)",
                 # Store `stim_delay_ms` so later simulation, analysis, or plotting code can reuse it.
                 stim_delay_ms=100, stim_dur_ms=1000.0, xlim_s=1.5)

# Apply a plotting/style command to format the current figure.
fig.suptitle(
    # Execute this statement as part of the current analysis step.
    "Neuron morphology with selected recording sites, current injection, and voltage responses",
    # Store `fontsize` so later simulation, analysis, or plotting code can reuse it.
    fontsize=FONT_SUPTITLE, fontweight='bold', y=1.01
)

# Tighten the outer figure margins so the full composite looks less spread out.
# Adjust spacing so labels and panels do not overlap.
fig.subplots_adjust(top=0.92, left=0.06, right=0.98, bottom=0.07)

# Save the completed figure file for poster/report use.
plt.savefig("neuron_morphology_selected_traces.png", dpi=150, bbox_inches='tight')
# Print a diagnostic message/value so the run can be checked while executing.
print("\nSaved: neuron_morphology_selected_traces.png")
# Display the completed figure after saving it.
plt.show()


# ===================================================================================================
# STEP 9 — TTM/FWHM VISUALIZATION, SHAPE FILTERING, CLUSTERING, FILTERED PLOTS, AND JSON EXPORT
# ===================================================================================================

# FIX #2/#3/#4/#5/#6: All duplicate function definitions that were repeated in Steps 9-13
# have been removed. The single canonical definitions above are used throughout.

# ---------------------------------------------------------------------------------------------------
# Helper: build trace feature matrix for clustering
# ---------------------------------------------------------------------------------------------------
# Define `build_trace_feature_matrix` to isolate this repeated calculation/plotting step.
def build_trace_feature_matrix(v_dict, all_keys, t, t_start, t_end):
    # Store `mask` so later simulation, analysis, or plotting code can reuse it.
    mask       = (t >= t_start) & (t <= t_end)
    # Store `X` so later simulation, analysis, or plotting code can reuse it.
    X          = []
    # Store `valid_keys` so later simulation, analysis, or plotting code can reuse it.
    valid_keys = []
    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for key in all_keys:
        # Test this condition so the script handles this case safely/correctly.
        if key not in v_dict:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue
        # Store `trace` so later simulation, analysis, or plotting code can reuse it.
        trace    = v_dict[key]
        # Compute the mean used as a baseline or summary value.
        baseline = np.mean(trace[t < t_start])
        # Append this result to a list so it is included in later processing.
        X.append(trace[mask] - baseline)
        # Append this result to a list so it is included in later processing.
        valid_keys.append(key)
    # Return the final value(s) produced by this helper function.
    return valid_keys, np.array(X)


# ---------------------------------------------------------------------------------------------------
# Helper: cluster filtered segments
# ---------------------------------------------------------------------------------------------------
# Define `cluster_segments_by_trace_similarity` to isolate this repeated calculation/plotting step.
def cluster_segments_by_trace_similarity(v_dict, all_keys, t, t_start, t_end, n_clusters=5):
    # Store `valid_keys, X` so later simulation, analysis, or plotting code can reuse it.
    valid_keys, X = build_trace_feature_matrix(v_dict, all_keys, t, t_start, t_end)

    # Test this condition so the script handles this case safely/correctly.
    if len(valid_keys) == 0 or X.size == 0:
        # Print a diagnostic message/value so the run can be checked while executing.
        print("WARNING: No valid segments available for clustering in this run.")
        # Return the final value(s) produced by this helper function.
        return {}, 0

    # Store `n_clusters` so later simulation, analysis, or plotting code can reuse it.
    n_clusters = min(n_clusters, len(valid_keys))
    # Test this condition so the script handles this case safely/correctly.
    if len(valid_keys) == 1:
        # Return the final value(s) produced by this helper function.
        return {valid_keys[0]: 0}, 1

    # Standardize traces so clustering compares shape instead of raw scale only.
    scaler = StandardScaler()
    # Store `Xs` so later simulation, analysis, or plotting code can reuse it.
    Xs     = scaler.fit_transform(X)
    # Create the K-means model used to group similar voltage responses.
    km     = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    # Run clustering and return a cluster label for each valid segment.
    labels = km.fit_predict(Xs)
    # Return the final value(s) produced by this helper function.
    return {k: int(c) for k, c in zip(valid_keys, labels)}, n_clusters


# ---------------------------------------------------------------------------------------------------
# Helper: reorder clusters so cluster 0 is most stim-like
# ---------------------------------------------------------------------------------------------------
# Define `reorder_clusters_by_similarity_to_stim` to isolate this repeated calculation/plotting step.
def reorder_clusters_by_similarity_to_stim(cluster_ids, att_values, k):
    # Test this condition so the script handles this case safely/correctly.
    if k == 0 or len(cluster_ids) == 0:
        # Return the final value(s) produced by this helper function.
        return {}
    # Store `cluster_means` so later simulation, analysis, or plotting code can reuse it.
    cluster_means = []
    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for c in range(k):
        # Store `vals` so later simulation, analysis, or plotting code can reuse it.
        vals     = [att_values[key] for key, cc in cluster_ids.items()
                    # Test this condition so the script handles this case safely/correctly.
                    if cc == c and key in att_values]
        # Compute the mean used as a baseline or summary value.
        mean_val = np.mean(vals) if vals else -np.inf
        # Append this result to a list so it is included in later processing.
        cluster_means.append((c, mean_val))
    # Sort these values so ranking/order-based selection is deterministic.
    cluster_means.sort(key=lambda x: x[1], reverse=True)
    # Store `old_to_new` so later simulation, analysis, or plotting code can reuse it.
    old_to_new = {old: new for new, (old, _) in enumerate(cluster_means)}
    # Return the final value(s) produced by this helper function.
    return {key: old_to_new[c] for key, c in cluster_ids.items()}


# ---------------------------------------------------------------------------------------------------
# Helper: red gradient for clusters
# ---------------------------------------------------------------------------------------------------
# Define `build_cluster_red_gradient` to isolate this repeated calculation/plotting step.
def build_cluster_red_gradient(k):
    # Test this condition so the script handles this case safely/correctly.
    if k <= 0:
        # Return the final value(s) produced by this helper function.
        return {}
    # Store `cmap` so later simulation, analysis, or plotting code can reuse it.
    cmap = plt.cm.Reds
    # Store `vals` so later simulation, analysis, or plotting code can reuse it.
    vals = np.linspace(0.35, 0.95, k)[::-1]
    # Return the final value(s) produced by this helper function.
    return {i: cmap(vals[i]) for i in range(k)}


# ---------------------------------------------------------------------------------------------------
# Helper: filtered voltage plot
# ---------------------------------------------------------------------------------------------------
# Define `plot_voltage` to isolate this repeated calculation/plotting step.
def plot_voltage(ax, t, v, keys_to_plot, cluster_ids, cluster_colors,
                 # Execute this statement as part of the current analysis step.
                 stim_key, segment_labels, title):
    # Store `valid_keys` so later simulation, analysis, or plotting code can reuse it.
    valid_keys = [k for k in keys_to_plot if k in v and k in cluster_ids]

    # Test this condition so the script handles this case safely/correctly.
    if len(valid_keys) == 0:
        # Apply a plotting/style command to format the current figure.
        ax.text(0.5, 0.5, "No clustered segments to plot",
                # Store `ha` so later simulation, analysis, or plotting code can reuse it.
                ha='center', va='center', transform=ax.transAxes, fontsize=FONT_TICK)
        # Set the panel title so the stimulus/metric being plotted is clear.
        ax.set_title(title)
        # Label the horizontal axis with the correct variable and units.
        ax.set_xlabel("Time (s)")
        # Label the vertical axis with the correct variable and units.
        ax.set_ylabel("Voltage (mV)")
        # Add a light grid to make values easier to read from the plot.
        ax.grid(True, alpha=0.25)
        # Execute this statement as part of the current analysis step.
        return

    # Sort these values so ranking/order-based selection is deterministic.
    keys_by_cluster = sorted(valid_keys, key=lambda k: cluster_ids[k])
    # Store `cluster_handle` so later simulation, analysis, or plotting code can reuse it.
    cluster_handle  = {}

    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for key in keys_by_cluster:
        # Store `c` so later simulation, analysis, or plotting code can reuse it.
        c     = cluster_ids[key]
        # Store `color` so later simulation, analysis, or plotting code can reuse it.
        color = cluster_colors.get(c, 'lightcoral')
        # Draw a line showing morphology edges, current waveform, or voltage trace data.
        ax.plot(t / 1000.0, v[key], color=color, lw=0.6, alpha=0.7)
        # Test this condition so the script handles this case safely/correctly.
        if c not in cluster_handle:
            # Store `cluster_handle[c]` so later simulation, analysis, or plotting code can reuse it.
            cluster_handle[c] = plt.Line2D([], [], color=color, lw=2, label=f"Cluster {c}")

    # Test this condition so the script handles this case safely/correctly.
    if stim_key in v and stim_key in cluster_ids:
        # Draw a line showing morphology edges, current waveform, or voltage trace data.
        ax.plot(t / 1000.0, v[stim_key], color='black', lw=2,
                # Store `label` so later simulation, analysis, or plotting code can reuse it.
                label=f"Stimulated: {segment_labels.get(stim_key, stim_key)}")
        # Store `cluster_handle['stim']` so later simulation, analysis, or plotting code can reuse it.
        cluster_handle['stim'] = plt.Line2D([], [], color='black', lw=2, label="Stimulated")

    # Draw a horizontal reference line for baseline or filter thresholds.
    ax.axhline(-65, linestyle='--', color='k', alpha=0.3)
    # Set the panel title so the stimulus/metric being plotted is clear.
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold')
    # Label the horizontal axis with the correct variable and units.
    ax.set_xlabel("Time (s)", fontsize=FONT_LABEL)
    # Label the vertical axis with the correct variable and units.
    ax.set_ylabel("Voltage (mV)", fontsize=FONT_LABEL)
    # Apply a plotting/style command to format the current figure.
    ax.tick_params(labelsize=FONT_TICK)
    # Add a light grid to make values easier to read from the plot.
    ax.grid(True)

    # Sort these values so ranking/order-based selection is deterministic.
    handles = [cluster_handle[c] for c in sorted(c for c in cluster_handle if c != 'stim')]
    # Test this condition so the script handles this case safely/correctly.
    if 'stim' in cluster_handle:
        # Append this result to a list so it is included in later processing.
        handles.append(cluster_handle['stim'])
    # Add a legend so the plotted colors/markers can be interpreted.
    ax.legend(handles=handles, loc='upper right', fontsize=FONT_LEGEND)


# ---------------------------------------------------------------------------------------------------
# Helper: cluster similarity / distance panel
# ---------------------------------------------------------------------------------------------------
# Define `plot_cluster_similarity_distance_panel` to isolate this repeated calculation/plotting step.
def plot_cluster_similarity_distance_panel(
    # Execute this statement as part of the current analysis step.
    att_values, cluster_ids, cluster_colors, all_keys,
    # Execute this statement as part of the current analysis step.
    soma_index, non_soma_index, dist_map, k, ax, title
):
    # Test this condition so the script handles this case safely/correctly.
    if k <= 0:
        # Apply a plotting/style command to format the current figure.
        ax.text(0.5, 0.5, "No clustered segments available",
                # Store `ha` so later simulation, analysis, or plotting code can reuse it.
                ha='center', va='center', transform=ax.transAxes, fontsize=FONT_TICK)
        # Set the panel title so the stimulus/metric being plotted is clear.
        ax.set_title(title)
        # Label the horizontal axis with the correct variable and units.
        ax.set_xlabel("Cluster number")
        # Label the vertical axis with the correct variable and units.
        ax.set_ylabel("Peak ratio vs stimulated segment")
        # Add a light grid to make values easier to read from the plot.
        ax.grid(True, alpha=0.25)
        # Execute this statement as part of the current analysis step.
        return

    # Store `cluster_to_keys` so later simulation, analysis, or plotting code can reuse it.
    cluster_to_keys = {c: [] for c in range(k)}
    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for key in all_keys:
        # Store `c` so later simulation, analysis, or plotting code can reuse it.
        c = cluster_ids.get(key, None)
        # Test this condition so the script handles this case safely/correctly.
        if c is not None:
            # Append this result to a list so it is included in later processing.
            cluster_to_keys[c].append(key)

    # Store `cluster_centers` so later simulation, analysis, or plotting code can reuse it.
    cluster_centers = np.arange(k)
    # Store `local_halfwidth` so later simulation, analysis, or plotting code can reuse it.
    local_halfwidth = 0.28
    # Store `axis_y` so later simulation, analysis, or plotting code can reuse it.
    axis_y  = -0.06
    # Store `tick_y0` so later simulation, analysis, or plotting code can reuse it.
    tick_y0 = axis_y - 0.012
    # Store `tick_y1` so later simulation, analysis, or plotting code can reuse it.
    tick_y1 = axis_y + 0.012

    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for c in range(k):
        # Store `keys_c` so later simulation, analysis, or plotting code can reuse it.
        keys_c = cluster_to_keys.get(c, [])
        # Test this condition so the script handles this case safely/correctly.
        if not keys_c:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue

        # Convert data into a NumPy array for vectorized numerical operations.
        dvals = np.array([dist_map.get(key, np.nan) for key in keys_c], dtype=float)
        # Store `valid` so later simulation, analysis, or plotting code can reuse it.
        valid = np.isfinite(dvals)
        # Store `keys_c` so later simulation, analysis, or plotting code can reuse it.
        keys_c = [key for key, ok in zip(keys_c, valid) if ok]
        # Store `dvals` so later simulation, analysis, or plotting code can reuse it.
        dvals  = dvals[valid]

        # Test this condition so the script handles this case safely/correctly.
        if len(keys_c) == 0:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue

        # Compute the maximum used for peak detection or normalization.
        dmax = float(np.max(dvals))
        # Test this condition so the script handles this case safely/correctly.
        if dmax <= 0:
            # Store `dmax` so later simulation, analysis, or plotting code can reuse it.
            dmax = 1.0

        # Store `xc` so later simulation, analysis, or plotting code can reuse it.
        xc      = cluster_centers[c]
        # Store `x_left` so later simulation, analysis, or plotting code can reuse it.
        x_left  = xc - local_halfwidth
        # Store `x_right` so later simulation, analysis, or plotting code can reuse it.
        x_right = xc + local_halfwidth

        # Draw a line showing morphology edges, current waveform, or voltage trace data.
        ax.plot([x_left, x_right], [axis_y, axis_y],
                # Store `color` so later simulation, analysis, or plotting code can reuse it.
                color='black', lw=0.8, zorder=1, clip_on=False)

        # Draw ONLY 3 labels: 0, mid, max (clean and readable)
        # Iterate through every relevant item so none of the morphology/simulation data is skipped.
        for frac, d in zip([0.0, 0.5, 1.0], [0.0, 0.5 * dmax, dmax]):
            # Store `xt` so later simulation, analysis, or plotting code can reuse it.
            xt = x_left + frac * (x_right - x_left)

            # Tick mark
            # Draw a line showing morphology edges, current waveform, or voltage trace data.
            ax.plot([xt, xt], [tick_y0, tick_y1],
                    # Store `color` so later simulation, analysis, or plotting code can reuse it.
                    color='black', lw=0.9, zorder=1, clip_on=False)

            # Label (rounded, fewer digits, slightly smaller font)
            # Apply a plotting/style command to format the current figure.
            ax.text(
                # Execute this statement as part of the current analysis step.
                xt,
                # Execute this statement as part of the current analysis step.
                axis_y - 0.03,                 # push further DOWN (key fix)
                # Execute this statement as part of the current analysis step.
                f"{d:.0f}" if frac < 1.0 else f"{d:.0f} µm",                  # INTEGER → no clutter
                # Store `ha` so later simulation, analysis, or plotting code can reuse it.
                ha='center',
                # Store `va` so later simulation, analysis, or plotting code can reuse it.
                va='top',
                # Store `fontsize` so later simulation, analysis, or plotting code can reuse it.
                fontsize=FONT_TICK - 6,       # smaller text → readable
                # Store `clip_on` so later simulation, analysis, or plotting code can reuse it.
                clip_on=False
            )

        # Iterate through every relevant item so none of the morphology/simulation data is skipped.
        for key, d in zip(keys_c, dvals):
            # Store `frac` so later simulation, analysis, or plotting code can reuse it.
            frac = d / dmax
            # Store `x` so later simulation, analysis, or plotting code can reuse it.
            x    = x_left + frac * (x_right - x_left)
            # Store `y` so later simulation, analysis, or plotting code can reuse it.
            y    = att_values.get(key, 0.0)
            # Store `marker` so later simulation, analysis, or plotting code can reuse it.
            marker = 'D' if key in soma_index else 'o'
            # Store `size` so later simulation, analysis, or plotting code can reuse it.
            size   = 60  if key in soma_index else 42
            # Store `txt` so later simulation, analysis, or plotting code can reuse it.
            txt    = f"S{soma_index[key]}" if key in soma_index \
                     else f"{non_soma_index.get(key, '?')}"

            # Draw points marking segment metrics or key anatomical locations.
            ax.scatter(x, y, color=cluster_colors.get(c, 'lightcoral'),
                       # Store `marker` so later simulation, analysis, or plotting code can reuse it.
                       marker=marker, s=MS_DOT_CLUST, edgecolors='white', linewidths=0.8, zorder=3)
            # Apply a plotting/style command to format the current figure.
            ax.annotate(txt, (x, y), fontsize=FONT_TICK - 2, ha='center', va='bottom',
                        # Store `xytext` so later simulation, analysis, or plotting code can reuse it.
                        xytext=(0, 4), textcoords='offset points')

    # Limit the visible axis range to focus on the important part of the data.
    ax.set_xlim(-0.5, k - 0.5)
    # Limit the visible axis range to focus on the important part of the data.
    ax.set_ylim(-0.14, 1.05)
    # Apply a plotting/style command to format the current figure.
    ax.set_xticks(cluster_centers)
    # Apply a plotting/style command to format the current figure.
    ax.set_xticklabels([f"{c}" for c in range(k)], fontsize=FONT_TICK)
    # Label the horizontal axis with the correct variable and units.
    ax.set_xlabel(
    # Store `"Branch order (within each order, horizontal position` so later simulation, analysis, or plotting code can reuse it.
    "Branch order (within each order, horizontal position = distance from stim site [µm])",
    # Store `fontsize` so later simulation, analysis, or plotting code can reuse it.
    fontsize=FONT_LABEL
    )
    # Label the vertical axis with the correct variable and units.
    ax.set_ylabel("Peak ratio vs stimulated segment", fontsize=FONT_LABEL)
    # Set the panel title so the stimulus/metric being plotted is clear.
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold')
    # Apply a plotting/style command to format the current figure.
    ax.tick_params(labelsize=FONT_TICK)
    # Add a light grid to make values easier to read from the plot.
    ax.grid(alpha=0.25, axis='y')

    # Store `cluster_handles` so later simulation, analysis, or plotting code can reuse it.
    cluster_handles = [
        # Apply a plotting/style command to format the current figure.
        plt.Line2D([0], [0], marker='o', color='w',
                   # Store `markerfacecolor` so later simulation, analysis, or plotting code can reuse it.
                   markerfacecolor=cluster_colors.get(c, 'lightcoral'),
                   # Store `markersize` so later simulation, analysis, or plotting code can reuse it.
                   markersize=12, label=f"Cluster {c}")
        # Iterate through every relevant item so none of the morphology/simulation data is skipped.
        for c in range(k)
    ]
    # Store `soma_handle` so later simulation, analysis, or plotting code can reuse it.
    soma_handle = plt.Line2D([0], [0], marker='D', color='w',
                             # Store `markerfacecolor` so later simulation, analysis, or plotting code can reuse it.
                             markerfacecolor='grey', markersize=12, label='Soma segment')
    # Add a legend so the plotted colors/markers can be interpreted.
    ax.legend(handles=cluster_handles + [soma_handle], loc='upper right',
              # Store `fontsize` so later simulation, analysis, or plotting code can reuse it.
              fontsize=FONT_LEGEND, framealpha=0.9)
    # Apply a plotting/style command to format the current figure.
    ax.text(-0.48, axis_y - 0.005, "Distance (µm)",
            # Store `ha` so later simulation, analysis, or plotting code can reuse it.
            ha='left', va='center', fontsize=FONT_LEGEND, fontweight='bold', clip_on=False)


# ---------------------------------------------------------------------------------------------------
# Helper: print cluster table (single canonical definition — FIX #4)
# ---------------------------------------------------------------------------------------------------
# Define `print_cluster_table` to isolate this repeated calculation/plotting step.
def print_cluster_table(att_values, cluster_ids, all_keys,
                        # Execute this statement as part of the current analysis step.
                        soma_index, non_soma_index, segment_labels,
                        # Store `dist_map, title` so later simulation, analysis, or plotting code can reuse it.
                        dist_map, title="Cluster Table"):
    # Print a diagnostic message/value so the run can be checked while executing.
    print("\n" + "=" * 110)
    # Print a diagnostic message/value so the run can be checked while executing.
    print(f"  {title}")
    # Print a diagnostic message/value so the run can be checked while executing.
    print("=" * 110)
    # Print a diagnostic message/value so the run can be checked while executing.
    print(f"  {'Rank':>4}  {'Segment label':<35}  {'Class':<9}  "
          # Execute this statement as part of the current analysis step.
          f"{'Peak ratio':>10}  {'Cluster':>10}  {'Dist (um)':>10}")
    # Print a diagnostic message/value so the run can be checked while executing.
    print("-" * 110)

    # Sort these values so ranking/order-based selection is deterministic.
    sorted_keys = sorted(all_keys, key=lambda k: att_values.get(k, 0.0), reverse=True)

    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for rank, key in enumerate(sorted_keys, 1):
        # Store `att` so later simulation, analysis, or plotting code can reuse it.
        att = att_values.get(key, 0.0)
        # Store `c` so later simulation, analysis, or plotting code can reuse it.
        c   = cluster_ids.get(key, -1)
        # Store `d` so later simulation, analysis, or plotting code can reuse it.
        d   = dist_map.get(key, np.nan)
        # Store `lbl` so later simulation, analysis, or plotting code can reuse it.
        lbl = segment_labels.get(key, key)
        # Store `cls` so later simulation, analysis, or plotting code can reuse it.
        cls = ("SOMA"     if key in soma_index     else
               # Execute this statement as part of the current analysis step.
               "NON-SOMA" if key in non_soma_index else "UNKNOWN")
        # Store `cluster_txt` so later simulation, analysis, or plotting code can reuse it.
        cluster_txt = f"Cluster {c}" if c >= 0 else "Filtered"
        # Print a diagnostic message/value so the run can be checked while executing.
        print(f"  {rank:>4}  {lbl:<35}  {cls:<9}  {att:>10.4f}  {cluster_txt:<10}  {d:>10.2f}")

    # Print a diagnostic message/value so the run can be checked while executing.
    print("=" * 110)


# ---------------------------------------------------------------------------------------------------
# PART B — Run clustering
# NOTE: The TTM/FWHM filter scatter is deferred to the very end of the script (Step 15)
# so it appears AFTER all analysis plots rather than before them.
# FIX #1: clust1/clust2/k1/k2 are now ASSIGNED HERE before any code tries to use them.
# The original code called reorder_clusters_by_similarity_to_stim(clust1, ...) before
# cluster_segments_by_trace_similarity() had ever been invoked, causing a NameError.
# ---------------------------------------------------------------------------------------------------
# Execute this statement as part of the current analysis step.
clust1_raw, k1 = cluster_segments_by_trace_similarity(
    # Store `v1, filtered_keys1, t1, t_start` so later simulation, analysis, or plotting code can reuse it.
    v1, filtered_keys1, t1, t_start=100.0, t_end=250.0, n_clusters=5)
# Execute this statement as part of the current analysis step.
clust2_raw, k2 = cluster_segments_by_trace_similarity(
    # Store `v2, filtered_keys2, t2, t_start` so later simulation, analysis, or plotting code can reuse it.
    v2, filtered_keys2, t2, t_start=100.0, t_end=1100.0, n_clusters=5)

# Store reordered cluster labels for the burst condition.
clust1 = reorder_clusters_by_similarity_to_stim(clust1_raw, corr1, k1)
# Store reordered cluster labels for the sustained condition.
clust2 = reorder_clusters_by_similarity_to_stim(clust2_raw, corr2, k2)

# Assign red-gradient colors to burst clusters.
cluster_colors1 = build_cluster_red_gradient(k1)
# Assign red-gradient colors to sustained clusters.
cluster_colors2 = build_cluster_red_gradient(k2)

# Print a diagnostic message/value so the run can be checked while executing.
print(f"1 ms clustered segments: {len(clust1)}, k1={k1}, colors={len(cluster_colors1)}")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"1 s  clustered segments: {len(clust2)}, k2={k2}, colors={len(cluster_colors2)}")

# ---------------------------------------------------------------------------------------------------
# PART D — print cluster tables
# ---------------------------------------------------------------------------------------------------
# Call this function/method to execute the next operation in the pipeline.
print_cluster_table(corr1, clust1, all_segment_keys, soma_index, non_soma_index,
                    # Execute this statement as part of the current analysis step.
                    segment_labels, segment_distance_from_stim,
                    # Store `title` so later simulation, analysis, or plotting code can reuse it.
                    title="Peak-ratio / cluster table — 1 ms stimulus")
# Call this function/method to execute the next operation in the pipeline.
print_cluster_table(corr2, clust2, all_segment_keys, soma_index, non_soma_index,
                    # Execute this statement as part of the current analysis step.
                    segment_labels, segment_distance_from_stim,
                    # Store `title` so later simulation, analysis, or plotting code can reuse it.
                    title="Peak-ratio / cluster table — 1 s stimulus")

# ---------------------------------------------------------------------------------------------------
# PART E — filtered voltage traces
# ---------------------------------------------------------------------------------------------------
# Create the figure/axes where the next plot will be drawn.
fig_v, axes_v = plt.subplots(2, 1, figsize=FIG_VOLTAGE)

# Call this function/method to execute the next operation in the pipeline.
plot_voltage(axes_v[0], t1, v1, filtered_keys1, clust1, cluster_colors1,
             # Execute this statement as part of the current analysis step.
             stim_key, segment_labels,
             # Store `f"Voltage traces — filtered/clustered — 1 ms stimulus (k` so later simulation, analysis, or plotting code can reuse it.
             f"Voltage traces — filtered/clustered — 1 ms stimulus (k={k1})")
# Call this function/method to execute the next operation in the pipeline.
plot_voltage(axes_v[1], t2, v2, filtered_keys2, clust2, cluster_colors2,
             # Execute this statement as part of the current analysis step.
             stim_key, segment_labels,
             # Store `f"Voltage traces — filtered/clustered — 1 s stimulus (k` so later simulation, analysis, or plotting code can reuse it.
             f"Voltage traces — filtered/clustered — 1 s stimulus (k={k2})")

# Adjust spacing so labels and panels do not overlap.
plt.tight_layout()
# Save the completed figure file for poster/report use.
plt.savefig("neuron_voltage_clustered.png", dpi=150)
# Print a diagnostic message/value so the run can be checked while executing.
print("Saved: neuron_voltage_clustered.png")
# Display the completed figure after saving it.
plt.show()

# ---------------------------------------------------------------------------------------------------
# PART F — cluster similarity / distance panel removed:
#           the branch-order plot (Step 14) is the primary spatial view.
# ---------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------
# build_seg_key_to_branch_level — defined here so it is available for the JSON
# export in Part G below AND for the branch-order plot in Step 14.
# ---------------------------------------------------------------------------------------------------
# Define `build_seg_key_to_branch_level` to isolate this repeated calculation/plotting step.
def build_seg_key_to_branch_level(nodes, node_order, children, node_to_segment):
    """
    # Execute this statement as part of the current analysis step.
    BFS over the SWC node tree.  Assigns an integer branch order to every node:
      # Store `- The root (parent` so later simulation, analysis, or plotting code can reuse it.
      - The root (parent == -1) is order 1.
      # Execute this statement as part of the current analysis step.
      - Traversing a bifurcation node (2+ children) increments the order for all
        # Execute this statement as part of the current analysis step.
        children of that fork.  Non-branching pass-through nodes keep the same order.

    # Execute this statement as part of the current analysis step.
    Then each NEURON segment key is assigned the branch order of the SWC node that
    # Execute this statement as part of the current analysis step.
    node_to_segment maps to it (nearest-neighbour mapping already computed earlier).

    # Execute this statement as part of the current analysis step.
    Returns
    # Execute this statement as part of the current analysis step.
    -------
    # Execute this statement as part of the current analysis step.
    seg_key_to_level : dict  segment_key -> int branch order (1-based)
    # Execute this statement as part of the current analysis step.
    node_level       : dict  node_id     -> int branch order (for diagnostics)
    """
    # Import this package/function because a later step depends on it.
    from collections import deque

    # Store `root` so later simulation, analysis, or plotting code can reuse it.
    root = next(nid for nid in node_order if nodes[nid]['parent_id'] == -1)

    # Store `node_level` so later simulation, analysis, or plotting code can reuse it.
    node_level = {}
    # Store `queue` so later simulation, analysis, or plotting code can reuse it.
    queue      = deque()
    # Append this result to a list so it is included in later processing.
    queue.append((root, 1))

    # Keep traversing/searching until the morphology path reaches its stopping condition.
    while queue:
        # Store `nid, level` so later simulation, analysis, or plotting code can reuse it.
        nid, level = queue.popleft()
        # Store `node_level[nid]` so later simulation, analysis, or plotting code can reuse it.
        node_level[nid] = level

        # Store `kids` so later simulation, analysis, or plotting code can reuse it.
        kids = children[nid]
        # Test this condition so the script handles this case safely/correctly.
        if len(kids) >= 2:
            # Store `next_level` so later simulation, analysis, or plotting code can reuse it.
            next_level = level + 1
        # Use this fallback when the earlier condition is not true.
        else:
            # Store `next_level` so later simulation, analysis, or plotting code can reuse it.
            next_level = level

        # Iterate through every relevant item so none of the morphology/simulation data is skipped.
        for child in kids:
            # Test this condition so the script handles this case safely/correctly.
            if child not in node_level:
                # Append this result to a list so it is included in later processing.
                queue.append((child, next_level))

    # Map each segment key to its branch order for branch-level plots.
    seg_key_to_level = {}
    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for nid, seg_key in node_to_segment.items():
        # Test this condition so the script handles this case safely/correctly.
        if seg_key is None:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue
        # Store `lvl` so later simulation, analysis, or plotting code can reuse it.
        lvl = node_level.get(nid, 1)
        # Test this condition so the script handles this case safely/correctly.
        if seg_key not in seg_key_to_level or lvl < seg_key_to_level[seg_key]:
            # Store `seg_key_to_level[seg_key]` so later simulation, analysis, or plotting code can reuse it.
            seg_key_to_level[seg_key] = lvl

    # Return the final value(s) produced by this helper function.
    return seg_key_to_level, node_level


# ---------------------------------------------------------------------------------------------------
# Pre-compute branch-order map here so it is available for the JSON export below.
# ---------------------------------------------------------------------------------------------------
# Execute this statement as part of the current analysis step.
seg_key_to_branch_level, node_level_map = build_seg_key_to_branch_level(
    # Execute this statement as part of the current analysis step.
    nodes, node_order, children, node_to_segment)

# Store `max_level_found` so later simulation, analysis, or plotting code can reuse it.
max_level_found = max(seg_key_to_branch_level.values()) if seg_key_to_branch_level else 1
# Print a diagnostic message/value so the run can be checked while executing.
print(f"\nBranch order range: 1 – {max_level_found}")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"Segments with branch-order assignment: {len(seg_key_to_branch_level)}")

# ---------------------------------------------------------------------------------------------------
# PART G — JSON export
# FIX #8 / #9: corrected two broken dict literals where a closing brace was missing,
# causing the next key to be parsed as part of the preceding value.
# ---------------------------------------------------------------------------------------------------
# Store `export` so later simulation, analysis, or plotting code can reuse it.
export = {
    # Execute this statement as part of the current analysis step.
    "node_to_segment": {str(k): v for k, v in node_to_segment.items()},
    # Execute this statement as part of the current analysis step.
    "nodes": [
        # Execute this statement as part of the current analysis step.
        {
            # Execute this statement as part of the current analysis step.
            "id":           nid,
            # Execute this statement as part of the current analysis step.
            "x":            nodes[nid]["x"],
            # Execute this statement as part of the current analysis step.
            "y":            nodes[nid]["y"],
            # Execute this statement as part of the current analysis step.
            "z":            nodes[nid]["z"],
            # Execute this statement as part of the current analysis step.
            "type":         nodes[nid]["type"],
            # Execute this statement as part of the current analysis step.
            "parent":       nodes[nid]["parent_id"],
            # Execute this statement as part of the current analysis step.
            "radius":       nodes[nid]["radius"],
            # topology flags — needed by the 3D viewer
            # Execute this statement as part of the current analysis step.
            "is_axon":      (node_to_segment.get(nid) is not None and
                             # Call this function/method to execute the next operation in the pipeline.
                             seg_is_axon.get(node_to_segment.get(nid), False)),
            # Execute this statement as part of the current analysis step.
            "branch_order": node_level_map.get(nid, 1),
        }
        # Iterate through every relevant item so none of the morphology/simulation data is skipped.
        for nid in node_order
    ],
    # Exact SWC node nearest to the stim section's distal 3D tip.
    # Computed here once so the 3D viewer doesn't have to guess.
    # Execute this statement as part of the current analysis step.
    "stim_node_id": int(min(
        # Execute this statement as part of the current analysis step.
        node_order,
        # Execute this statement as part of the current analysis step.
        key=lambda nid: (
            # Execute this statement as part of the current analysis step.
            (nodes[nid]["x"] - float(distal_xyz[0]))**2 +
            # Execute this statement as part of the current analysis step.
            (nodes[nid]["y"] - float(distal_xyz[1]))**2 +
            # Execute this statement as part of the current analysis step.
            (nodes[nid]["z"] - float(distal_xyz[2]))**2
        )
    )),
    # Execute this statement as part of the current analysis step.
    "stim_key": stim_key,
    # Execute this statement as part of the current analysis step.
    "runs": {
        # Execute this statement as part of the current analysis step.
        "1ms": {
            # Execute this statement as part of the current analysis step.
            "corr":          {k: float(corr1[k])           for k in all_segment_keys},
            # Execute this statement as part of the current analysis step.
            "cluster":       {k: int(clust1.get(k, -1))    for k in all_segment_keys},
            # Compute the maximum used for peak detection or normalization.
            "peak":          {k: float(np.max(v1[k]))       for k in v1},   # FIX #8: was missing closing brace
            # Execute this statement as part of the current analysis step.
            "k":             int(k1),
            # Execute this statement as part of the current analysis step.
            "filtered_keys": filtered_keys1,
        },
        # Execute this statement as part of the current analysis step.
        "1s": {
            # Execute this statement as part of the current analysis step.
            "corr":          {k: float(corr2[k])           for k in all_segment_keys},
            # Execute this statement as part of the current analysis step.
            "cluster":       {k: int(clust2.get(k, -1))    for k in all_segment_keys},
            # Compute the maximum used for peak detection or normalization.
            "peak":          {k: float(np.max(v2[k]))       for k in v2},   # FIX #9: was missing closing brace
            # Execute this statement as part of the current analysis step.
            "k":             int(k2),
            # Execute this statement as part of the current analysis step.
            "filtered_keys": filtered_keys2,
        },
    },
    # Execute this statement as part of the current analysis step.
    "segment_labels": segment_labels,
    # Execute this statement as part of the current analysis step.
    "soma_keys":       soma_segment_keys,
    # Execute this statement as part of the current analysis step.
    "non_soma_keys":   non_soma_segment_keys,
    # Execute this statement as part of the current analysis step.
    "soma_index":      soma_index,
    # Execute this statement as part of the current analysis step.
    "non_soma_index":  {k: int(v) for k, v in non_soma_index.items()},
}

# Open the SWC file for reading while automatically closing it afterward.
with open("neuron_morphology_data.json", "w") as f:
    # Write the structured analysis results to a JSON file.
    json.dump(export, f)
# Print a diagnostic message/value so the run can be checked while executing.
print("Exported: neuron_morphology_data.json")


# ===================================================================================================
# STEP 11 — NODE SUMMARY
# FIX #7: The original rebuilt node_to_segment from scratch with an expensive O(N²) loop,
# discarding the correct result already computed by build_node_to_segment_map() earlier.
# We now reuse the existing node_to_segment dict and only add the peak-voltage annotation.
# ===================================================================================================
# Print a diagnostic message/value so the run can be checked while executing.
print("\nNode summary:\n")

# Pre-compute peak voltage for every segment in the sustained run
# Compute the maximum used for peak detection or normalization.
seg_peak = {k: np.max(v2[k]) for k in v2}

# Iterate through every relevant item so none of the morphology/simulation data is skipped.
for nid in node_order:
    # Store `node` so later simulation, analysis, or plotting code can reuse it.
    node     = nodes[nid]
    # Store `best_key` so later simulation, analysis, or plotting code can reuse it.
    best_key = node_to_segment[nid]   # FIX #7: use pre-computed mapping, not a new O(N²) search
    # Store `peak` so later simulation, analysis, or plotting code can reuse it.
    peak     = seg_peak.get(best_key, None)

    # Test this condition so the script handles this case safely/correctly.
    if best_key in soma_index:
        # Store `seg_class` so later simulation, analysis, or plotting code can reuse it.
        seg_class = "SOMA"
        # Store `seg_lbl` so later simulation, analysis, or plotting code can reuse it.
        seg_lbl   = f"Soma segment {soma_index[best_key]}"
    # Check this alternate case only after the previous condition failed.
    elif best_key in non_soma_index:
        # Store `seg_class` so later simulation, analysis, or plotting code can reuse it.
        seg_class = "NON-SOMA"
        # Store `seg_lbl` so later simulation, analysis, or plotting code can reuse it.
        seg_lbl   = f"Non-soma segment {non_soma_index[best_key]}"
    # Use this fallback when the earlier condition is not true.
    else:
        # Store `seg_class` so later simulation, analysis, or plotting code can reuse it.
        seg_class = "UNKNOWN"
        # Store `seg_lbl` so later simulation, analysis, or plotting code can reuse it.
        seg_lbl   = "Unknown segment"

    # Print a diagnostic message/value so the run can be checked while executing.
    print(f"{nid:>4}  "
          # Execute this statement as part of the current analysis step.
          f"{node['x']:>8.1f} {node['y']:>8.1f} {node['z']:>8.1f}  "
          # Execute this statement as part of the current analysis step.
          f"{node['radius']:>6.2f}  "
          # Execute this statement as part of the current analysis step.
          f"{peak:>8.3f}  {best_key:>30s}  {seg_class:>8s}  {seg_lbl}")

# ===================================================================================================
# STEP 14 — BRANCH-LEVEL CORRELATION DOT PLOT
#
# Branch level definition:
#   Level 1 = nodes reachable from the root (soma) WITHOUT crossing any bifurcation point.
#             These are the most proximal (closest) dendritic/axonal segments.
#   Level 2 = nodes reachable only after crossing exactly ONE bifurcation.
#   Level N = nodes reachable after crossing N-1 bifurcations.
#
# Implementation:
#   BFS from the root node over the SWC tree.  Each time we cross a node that has 2+
#   children (a fork / bifurcation), the level counter increments for all descendant paths.
#   Nodes that sit ON a bifurcation belong to the same level as their parent (the fork
#   itself is still in the same "branch", only its children advance the level).
#
# Why this is the right definition:
#   - Level 1 always contains the soma-adjacent trunk (no forks crossed yet).
#   - Every time you "turn a corner" at a bifurcation the level goes up by 1.
#   - Distal tips of long dendrites therefore have the highest level numbers.
# ===================================================================================================

# seg_key_to_branch_level and node_level_map are already built before the JSON
# export (after Part F). build_seg_key_to_branch_level() is defined above Part F.


# Add legend_loc so each stimulus panel can place its legend differently.
# This lets the 1 s plot use the bottom-right legend position.
# Plot peak ratio by branch order.
# X position = branch order column + local offset based on distance from stim site.
# Within each branch-order column, a small horizontal distance ruler is drawn:
# left end  = 0 distance from stim
# right end = maximum distance in that branch-order group
# Each dot is placed along that local ruler according to its relative distance.
# Define `plot_branch_correlation_dotplot` to isolate this repeated calculation/plotting step.
def plot_branch_correlation_dotplot(corr_values, cluster_ids, cluster_colors,
                                     # Execute this statement as part of the current analysis step.
                                     seg_key_to_level, all_keys, k,
                                     # Execute this statement as part of the current analysis step.
                                     stim_key, segment_labels, ax, title, v_dict,
                                     # Store `legend_loc` so later simulation, analysis, or plotting code can reuse it.
                                     legend_loc='upper left'):
    """
    # Execute this statement as part of the current analysis step.
    Dot plot:
      # Store `y` so later simulation, analysis, or plotting code can reuse it.
      y = peak ratio vs stimulated segment
      # Store `major x grouping` so later simulation, analysis, or plotting code can reuse it.
      major x grouping = branch order
      # Store `local x offset inside each branch-order group` so later simulation, analysis, or plotting code can reuse it.
      local x offset inside each branch-order group = distance from stim site

    # Execute this statement as part of the current analysis step.
    For each branch order:
      # Execute this statement as part of the current analysis step.
      - a short horizontal scale line is drawn
      # Execute this statement as part of the current analysis step.
      - left side of that line represents 0 um from stim
      # Execute this statement as part of the current analysis step.
      - right side represents the maximum distance observed in that branch order
      # Execute this statement as part of the current analysis step.
      - each dot is placed along that local scale
    """

    # Keep only keys that actually have voltage data
    # Store `keys_present` so later simulation, analysis, or plotting code can reuse it.
    keys_present = [k for k in all_keys if k in v_dict]
    # Test this condition so the script handles this case safely/correctly.
    if not keys_present:
        # Apply a plotting/style command to format the current figure.
        ax.text(0.5, 0.5, "No branch-level data", ha='center', va='center',
                # Store `transform` so later simulation, analysis, or plotting code can reuse it.
                transform=ax.transAxes, fontsize=FONT_TICK)
        # Set the panel title so the stimulus/metric being plotted is clear.
        ax.set_title(title)
        # Execute this statement as part of the current analysis step.
        return

    # Determine all branch orders present
    # Sort these values so ranking/order-based selection is deterministic.
    levels_present = sorted({seg_key_to_level.get(k, 1) for k in keys_present})
    # Store `max_lvl` so later simulation, analysis, or plotting code can reuse it.
    max_lvl = max(levels_present)

    # Width of the mini distance ruler inside each branch-order column
    # Store `local_halfwidth` so later simulation, analysis, or plotting code can reuse it.
    local_halfwidth = 0.28

    # Vertical position for the local distance rulers and labels
    # Store `axis_y` so later simulation, analysis, or plotting code can reuse it.
    axis_y  = -0.06
    # Store `tick_y0` so later simulation, analysis, or plotting code can reuse it.
    tick_y0 = axis_y - 0.012
    # Store `tick_y1` so later simulation, analysis, or plotting code can reuse it.
    tick_y1 = axis_y + 0.012

    # Draw one local distance ruler per branch-order group
    # Iterate through every relevant item so none of the morphology/simulation data is skipped.
    for lvl in levels_present:
        # Store `keys_lvl` so later simulation, analysis, or plotting code can reuse it.
        keys_lvl = [k for k in keys_present if seg_key_to_level.get(k, 1) == lvl]

        # Collect valid distances for this branch order
        # Convert data into a NumPy array for vectorized numerical operations.
        dvals = np.array([segment_distance_from_stim.get(k, np.nan) for k in keys_lvl], dtype=float)
        # Store `valid` so later simulation, analysis, or plotting code can reuse it.
        valid = np.isfinite(dvals)
        # Store `keys_lvl` so later simulation, analysis, or plotting code can reuse it.
        keys_lvl = [k for k, ok in zip(keys_lvl, valid) if ok]
        # Store `dvals` so later simulation, analysis, or plotting code can reuse it.
        dvals = dvals[valid]

        # Test this condition so the script handles this case safely/correctly.
        if len(keys_lvl) == 0:
            # Skip this item because it is invalid or not needed for the current calculation.
            continue

        # Maximum distance within this branch-order group
        # Compute the maximum used for peak detection or normalization.
        dmax = float(np.max(dvals))
        # Test this condition so the script handles this case safely/correctly.
        if dmax <= 0:
            # Store `dmax` so later simulation, analysis, or plotting code can reuse it.
            dmax = 1.0

        # Left/right ends of the local ruler for this branch-order column
        # Store `x_center` so later simulation, analysis, or plotting code can reuse it.
        x_center = lvl
        # Store `x_left` so later simulation, analysis, or plotting code can reuse it.
        x_left   = x_center - local_halfwidth
        # Store `x_right` so later simulation, analysis, or plotting code can reuse it.
        x_right  = x_center + local_halfwidth

        # Draw the horizontal ruler line
        # Draw a line showing morphology edges, current waveform, or voltage trace data.
        ax.plot([x_left, x_right], [axis_y, axis_y],
                # Store `color` so later simulation, analysis, or plotting code can reuse it.
                color='black', lw=0.9, zorder=1, clip_on=False)

        # Draw only 3 labels per local ruler so they do not overlap:
        # left = 0, middle = half-max, right = max.
        # Show the unit only once, on the rightmost label.
        # Iterate through every relevant item so none of the morphology/simulation data is skipped.
        for frac, d in zip([0.0, 0.5, 1.0], [0.0, 0.5 * dmax, dmax]):
            # Store `xt` so later simulation, analysis, or plotting code can reuse it.
            xt = x_left + frac * (x_right - x_left)

            # Short ruler tick
            # Draw a line showing morphology edges, current waveform, or voltage trace data.
            ax.plot([xt, xt], [tick_y0, tick_y1],
                    # Store `color` so later simulation, analysis, or plotting code can reuse it.
                    color='black', lw=0.9, zorder=1, clip_on=False)

            # Compact integer labels; only the final label shows the unit.
            # Store `label_txt` so later simulation, analysis, or plotting code can reuse it.
            label_txt = f"{d:.0f}" if frac < 1.0 else f"{d:.0f} µm"
            # Apply a plotting/style command to format the current figure.
            ax.text(
                # Execute this statement as part of the current analysis step.
                xt, axis_y - 0.032, label_txt,
                # Store `ha` so later simulation, analysis, or plotting code can reuse it.
                ha='center', va='top',
                # Store `fontsize` so later simulation, analysis, or plotting code can reuse it.
                fontsize=FONT_TICK - 2,
                # Store `clip_on` so later simulation, analysis, or plotting code can reuse it.
                clip_on=False
            )

        # Plot each segment in this branch-order group using its distance-based local x-position
        # Iterate through every relevant item so none of the morphology/simulation data is skipped.
        for key, d in zip(keys_lvl, dvals):
            # Store `r` so later simulation, analysis, or plotting code can reuse it.
            r       = corr_values.get(key, 0.0)
            # Store `c` so later simulation, analysis, or plotting code can reuse it.
            c       = cluster_ids.get(key, None)
            # Store `is_axon` so later simulation, analysis, or plotting code can reuse it.
            is_axon = seg_is_axon.get(key, False)

            # Map distance to a local position on the ruler
            # Store `frac` so later simulation, analysis, or plotting code can reuse it.
            frac = d / dmax
            # Store `x` so later simulation, analysis, or plotting code can reuse it.
            x    = x_left + frac * (x_right - x_left)

            # Choose marker style by segment class
            # Test this condition so the script handles this case safely/correctly.
            if is_axon:
                # Store `marker` so later simulation, analysis, or plotting code can reuse it.
                marker = 's'
                # Store `color` so later simulation, analysis, or plotting code can reuse it.
                color  = 'cyan'
                # Store `size` so later simulation, analysis, or plotting code can reuse it.
                size   = MS_DOT_BRANCH
            # Check this alternate case only after the previous condition failed.
            elif key in soma_index:
                # Store `marker` so later simulation, analysis, or plotting code can reuse it.
                marker = 'D'
                # Store `color` so later simulation, analysis, or plotting code can reuse it.
                color  = cluster_colors.get(c, '#888888') if c is not None else '#888888'
                # Store `size` so later simulation, analysis, or plotting code can reuse it.
                size   = MS_DOT_BRANCH
            # Use this fallback when the earlier condition is not true.
            else:
                # Store `marker` so later simulation, analysis, or plotting code can reuse it.
                marker = 'o'
                # Store `color` so later simulation, analysis, or plotting code can reuse it.
                color  = cluster_colors.get(c, '#cccccc') if c is not None else '#cccccc'
                # Store `size` so later simulation, analysis, or plotting code can reuse it.
                size   = MS_DOT_BRANCH

            # Draw points marking segment metrics or key anatomical locations.
            ax.scatter(x, r, color=color, marker=marker, s=size,
                       # Store `alpha` so later simulation, analysis, or plotting code can reuse it.
                       alpha=0.80, edgecolors='white', linewidths=0.5, zorder=3)

    # Draw the stimulated segment as a black star
    # Test this condition so the script handles this case safely/correctly.
    if stim_key in v_dict:
        # Store `r_stim` so later simulation, analysis, or plotting code can reuse it.
        r_stim = corr_values.get(stim_key, 1.0)
        # Store `lvl_s` so later simulation, analysis, or plotting code can reuse it.
        lvl_s  = seg_key_to_branch_level.get(stim_key, 1)

        # Place stim star using the same local distance ruler logic
        # Store `keys_lvl` so later simulation, analysis, or plotting code can reuse it.
        keys_lvl = [k for k in keys_present if seg_key_to_branch_level.get(k, 1) == lvl_s]
        # Convert data into a NumPy array for vectorized numerical operations.
        dvals = np.array([segment_distance_from_stim.get(k, np.nan) for k in keys_lvl], dtype=float)
        # Store `dvals` so later simulation, analysis, or plotting code can reuse it.
        dvals = dvals[np.isfinite(dvals)]
        # Compute the maximum used for peak detection or normalization.
        dmax = float(np.max(dvals)) if len(dvals) else 1.0
        # Test this condition so the script handles this case safely/correctly.
        if dmax <= 0:
            # Store `dmax` so later simulation, analysis, or plotting code can reuse it.
            dmax = 1.0

        # Store `x_left` so later simulation, analysis, or plotting code can reuse it.
        x_left  = lvl_s - local_halfwidth
        # Store `x_right` so later simulation, analysis, or plotting code can reuse it.
        x_right = lvl_s + local_halfwidth
        # Store `d_stim` so later simulation, analysis, or plotting code can reuse it.
        d_stim  = segment_distance_from_stim.get(stim_key, 0.0)
        # Store `frac_s` so later simulation, analysis, or plotting code can reuse it.
        frac_s  = d_stim / dmax
        # Store `x_stim` so later simulation, analysis, or plotting code can reuse it.
        x_stim  = x_left + frac_s * (x_right - x_left)

        # Draw points marking segment metrics or key anatomical locations.
        ax.scatter(x_stim, r_stim, color='black', marker='*', s=MS_STIM,
                   # Store `zorder` so later simulation, analysis, or plotting code can reuse it.
                   zorder=6, label='Stimulated segment',
                   # Store `edgecolors` so later simulation, analysis, or plotting code can reuse it.
                   edgecolors='white', linewidths=1.0)

    # Optional axon annotation
    # Store `axon_orders` so later simulation, analysis, or plotting code can reuse it.
    axon_orders = [seg_key_to_level.get(key, 1)
                   # Iterate through every relevant item so none of the morphology/simulation data is skipped.
                   for key in keys_present if seg_is_axon.get(key, False)]
    # Test this condition so the script handles this case safely/correctly.
    if axon_orders:
        # Store `axon_ratios` so later simulation, analysis, or plotting code can reuse it.
        axon_ratios = [corr_values.get(key, 0.0)
                       # Iterate through every relevant item so none of the morphology/simulation data is skipped.
                       for key in keys_present if seg_is_axon.get(key, False)]
        # Store `ann_x` so later simulation, analysis, or plotting code can reuse it.
        ann_x = float(np.median(axon_orders))
        # Store `ann_y` so later simulation, analysis, or plotting code can reuse it.
        ann_y = float(np.median(axon_ratios))

        # Apply a plotting/style command to format the current figure.
        ax.annotate(
            # Execute this statement as part of the current analysis step.
            "Axon",
            # Store `xy` so later simulation, analysis, or plotting code can reuse it.
            xy=(ann_x, ann_y),
            # Store `xytext` so later simulation, analysis, or plotting code can reuse it.
            xytext=(ann_x + 0.6, ann_y + 0.12),
            # Store `ha` so later simulation, analysis, or plotting code can reuse it.
            ha='left', va='bottom',
            # Store `fontsize` so later simulation, analysis, or plotting code can reuse it.
            fontsize=FONT_ANNOT, color='darkcyan', fontweight='bold',
            # Store `arrowprops` so later simulation, analysis, or plotting code can reuse it.
            arrowprops=dict(arrowstyle='->', color='darkcyan', lw=1.8),
            # Store `bbox` so later simulation, analysis, or plotting code can reuse it.
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='darkcyan', alpha=0.9)
        )

    # Baseline
    # Draw a horizontal reference line for baseline or filter thresholds.
    ax.axhline(0, color='k', lw=1.0, linestyle='--', alpha=0.4)

    # Main axes formatting
    # Apply a plotting/style command to format the current figure.
    ax.set_xticks(range(1, max_lvl + 1))
    # Apply a plotting/style command to format the current figure.
    ax.set_xticklabels([f"{i}" for i in range(1, max_lvl + 1)], fontsize=FONT_TICK + 2)
    # Horizontal position inside each branch-order group represents local distance from
    # the stimulated site; the local ruler labels carry the unit in µm.
    # Label the horizontal axis with the correct variable and units.
    ax.set_xlabel(
        # Execute this statement as part of the current analysis step.
        "Branch order (within each order, horizontal position shows distance from stim site)",
        # Store `fontsize` so later simulation, analysis, or plotting code can reuse it.
        fontsize=FONT_LABEL + 2
    )
    # Label the vertical axis with the correct variable and units.
    ax.set_ylabel("Peak ratio vs stimulated segment", fontsize=FONT_LABEL + 2)
    # Limit the visible axis range to focus on the important part of the data.
    ax.set_xlim(0.3, max_lvl + 0.7)
    # Limit the visible axis range to focus on the important part of the data.
    ax.set_ylim(-0.14, 1.05)
    # Set the panel title so the stimulus/metric being plotted is clear.
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold')
    # Apply a plotting/style command to format the current figure.
    ax.tick_params(labelsize=FONT_TICK)
    # Add a light grid to make values easier to read from the plot.
    ax.grid(alpha=0.25, axis='y')

    # Legend
    # Store `cluster_handles` so later simulation, analysis, or plotting code can reuse it.
    cluster_handles = [
        # Apply a plotting/style command to format the current figure.
        plt.Line2D([0], [0], marker='o', color='w',
                   # Store `markerfacecolor` so later simulation, analysis, or plotting code can reuse it.
                   markerfacecolor=cluster_colors.get(c, 'lightcoral'),
                   # Store `markersize` so later simulation, analysis, or plotting code can reuse it.
                   markersize=13, label=f"Cluster {c}")
        # Iterate through every relevant item so none of the morphology/simulation data is skipped.
        for c in range(k)
    ]
    # Store `stim_handle` so later simulation, analysis, or plotting code can reuse it.
    stim_handle = plt.Line2D([0], [0], marker='*', color='w',
                             # Store `markerfacecolor` so later simulation, analysis, or plotting code can reuse it.
                             markerfacecolor='black', markersize=18, label='Stimulated')
    # Store `soma_handle` so later simulation, analysis, or plotting code can reuse it.
    soma_handle = plt.Line2D([0], [0], marker='D', color='w',
                             # Store `markerfacecolor` so later simulation, analysis, or plotting code can reuse it.
                             markerfacecolor='grey', markersize=13, label='Soma segment')
    # Store `axon_handle` so later simulation, analysis, or plotting code can reuse it.
    axon_handle = plt.Line2D([0], [0], marker='s', color='w',
                             # Store `markerfacecolor` so later simulation, analysis, or plotting code can reuse it.
                             markerfacecolor='cyan', markersize=13, label='Axon (detected)')

   # Legend entry explaining the local ruler beneath each branch-order group.
    # Execute this statement as part of the current analysis step.
    distance_handle = plt.Line2D(
        # Execute this statement as part of the current analysis step.
        [0, 1], [0, 0],
        # Store `color` so later simulation, analysis, or plotting code can reuse it.
        color='black', lw=1.2,
        # Store `label` so later simulation, analysis, or plotting code can reuse it.
        label='Local ruler: 0 → max distance [µm]'
    )
    # Add a legend so the plotted colors/markers can be interpreted.
    ax.legend(
        # Store `handles` so later simulation, analysis, or plotting code can reuse it.
        handles=cluster_handles + [stim_handle, soma_handle, axon_handle, distance_handle],
        # Store `loc` so later simulation, analysis, or plotting code can reuse it.
        loc=legend_loc,
        # Store `fontsize` so later simulation, analysis, or plotting code can reuse it.
        fontsize=FONT_LEGEND,
        # Store `framealpha` so later simulation, analysis, or plotting code can reuse it.
        framealpha=0.55,
        # Store `facecolor` so later simulation, analysis, or plotting code can reuse it.
        facecolor='white',
        # Store `edgecolor` so later simulation, analysis, or plotting code can reuse it.
        edgecolor='black',
        # Store `ncol` so later simulation, analysis, or plotting code can reuse it.
        ncol=2
    )


# Create the figure/axes where the next plot will be drawn.
fig_b, axes_b = plt.subplots(2, 1, figsize=FIG_BRANCH)

# =====================================================================
# Compute space constant (λ) BEFORE plotting branch-order figures
# =====================================================================

# Execute this statement as part of the current analysis step.
lambda_1ms = compute_space_constant(
    # Execute this statement as part of the current analysis step.
    corr1,
    # Execute this statement as part of the current analysis step.
    segment_distance_from_stim,
    # Execute this statement as part of the current analysis step.
    all_segment_keys
)

# Execute this statement as part of the current analysis step.
lambda_1s = compute_space_constant(
    # Execute this statement as part of the current analysis step.
    corr2,
    # Execute this statement as part of the current analysis step.
    segment_distance_from_stim,
    # Execute this statement as part of the current analysis step.
    all_segment_keys
)

# Print a diagnostic message/value so the run can be checked while executing.
print("\n================ SPACE CONSTANT =================")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"λ (1 ms stimulus) = {lambda_1ms:.2f} µm")
# Print a diagnostic message/value so the run can be checked while executing.
print(f"λ (1 s stimulus)  = {lambda_1s:.2f} µm")
# Print a diagnostic message/value so the run can be checked while executing.
print("================================================\n")


# Plot the 1 ms branch-order panel.
# Title no longer shows k because k refers to KMeans clusters, not branch order.
# Call this function/method to execute the next operation in the pipeline.
plot_branch_correlation_dotplot(
    # Execute this statement as part of the current analysis step.
    corr1, clust1, cluster_colors1,
    # Execute this statement as part of the current analysis step.
    seg_key_to_branch_level, all_segment_keys, k1,
    # Execute this statement as part of the current analysis step.
    stim_key, segment_labels, axes_b[0],
    # Execute this statement as part of the current analysis step.
    "Peak ratio by Branch Order — 1 ms stimulus",
    # Store `v_dict` so later simulation, analysis, or plotting code can reuse it.
    v_dict=v1,
    # Store `legend_loc` so later simulation, analysis, or plotting code can reuse it.
    legend_loc='upper left'
)

# Plot the 1 s branch-order panel.
# Keep the legend in the bottom-right as requested.
# Call this function/method to execute the next operation in the pipeline.
plot_branch_correlation_dotplot(
    # Execute this statement as part of the current analysis step.
    corr2, clust2, cluster_colors2,
    # Execute this statement as part of the current analysis step.
    seg_key_to_branch_level, all_segment_keys, k2,
    # Execute this statement as part of the current analysis step.
    stim_key, segment_labels, axes_b[1],
    # Execute this statement as part of the current analysis step.
    "Peak ratio by Branch Order — 1 s stimulus",
    # Store `v_dict` so later simulation, analysis, or plotting code can reuse it.
    v_dict=v2,
    # Store `legend_loc` so later simulation, analysis, or plotting code can reuse it.
    legend_loc='center right'
)

# Adjust spacing so labels and panels do not overlap.
plt.tight_layout()
# Save the completed figure file for poster/report use.
plt.savefig("neuron_branch_order.png", dpi=150)
# Print a diagnostic message/value so the run can be checked while executing.
print("Saved: neuron_branch_order.png")
# Display the completed figure after saving it.
plt.show()

# ===================================================================================================
# STEP 15 — TTM AND FWHM vs DISTANCE FROM STIM + FILTER DIAGNOSTIC
#
# Each figure has three panels:
#   Row 0 : Time-to-max (ms) vs distance from stim site (µm), coloured by branch order
#   Row 1 : FWHM (ms)        vs distance from stim site (µm), coloured by branch order
#   Row 2 : TTM vs FWHM — filter diagnostic (passed = coloured, failed = grey)
# ===================================================================================================
# Create the figure/axes where the next plot will be drawn.
fig_feat1, axes_feat1 = plt.subplots(3, 1, figsize=FIG_DISTANCE)
# Call this function/method to execute the next operation in the pipeline.
plot_ttm_fwhm_vs_distance(
    # Execute this statement as part of the current analysis step.
    axes_feat1, ttm1, fwhm1,
    # Execute this statement as part of the current analysis step.
    segment_distance_from_stim, seg_key_to_branch_level,
    # Store `all_segment_keys, run_label` so later simulation, analysis, or plotting code can reuse it.
    all_segment_keys, run_label="1 ms stimulus",
    # Store `filtered_keys` so later simulation, analysis, or plotting code can reuse it.
    filtered_keys=filtered_keys1,
    # Store `ttm_max` so later simulation, analysis, or plotting code can reuse it.
    ttm_max=15.0, fwhm_max=25.0
)
# Adjust spacing so labels and panels do not overlap.
plt.tight_layout()
# Save the completed figure file for poster/report use.
plt.savefig("neuron_ttm_fwhm_vs_distance_1ms.png", dpi=150, bbox_inches='tight')
# Print a diagnostic message/value so the run can be checked while executing.
print("Saved: neuron_ttm_fwhm_vs_distance_1ms.png")
# Display the completed figure after saving it.
plt.show()

# Create the figure/axes where the next plot will be drawn.
fig_feat2, axes_feat2 = plt.subplots(3, 1, figsize=FIG_DISTANCE)
# Call this function/method to execute the next operation in the pipeline.
plot_ttm_fwhm_vs_distance(
    # Execute this statement as part of the current analysis step.
    axes_feat2, ttm2, fwhm2,
    # Execute this statement as part of the current analysis step.
    segment_distance_from_stim, seg_key_to_branch_level,
    # Store `all_segment_keys, run_label` so later simulation, analysis, or plotting code can reuse it.
    all_segment_keys, run_label="1 s stimulus",
    # Store `filtered_keys` so later simulation, analysis, or plotting code can reuse it.
    filtered_keys=filtered_keys2,
    # Store `ttm_max` so later simulation, analysis, or plotting code can reuse it.
    ttm_max=1000.0, fwhm_min=5.0
)
# Adjust spacing so labels and panels do not overlap.
plt.tight_layout()
# Save the completed figure file for poster/report use.
plt.savefig("neuron_ttm_fwhm_vs_distance_1s.png", dpi=150, bbox_inches='tight')
# Print a diagnostic message/value so the run can be checked while executing.
print("Saved: neuron_ttm_fwhm_vs_distance_1s.png")
# Display the completed figure after saving it.
plt.show()

# input("Press Enter to exit...")
