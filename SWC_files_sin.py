# ===================================================================================================
# NEURON + SWC PIPELINE (SEGMENT-LEVEL ANALYSIS)
# ---------------------------------------------------------------------------------------------------
# Key ideas:
# - SWC nodes define morphology
# - NEURON builds sections from that morphology
# - Sections are subdivided into segments (simulation compartments)
# - We classify EACH SEGMENT (not section) as soma/non-soma based on proximity
# ===================================================================================================

# =========================
# Imports
# =========================
import numpy as np
import matplotlib.pyplot as plt
import vtk
import os
import sys

from neuron import h, gui
from neuron.units import ms, mV, um

print(f'VTK version: {vtk.VTK_VERSION}')

# =========================
# Load NEURON tools
# =========================
h.load_file("stdrun.hoc")     # standard simulation tools
h.load_file("import3d.hoc")   # SWC import tools

# =========================
# Load SWC file
# =========================
swc_path = '/home/aksay_lab/NeuronProject/CloudvolNeuron_Test/SWC_files/76182_reRoot_reSample_5000.swc'

print("File exists?", os.path.exists(swc_path))
print("Loading SWC file:", swc_path)

reader = h.Import3d_SWC_read()   # parses SWC file
reader.input(swc_path)

importer = h.Import3d_GUI(reader, 1)  # GUI suppressed

# =========================
# Cell container
# =========================
class Cell:
    def __init__(self):
        self.sl = h.SectionList()  # will hold NEURON sections

cell = Cell()
importer.instantiate(cell)  # build NEURON sections


# =========================
# 3D visualization
# =========================
shape = h.Shape(cell.sl)
shape.exec_menu('3D Rotate')
shape.show(0)
shape.flush()
h.doNotify()

# ===================================================================================================
# STEP 1 — Parse SWC nodes manually
# ===================================================================================================
nodes = {}
node_order = []

# =====================================================================
# FIX: Match NEURON segments to SWC edges per section
# =====================================================================
total_fixed_segments = 0

for sec in h.allsec():

    n3d = sec.n3d()

    if n3d > 1:
        sec.nseg = n3d - 1   # one segment per edge
    else:
        sec.nseg = 1         # fallback

    total_fixed_segments += int(sec.nseg)

print("\nAfter fixing nseg:")
print("Total NEURON segments:", total_fixed_segments)
print("Expected SWC edges:", len(nodes) - 1)
#------------------------------------------------------------------------

with open(swc_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('#') or line == '':
            continue

        parts = line.split()

        node_id   = int(parts[0])
        node_type = int(parts[1])
        x, y, z   = float(parts[2]), float(parts[3]), float(parts[4])
        radius    = float(parts[5])
        parent_id = int(parts[6])

        nodes[node_id] = {
            'node_id': node_id,
            'type': node_type,
            'x': x, 'y': y, 'z': z,
            'radius': radius,
            'parent_id': parent_id
        }

        node_order.append(node_id)

# =====================================================================
# BUILD TRUE SECTIONS FROM SWC GRAPH
# =====================================================================

from collections import defaultdict

# Build children map
children = defaultdict(list)
for nid, d in nodes.items():
    parent = d['parent_id']
    if parent != -1:
        children[parent].append(nid)

# Identify branch points
def is_branch(nid):
    return len(children[nid]) > 1

def is_leaf(nid):
    return len(children[nid]) == 0

def is_root(nid):
    return nodes[nid]['parent_id'] == -1

# =====================================================================
# Extract sections (chains)
# =====================================================================
sections = []
visited_edges = set()

for nid in nodes:

    parent = nodes[nid]['parent_id']

    # Start a new section if:
    # - root
    # - branch point
    # - OR parent is a branch point
    if is_root(nid) or is_branch(nid) or (parent != -1 and is_branch(parent)):

        for child in children[nid]:

            path = [nid, child]
            visited_edges.add((nid, child))

            current = child

            # walk forward until branch or leaf
            while not is_branch(current) and not is_leaf(current):
                next_node = children[current][0]

                if (current, next_node) in visited_edges:
                    break

                path.append(next_node)
                visited_edges.add((current, next_node))
                current = next_node

            sections.append(path)

# =====================================================================
# PRINT RESULTS
# =====================================================================
print("\n" + "="*80)
print("TRUE SWC SECTIONS")
print("="*80)

for i, sec in enumerate(sections):

    print("\n" + "-"*60)
    print(f"Section {i}")
    print(f"Number of nodes: {len(sec)}")
    print(f"Nodes: {sec}")

    for j in range(len(sec) - 1):

        n0 = nodes[sec[j]]
        n1 = nodes[sec[j+1]]

        print(f"\n  Segment {j}: Node {sec[j]} → Node {sec[j+1]}")
        print(f"    Start: ({n0['x']:.2f}, {n0['y']:.2f}, {n0['z']:.2f})")
        print(f"    End  : ({n1['x']:.2f}, {n1['y']:.2f}, {n1['z']:.2f})")
        print(f"    Radius: {n0['radius']:.3f} → {n1['radius']:.3f}")

print("\n" + "="*80)
print(f"TOTAL TRUE SECTIONS: {len(sections)}")
print("="*80)

# ===================================================================================================
# STEP 2 — Identify soma nodes
# ===================================================================================================
# Soma = node 1 + nodes directly connected to node 1
soma_node_ids = {
    nid for nid, d in nodes.items()
    if nid == 1 or d['parent_id'] == 1
}

non_soma_node_ids = set(nodes.keys()) - soma_node_ids

# Convert to arrays for distance calculations
soma_coords = np.array([[nodes[n]['x'], nodes[n]['y'], nodes[n]['z']] for n in soma_node_ids])

print(f"Soma nodes: {len(soma_node_ids)}")
print(f"Non-soma nodes: {len(non_soma_node_ids)}")

# ===================================================================================================
# STEP 3 — Classify SEGMENTS (NOT sections)
# ===================================================================================================
SOMA_THRESHOLD = 10.0  # µm

soma_segments = []
non_soma_segments = []
segment_labels = {}

soma_count = 0
non_soma_count = 0

for sec in h.allsec():

    n3d_pts = sec.n3d()

    for seg in sec:

        # Default
        is_soma = False

        if n3d_pts > 0:
            total_arc = sec.arc3d(n3d_pts - 1)

            if total_arc > 0:

                # Normalize arc positions
                arc_fracs = np.array([
                    sec.arc3d(i) / total_arc for i in range(n3d_pts)
                ])

                # Find closest 3D point to segment location
                idx = int(np.argmin(np.abs(arc_fracs - seg.x)))

                seg_xyz = np.array([
                    sec.x3d(idx),
                    sec.y3d(idx),
                    sec.z3d(idx)
                ])

                # Check distance to soma
                for coord in soma_coords:
                    if np.linalg.norm(seg_xyz - coord) < SOMA_THRESHOLD:
                        is_soma = True
                        break

        key = f"{sec.name()}({seg.x:.3f})"

        if is_soma:
            soma_segments.append((sec, seg))
            segment_labels[key] = f"Soma_{soma_count}"
            soma_count += 1
        else:
            non_soma_segments.append((sec, seg))
            segment_labels[key] = f"NonSoma_{non_soma_count}"
            non_soma_count += 1

print(f"\nTotal segments: {soma_count + non_soma_count}")
print(f"Soma segments: {soma_count}")
print(f"Non-soma segments: {non_soma_count}")

# ===================================================================================================
# STEP 4 — Apply passive properties
# ===================================================================================================
for sec in h.allsec():
    sec.Ra = 150.0
    sec.cm = 1.0
    sec.insert('pas')

    for seg in sec:
        seg.pas.g = 0.00005
        seg.pas.e = -65.0

# ===================================================================================================
# STEP 5 — Find stimulation site (non-soma leaf)
# ===================================================================================================
# Helper: get soma sections (those whose *segments* are labeled soma)
soma_secs = {sec for sec, _ in soma_segments}

# Fallback: if soma_segments is empty for some reason, use the first section as soma
if not soma_secs:
    soma_secs = {list(h.allsec())[0]}

# Build a SectionRef for each section so we can walk up parents
sec_ref_map = {sec: h.SectionRef(sec=sec) for sec in h.allsec()}

def path_distance_from_soma(sec, soma_secs, dx=0.5):
    """
    Compute approximate path distance (in µm) from the soma to the
    middle of `sec` by walking up through parent sections and summing
    path lengths along the way.
    """
    # If this section itself is in soma, distance is 0
    if sec in soma_secs:
        return 0.0

    dist = 0.0

    # Add this section's own length up to its midpoint
    dist += sec.L * dx

    # Walk up to soma along parents
    sr = sec_ref_map[sec]
    while sr.has_parent():
        parent = sr.parent().sec

        # If parent is soma, just add distance from attachment point to soma center
        if parent in soma_secs:
            # assume midpoint of parent is somatic center for path measure
            dist += parent.L * 0.5
            break
        else:
            # add full length of the parent and continue up
            dist += parent.L
            sr = sec_ref_map[parent]

    return dist

def get_children(sec):
    return list(sec.children())

all_sections = list(h.allsec())

# Leaf sections are those with no children
leaf_sections = [sec for sec in all_sections if len(get_children(sec)) == 0]

non_soma_sections = {sec for sec, _ in non_soma_segments}

# Restrict to leaf, non-soma sections
candidate_leaves = [sec for sec in leaf_sections if sec in non_soma_sections]

# If for some reason no leaf is non-soma, fall back to all leaves
if not candidate_leaves:
    candidate_leaves = leaf_sections

# Compute path distance from soma for each candidate leaf
leaf_distances = []
for sec in candidate_leaves:
    d = path_distance_from_soma(sec, soma_secs)
    leaf_distances.append((d, sec))

leaf_distances.sort(key=lambda x: x[0])
max_dist, stim_section = leaf_distances[-1]

# --- NEW: choose the *segment* in this section whose center is closest to its distal end ---
# Distal 3D point index and its coordinates
n3d = stim_section.n3d()
distal_idx = n3d - 1
distal_xyz = np.array([
    stim_section.x3d(distal_idx),
    stim_section.y3d(distal_idx),
    stim_section.z3d(distal_idx),
])

# Find the segment whose 3D location (via arc length) is closest to the distal point
best_seg = None
best_dist_seg = float('inf')

total_arc = stim_section.arc3d(n3d - 1)
arc_fracs = np.array([stim_section.arc3d(i) / total_arc for i in range(n3d)])

for seg in stim_section:
    # seg.x is the normalized position 0..1; map that to the nearest 3D point
    idx = int(np.argmin(np.abs(arc_fracs - seg.x)))
    seg_xyz = np.array([
        stim_section.x3d(idx),
        stim_section.y3d(idx),
        stim_section.z3d(idx),
    ])
    d = np.linalg.norm(seg_xyz - distal_xyz)
    if d < best_dist_seg:
        best_dist_seg = d
        best_seg = seg

stim_seg = best_seg          # this is the *segment* we will stimulate
stim_x   = float(stim_seg.x) # its x-position within stim_section

print(f"\nStim site (most distal non-soma leaf): {stim_section.name()}({stim_x:.3f})")
print(f"Path distance from soma ≈ {max_dist:.2f} µm")
print(f"Chosen segment x = {stim_x:.3f}, distance to distal tip ≈ {best_dist_seg:.3f} µm")

# ===================================================================================================
# STEP 6 — Simulation function
# ===================================================================================================
T_STOP = 5000.0
DT = 0.025
I_AMP = 75 #before: 0.0075

def run_sim(stim_dur):

    h.dt = DT
    h.tstop = T_STOP
    h.v_init = -65

    stim = h.IClamp(stim_section(stim_x))
    stim.delay = 100
    stim.dur = stim_dur
    stim.amp = I_AMP

    t_vec = h.Vector().record(h._ref_t)

    v_recs = {}
    for sec in h.allsec():
        for seg in sec:
            key = f"{sec.name()}({seg.x:.3f})"
            v_recs[key] = h.Vector().record(seg._ref_v)

    h.finitialize(-65)
    h.run()

    t = np.array(t_vec)
    v = {k: np.array(v_recs[k]) for k in v_recs}

    return t, v

# ===================================================================================================
# STEP 7 — Run simulations
# ===================================================================================================
print("\nRunning simulations...")
t1, v1 = run_sim(1.0)
t2, v2 = run_sim(1000.0)

stim_key = f"{stim_section.name()}(0.500)"

#=====================================================================
# Build ordered lists / nicknames for soma / non-soma segments
# =====================================================================
soma_segment_keys = []
non_soma_segment_keys = []

for sec, seg in soma_segments:
    key = f"{sec.name()}({seg.x:.3f})"
    soma_segment_keys.append(key)

for sec, seg in non_soma_segments:
    key = f"{sec.name()}({seg.x:.3f})"
    non_soma_segment_keys.append(key)

# 1-based indices
soma_index     = {k: i + 1 for i, k in enumerate(soma_segment_keys)}
non_soma_index = {k: i + 1 for i, k in enumerate(non_soma_segment_keys)}

# Optional: override segment_labels to be explicit and consistent
for k in soma_segment_keys:
    segment_labels[k] = f"Soma segment {soma_index[k]}"

for k in non_soma_segment_keys:
    segment_labels[k] = f"Non-soma segment {non_soma_index[k]}"

# ===================================================================================================
# STEP 8 — Node summary (map nodes → closest segment)
# ===================================================================================================
print("\nNode summary:\n")

seg_peak = {k: np.max(v2[k]) for k in v2}

for nid in node_order:

    node = nodes[nid]
    node_xyz = np.array([node['x'], node['y'], node['z']])

    best_key = None
    best_dist = float('inf')

    for sec in h.allsec():

        n3d = sec.n3d() #n3d stands for number of 3D points
        if n3d == 0:
            continue

        total_arc = sec.arc3d(n3d - 1)
        if total_arc == 0:
            continue

        arc_fracs = np.array([sec.arc3d(i) / total_arc for i in range(n3d)])

        for seg in sec:
            key = f"{sec.name()}({seg.x:.3f})"
            idx = int(np.argmin(np.abs(arc_fracs - seg.x)))

            seg_xyz = np.array([
                sec.x3d(idx),
                sec.y3d(idx),
                sec.z3d(idx)
            ])

            dist = np.linalg.norm(node_xyz - seg_xyz)

            if dist < best_dist:
                best_dist = dist
                best_key = key

    peak = seg_peak.get(best_key, None)

    # Classify the segment key
    if best_key in soma_index:
        seg_class = "SOMA"
        seg_label = f"Soma segment {soma_index[best_key]}"
    elif best_key in non_soma_index:
        seg_class = "NON-SOMA"
        seg_label = f"Non-soma segment {non_soma_index[best_key]}"
    else:
        seg_class = "UNKNOWN"
        seg_label = "Unknown segment"

    print(f"{nid:>4}  "
          f"{node['x']:>8.1f} {node['y']:>8.1f} {node['z']:>8.1f}  "
          f"{node['radius']:>6.2f}  "
          f"{peak:>8.3f}  {best_key:>30s}  {seg_class:>8s}  {seg_label}")

# ===================================================================================================
# STEP 9 — Plot results
# ===================================================================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Build a combined ordered list of all segment keys to get consistent colors
all_segment_keys = soma_segment_keys + non_soma_segment_keys
n_segments = len(all_segment_keys)

# Use a colormap for distinct colors
cmap = plt.cm.get_cmap('tab20', n_segments)

for ax, t, v, title in [
    (axes[0], t1, v1, "1 ms stimulus"),
    (axes[1], t2, v2, "1 s stimulus")
]:

    handles = []
    labels = []

    for idx, key in enumerate(all_segment_keys):

        if key not in v:
            continue

        color = cmap(idx)

        # Decide class and label text
        if key in soma_index:
            seg_id = soma_index[key]
            line_label = f"Soma segment {seg_id}"
        elif key in non_soma_index:
            seg_id = non_soma_index[key]
            line_label = f"Non-soma segment {seg_id}"
        else:
            line_label = key  # fallback

        line, = ax.plot(t / 1000.0, v[key], color=color, lw=0.8, label=line_label)
        handles.append(line)
        labels.append(line_label)

    # Highlight stimulated segment on top (keeps its own label if you want)
    if stim_key in v:
        stim_line, = ax.plot(t / 1000.0, v[stim_key], color='black', lw=2,
                             label=f"Stimulated: {segment_labels.get(stim_key, stim_key)}")
        handles.append(stim_line)
        labels.append(stim_line.get_label())

    ax.axhline(-65, linestyle='--', color='k', alpha=0.3)

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (mV)")
    ax.grid(True)

    # Legend: every segment has its own entry
    ax.legend(handles=handles, ncols=4,labels=labels, loc='upper right', fontsize=6, ncol=1)

plt.tight_layout()
plt.savefig("neuron_voltage.png")
plt.show()

print("\nSaved: neuron_voltage.png")

print("\nSaved: neuron_voltage.png")

input("Press Enter to exit...")