# =========================
# Imports
# =========================
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcbar
import vtk
import os
import sys

from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from collections import defaultdict

from neuron import h, gui
from neuron.units import ms, mV, um

print(f'VTK version: {vtk.VTK_VERSION}')

# ===================================================================================================
# POSTER SIZING PARAMETERS — edit these to scale the entire pipeline at once
# ===================================================================================================
# Every visual property in every plot references one of these constants.
# Change a value here and it propagates everywhere automatically.
#
#  FONT_TITLE    — plot/panel title text size
#  FONT_LABEL    — x/y axis label text size
#  FONT_TICK     — axis tick number size
#  FONT_LEGEND   — legend entry text size
#  FONT_CBAR     — colorbar label and tick size
#  FONT_ANNOT    — in-plot annotations (e.g. "Axon" arrow label)
#  FONT_SUPTITLE — figure-level super-title (composite figure)
#
#  LW_MORPH      — morphology edge line width (gradient plots)
#  LW_MORPH_OUT  — outline behind each morphology edge
#  LW_TRACE      — voltage trace line width
#  LW_SKEL_SEL   — selected-segment line width in skeleton plot
#  LW_SKEL_SOMA  — soma-adjacent line width in skeleton plot
#  LW_SKEL_GREY  — unselected/grey branch line width
#  LW_SKEL_AXON  — axon line width in skeleton plot
#
#  MS_SOMA       — soma marker size (scatter s= units)
#  MS_STIM       — stim site star marker size
#  MS_DOT_BRANCH — dot size in branch-order plot
#  MS_DOT_DIST   — dot size in TTM/FWHM distance plots
#  MS_DOT_CLUST  — dot size in cluster distance panel
#
#  FIG_GRADIENT  — (width, height) for morphology gradient figures
#  FIG_COMPOSITE — (width, height) for skeleton + traces composite figure
#  FIG_VOLTAGE   — (width, height) for clustered voltage figure
#  FIG_BRANCH    — (width, height) for branch-order figure
#  FIG_DISTANCE  — (width, height) for TTM/FWHM distance figure (3 rows)
#
#  CBAR_FRACTION — colorbar width as fraction of axes (gradient plots)
#  CBAR_THICK    — colorbar thickness in pixels (distance/branch plots)
# ===================================================================================================

# ===================================================================================================
# POSTER SIZING PARAMETERS
# These constants control text size, line thickness, marker size, and figure size globally.
# Change them here once and the visual updates propagate through the whole script.
# ===================================================================================================

# ----- Font sizes -----
FONT_TITLE    = 24   # subplot titles
FONT_LABEL    = 20   # x/y axis labels
FONT_TICK     = 18   # axis tick labels
FONT_LEGEND   = 17   # legend text
FONT_CBAR     = 17   # colorbar label + ticks
FONT_ANNOT    = 16   # annotations inside plots
FONT_SUPTITLE = 26   # figure-level title

# ----- Line widths -----
LW_MORPH      = 3.2  # morphology branch line width
LW_MORPH_OUT  = 4.4  # black outline behind morphology lines
LW_TRACE      = 3.4  # voltage trace thickness
LW_SKEL_SEL   = 5.0  # highlighted selected arbor thickness
LW_SKEL_SOMA  = 8.0  # soma-adjacent arbor thickness
LW_SKEL_GREY  = 2.2  # non-selected grey arbor thickness
LW_SKEL_AXON  = 3.4  # axon thickness

# ----- Marker sizes -----
MS_SOMA       = 260  # soma marker size
MS_STIM       = 430  # stimulus site star size
MS_DOT_BRANCH = 80   # branch-order scatter dots
MS_DOT_DIST   = 85   # TTM/FWHM scatter dots
MS_DOT_CLUST  = 95   # cluster scatter dots

# ----- Figure sizes -----
FIG_GRADIENT  = (11, 19)    # morphology gradient figures
FIG_COMPOSITE = (23, 12.5)  # tree + current + voltage composite
FIG_VOLTAGE   = (16, 12)    # clustered voltage figure
FIG_BRANCH    = (21, 14)    # branch-order figure
FIG_DISTANCE  = (14, 21)    # TTM/FWHM figure

# ----- Colorbar sizing -----
CBAR_FRACTION = 0.050
CBAR_THICK    = 22

# Apply global matplotlib defaults so anything not explicitly set also scales up
plt.rcParams.update({
    'font.size':        FONT_TICK,
    'axes.titlesize':   FONT_TITLE,
    'axes.labelsize':   FONT_LABEL,
    'xtick.labelsize':  FONT_TICK,
    'ytick.labelsize':  FONT_TICK,
    'legend.fontsize':  FONT_LEGEND,
    'figure.titlesize': FONT_SUPTITLE,
    'lines.linewidth':  LW_TRACE,
})

# =========================
# Load NEURON tools
# =========================
h.load_file("stdrun.hoc")
h.load_file("import3d.hoc")

# =========================
# Load SWC file
# =========================
swc_path = '/home/jd/NeuronProject/CloudvolNeuron_Test/SWC_files/76182_reRoot_reSample_5000.swc'

print("File exists?", os.path.exists(swc_path))
print("Loading SWC file:", swc_path)

reader = h.Import3d_SWC_read()
reader.input(swc_path)

importer = h.Import3d_GUI(reader, 1)

class Cell:
    def __init__(self):
        self.sl = h.SectionList()

cell = Cell()
importer.instantiate(cell)

# ── Scale NEURON's internal 3D coordinates from nm to um ─────────────────────
NM_TO_UM = 1000.0  # FIX #10: defined once here; removed the duplicate inside the SWC parse loop
for sec in h.allsec():
    n3d = sec.n3d()
    if n3d == 0:
        continue
    pts = []
    for i in range(n3d):
        pts.append((sec.x3d(i)    / NM_TO_UM,
                    sec.y3d(i)    / NM_TO_UM,
                    sec.z3d(i)    / NM_TO_UM,
                    sec.diam3d(i) / NM_TO_UM))
    h.pt3dclear(sec=sec)
    for x, y, z, d in pts:
        h.pt3dadd(x, y, z, d, sec=sec)

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
# FIX #13: SWC parsing moved BEFORE the nseg-fixing loop so node_order/nodes are available
#          for any downstream code that might reference them immediately after Step 1.

nodes      = {}
node_order = []

# FIX #10: NM_TO_UM already defined above; no redefinition inside this loop
with open(swc_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('#') or line == '':
            continue
        parts     = line.split()
        node_id   = int(parts[0])
        node_type = int(parts[1])
        x         = float(parts[2]) / NM_TO_UM
        y         = float(parts[3]) / NM_TO_UM
        z         = float(parts[4]) / NM_TO_UM
        radius    = float(parts[5]) / NM_TO_UM
        parent_id = int(parts[6])
        nodes[node_id] = {
            'node_id': node_id, 'type': node_type,
            'x': x, 'y': y, 'z': z,
            'radius': radius, 'parent_id': parent_id
        }
        node_order.append(node_id)

print(f"Expected SWC edges: {len(nodes) - 1}")

# FIX #13: nseg loop moved here, AFTER nodes is populated (logical ordering)
total_fixed_segments = 0
for sec in h.allsec():
    n3d = sec.n3d()
    if n3d > 1:
        sec.nseg = n3d - 1
    else:
        sec.nseg = 1
    total_fixed_segments += int(sec.nseg)

print("\nAfter fixing nseg:")
print("Total NEURON segments:", total_fixed_segments)

# Print bounding box
all_x = [nodes[n]['x'] for n in nodes]
all_y = [nodes[n]['y'] for n in nodes]
all_z = [nodes[n]['z'] for n in nodes]
print(f"\nSWC coordinate bounding box:")
print(f"  x: {min(all_x):.1f} to {max(all_x):.1f}")
print(f"  y: {min(all_y):.1f} to {max(all_y):.1f}")
print(f"  z: {min(all_z):.1f} to {max(all_z):.1f}")
print(f"  If these values are > 10,000 the coordinates are likely in nanometres.")

edge_lengths = []
for nid, d in nodes.items():
    parent = d['parent_id']
    if parent == -1:
        continue
    n  = nodes[nid]
    p  = nodes[parent]
    dx = n['x'] - p['x']
    dy = n['y'] - p['y']
    dz = n['z'] - p['z']
    edge_lengths.append(np.sqrt(dx**2 + dy**2 + dz**2))

edge_lengths = np.array(edge_lengths)
print(f"\nSWC edge length statistics:")
print(f"  Min    : {edge_lengths.min():.1f}")
print(f"  Max    : {edge_lengths.max():.1f}")
print(f"  Mean   : {edge_lengths.mean():.1f}")
print(f"  Median : {np.median(edge_lengths):.1f}")

# =====================================================================
# BUILD TRUE SECTIONS FROM SWC GRAPH
# =====================================================================
children = defaultdict(list)
for nid, d in nodes.items():
    parent = d['parent_id']
    if parent != -1:
        children[parent].append(nid)

def is_branch(nid): return len(children[nid]) > 1
def is_leaf(nid):   return len(children[nid]) == 0
def is_root(nid):   return nodes[nid]['parent_id'] == -1

sections      = []
visited_edges = set()

for nid in nodes:
    parent = nodes[nid]['parent_id']
    if is_root(nid) or is_branch(nid) or (parent != -1 and is_branch(parent)):
        for child in children[nid]:
            path = [nid, child]
            visited_edges.add((nid, child))
            current = child
            while not is_branch(current) and not is_leaf(current):
                next_node = children[current][0]
                if (current, next_node) in visited_edges:
                    break
                path.append(next_node)
                visited_edges.add((current, next_node))
                current = next_node
            sections.append(path)

print(f"\nTOTAL TRUE SECTIONS: {len(sections)}")

# ===================================================================================================
# STEP 2 — Identify soma nodes
# ===================================================================================================
soma_node_ids = {
    nid for nid, d in nodes.items()
    if nid == 1 or d['parent_id'] == 1
}
non_soma_node_ids = set(nodes.keys()) - soma_node_ids
soma_coords = np.array([[nodes[n]['x'], nodes[n]['y'], nodes[n]['z']] for n in soma_node_ids])

print(f"Soma nodes: {len(soma_node_ids)}")
print(f"Non-soma nodes: {len(non_soma_node_ids)}")

# ===================================================================================================
# STEP 3 — Classify SEGMENTS
# ===================================================================================================
SOMA_THRESHOLD = 1.5

soma_segments     = []
non_soma_segments = []
segment_labels    = {}

soma_count     = 0
non_soma_count = 0

for sec in h.allsec():
    n3d_pts = sec.n3d()
    for seg in sec:
        is_soma = False
        if n3d_pts > 0:
            total_arc = sec.arc3d(n3d_pts - 1)
            if total_arc > 0:
                arc_fracs = np.array([sec.arc3d(i) / total_arc for i in range(n3d_pts)])
                idx = int(np.argmin(np.abs(arc_fracs - seg.x)))
                seg_xyz = np.array([sec.x3d(idx), sec.y3d(idx), sec.z3d(idx)])
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
    sec.cm  = 1.0
    sec.insert('pas')
    for seg in sec:
        seg.pas.g = 0.00005
        seg.pas.e = -65.0

# ===================================================================================================
# STEP 5 — Find stimulation site (non-soma leaf)
# ===================================================================================================
soma_secs = {sec for sec, _ in soma_segments}
if not soma_secs:
    soma_secs = {list(h.allsec())[0]}

sec_ref_map = {sec: h.SectionRef(sec=sec) for sec in h.allsec()}

def path_distance_from_soma(sec, soma_secs, dx=0.5):
    if sec in soma_secs:
        return 0.0
    dist = sec.L * dx
    sr = sec_ref_map[sec]
    while sr.has_parent():
        parent = sr.parent().sec
        if parent in soma_secs:
            dist += parent.L * 0.5
            break
        else:
            dist += parent.L
            sr = sec_ref_map[parent]
    return dist

def get_children(sec):
    return list(sec.children())

all_sections  = list(h.allsec())
leaf_sections = [sec for sec in all_sections if len(get_children(sec)) == 0]
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

def _section_mean_diam(sec):
    """Mean diameter (um) over all pt3d points of a section."""
    n = sec.n3d()
    if n == 0:
        return sec.diam   # fallback to uniform diameter
    return float(np.mean([sec.diam3d(i) for i in range(n)]))

def _unbranched_trunk_length(root_sec):
    """
    Walk from root_sec toward distal tips, always following the SINGLE child
    at each step.  Stop as soon as we hit a bifurcation (2+ children) or a leaf.
    Return total cable length of this unbranched trunk.
    """
    total = root_sec.L
    sec   = root_sec
    while True:
        kids = list(sec.children())
        if len(kids) != 1:   # bifurcation or leaf → stop
            break
        sec    = kids[0]
        total += sec.L
    return total

def _all_descendant_sections(root_sec):
    """Return the set of all sections reachable from root_sec (inclusive) by DFS."""
    result = set()
    stack  = [root_sec]
    while stack:
        s = stack.pop()
        if s in result:
            continue
        result.add(s)
        stack.extend(list(s.children()))
    return result

# Candidate axon roots = direct children of soma sections (non-soma themselves)
soma_children = []
for s in soma_secs:
    for child in list(s.children()):
        if child not in soma_secs:
            soma_children.append(child)

# Compute median diameter across all non-soma sections (used as thinness threshold)
all_non_soma_sec_list = list(non_soma_sections)
median_diam = float(np.median([_section_mean_diam(s) for s in all_non_soma_sec_list])) \
              if all_non_soma_sec_list else 1.0

print(f"\nMedian non-soma section diameter: {median_diam:.3f} um")
print(f"Soma direct children (axon candidates): {len(soma_children)}")

# Score each candidate: primary key = unbranched trunk length (longer = more axon-like),
# secondary key = thin diameter (below median = more axon-like)
best_axon_root  = None
best_trunk_len  = -1.0

for cand in soma_children:
    trunk_len  = _unbranched_trunk_length(cand)
    mean_diam  = _section_mean_diam(cand)
    is_thin    = mean_diam < median_diam
    # Prefer the longest unbranched trunk; among ties prefer the thinner one.
    # We give a 20 % bonus to thin candidates so thinness can break close ties
    # but a grossly shorter thin section won't beat a clearly longer thick one.
    effective_len = trunk_len * (1.20 if is_thin else 1.0)
    print(f"  Candidate {cand.name():30s}  trunk={trunk_len:8.1f} um  "
          f"diam={mean_diam:.3f} um  thin={is_thin}  eff={effective_len:.1f}")
    if effective_len > best_trunk_len:
        best_trunk_len = effective_len
        best_axon_root = cand

# Collect the entire axonal tree (trunk + all collaterals branching off it)
if best_axon_root is not None:
    axon_sections = _all_descendant_sections(best_axon_root)
else:
    axon_sections = set()

print(f"\nAxon root section: {best_axon_root.name() if best_axon_root else 'None'}")
print(f"Total axon sections: {len(axon_sections)}")
print(f"Axon unbranched trunk length: {_unbranched_trunk_length(best_axon_root):.1f} um"
      if best_axon_root else "")

# Build a per-segment axon flag dict for fast lookup later
# Key: segment key string → True/False
seg_is_axon = {}
for sec in h.allsec():
    is_ax = sec in axon_sections
    for seg in sec:
        key = f"{sec.name()}({seg.x:.3f})"
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

def _swc_branch_order_map(nodes, node_order, children):
    """BFS branch order: root=1, increments at each bifurcation."""
    from collections import deque
    root = next(nid for nid in node_order if nodes[nid]['parent_id'] == -1)
    order = {}
    q = deque([(root, 1)])
    while q:
        nid, lvl = q.popleft()
        order[nid] = lvl
        kids = children[nid]
        nxt  = lvl + 1 if len(kids) >= 2 else lvl
        for child in kids:
            if child not in order:
                q.append((child, nxt))
    return order

_swc_order = _swc_branch_order_map(nodes, node_order, children)

def _section_distal_swc_order(sec, swc_order, nodes):
    """
    Return the branch order of the SWC node geometrically nearest to the
    distal tip (last pt3d point) of a NEURON section.
    """
    n3d_n = sec.n3d()
    if n3d_n == 0:
        return 1
    dx, dy, dz = sec.x3d(n3d_n - 1), sec.y3d(n3d_n - 1), sec.z3d(n3d_n - 1)
    best_dist = float('inf')
    best_order = 1
    for nid, nd in nodes.items():
        d = (nd['x'] - dx)**2 + (nd['y'] - dy)**2 + (nd['z'] - dz)**2
        if d < best_dist:
            best_dist  = d
            best_order = swc_order.get(nid, 1)
    return best_order

dendritic_leaves = [sec for sec in leaf_sections
                    if sec in non_soma_sections and sec not in axon_sections]
if not dendritic_leaves:
    dendritic_leaves = [sec for sec in leaf_sections if sec not in axon_sections]
if not dendritic_leaves:
    dendritic_leaves = leaf_sections   # last-resort fallback

# Annotate each dendritic leaf with (path_distance, branch_order)
leaf_info = [
    (path_distance_from_soma(sec, soma_secs),
     _section_distal_swc_order(sec, _swc_order, nodes),
     sec)
    for sec in dendritic_leaves
]

# Find the maximum branch order among all dendritic leaves
max_dendritic_order = max(bo for _, bo, _ in leaf_info)
print(f"\nMax dendritic branch order among leaves: {max_dendritic_order}")

# Keep only leaves AT the maximum branch order, then pick longest path distance
deepest_leaves = [(pd, bo, sec) for pd, bo, sec in leaf_info if bo == max_dendritic_order]
if not deepest_leaves:
    deepest_leaves = leaf_info   # fallback: shouldn't happen

deepest_leaves.sort(key=lambda x: x[0])   # sort ascending by path distance
max_dist, _, stim_section = deepest_leaves[-1]  # most distal among deepest

# Full ranked list for the diagnostic printout (all non-soma leaves)
all_leaf_distances = sorted(
    [(path_distance_from_soma(sec, soma_secs), sec) for sec in
     ([sec for sec in leaf_sections if sec in non_soma_sections] or leaf_sections)],
    key=lambda x: x[0]
)

n3d = stim_section.n3d()
distal_idx  = n3d - 1
distal_xyz  = np.array([stim_section.x3d(distal_idx),
                        stim_section.y3d(distal_idx),
                        stim_section.z3d(distal_idx)])
total_arc   = stim_section.arc3d(n3d - 1)
arc_fracs   = np.array([stim_section.arc3d(i) / total_arc for i in range(n3d)])

best_seg      = None
best_dist_seg = float('inf')
for seg in stim_section:
    idx     = int(np.argmin(np.abs(arc_fracs - seg.x)))
    seg_xyz = np.array([stim_section.x3d(idx),
                        stim_section.y3d(idx),
                        stim_section.z3d(idx)])
    d = np.linalg.norm(seg_xyz - distal_xyz)
    if d < best_dist_seg:
        best_dist_seg = d
        best_seg      = seg

stim_seg = best_seg
stim_x   = float(stim_seg.x)
stim_key = f"{stim_section.name()}({stim_x:.3f})"

is_stim_axon = stim_section in axon_sections
print(f"\nStim site : {stim_section.name()}({stim_x:.3f})")
print(f"Path dist : {max_dist:.2f} um from soma")
print(f"Is axon?  : {is_stim_axon}  ← should be False")

# Print ALL non-soma leaves ranked by path distance; flag axon and chosen stim site
print("\nAll non-soma leaf sections ranked by path distance:")
for rank, (dist, sec) in enumerate(reversed(all_leaf_distances)):
    axon_tag = " [AXON]"   if sec in axon_sections else ""
    stim_tag = " <-- STIM" if sec is stim_section  else ""
    print(f"  rank {rank+1:>2}  dist={dist:>10.1f} um  {sec.name()}{axon_tag}{stim_tag}")

# ===================================================================================================
# STEP 6 — Simulation function
# ===================================================================================================
T_STOP = 5000.0
DT     = 0.025
I_AMP  = 0.002

def run_sim(stim_dur):
    h.dt     = DT
    h.tstop  = T_STOP
    h.v_init = -65

    stim       = h.IClamp(stim_section(stim_x))
    stim.delay = 100
    stim.dur   = stim_dur
    stim.amp   = I_AMP

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

# =====================================================================
# Build ordered segment key lists
# =====================================================================
soma_segment_keys     = [f"{sec.name()}({seg.x:.3f})" for sec, seg in soma_segments]
non_soma_segment_keys = [f"{sec.name()}({seg.x:.3f})" for sec, seg in non_soma_segments]

soma_index     = {k: i + 1 for i, k in enumerate(soma_segment_keys)}
non_soma_index = {k: i + 1 for i, k in enumerate(non_soma_segment_keys)}

for k in soma_segment_keys:
    segment_labels[k] = f"Soma segment {soma_index[k]}"
for k in non_soma_segment_keys:
    segment_labels[k] = f"Non-soma segment {non_soma_index[k]}"

all_segment_keys = soma_segment_keys + non_soma_segment_keys

# ===================================================================================================
# STEP 8 — PEAK-RATIO METRIC
# ===================================================================================================

def compute_peak_ratio(v_dict, reference_key, all_keys, t, t_start=100.0, t_end=300.0):
    mask     = (t >= t_start) & (t <= t_end)
    ref_full = v_dict[reference_key]
    ref_base = np.mean(ref_full[t < t_start])
    ref_peak = np.max(ref_full[mask] - ref_base)

    ratio_values = {}
    for key in all_keys:
        if key not in v_dict or ref_peak <= 0:
            ratio_values[key] = 0.0
            continue
        x_full = v_dict[key]
        x_base = np.mean(x_full[t < t_start])
        x_peak = np.max(x_full[mask] - x_base)
        ratio_values[key] = float(np.clip(x_peak / ref_peak, 0.0, 1.0))
    return ratio_values


def compute_time_to_max_and_fwhm(v_dict, all_keys, t, t_start=100.0, t_end=300.0):
    mask = (t >= t_start) & (t <= t_end)
    time_to_max = {}
    fwhm = {}

    for key in all_keys:
        if key not in v_dict:
            time_to_max[key] = np.nan
            fwhm[key]        = np.nan
            continue

        trace_full = v_dict[key]
        baseline   = np.mean(trace_full[t < t_start])
        trace      = trace_full[mask] - baseline
        tt         = t[mask]

        if len(trace) == 0:
            time_to_max[key] = np.nan
            fwhm[key]        = np.nan
            continue

        peak_idx = int(np.argmax(trace))
        peak_val = float(trace[peak_idx])

        if peak_val <= 0:
            time_to_max[key] = np.nan
            fwhm[key]        = np.nan
            continue

        time_to_max[key] = float(tt[peak_idx] - t_start)

        half_val = 0.5 * peak_val
        above    = np.where(trace >= half_val)[0]

        if len(above) == 0:
            fwhm[key] = np.nan
        else:
            fwhm[key] = float(tt[above[-1]] - tt[above[0]])

    return time_to_max, fwhm


# FIX #2 / #3: plot_ttm_fwhm_scatter and filter_segments_by_shape defined ONCE here.
# The duplicate definitions that appeared in Step 8d and again at the top of Step 9
# have been removed. Only one canonical copy of each function exists below.
# =====================================================================
# Compute space constant (lambda) from attenuation vs distance
# =====================================================================
def compute_space_constant(corr_values, distances, keys):
    """
    Estimate space constant λ using:
        log(V/V0) = -x / λ
    """

    import numpy as np

    xs = []
    ys = []

    for k in keys:
        if k not in corr_values or k not in distances:
            continue

        r = corr_values[k]
        d = distances[k]

        # Skip invalid values
        if r <= 0 or r >= 1.0:
            continue

        xs.append(d)
        ys.append(np.log(r))

    xs = np.array(xs)
    ys = np.array(ys)

    if len(xs) < 5:
        print("Not enough points to estimate lambda.")
        return None

    # Linear fit
    slope, intercept = np.polyfit(xs, ys, 1)

    # λ = -1 / slope
    lambda_um = -1.0 / slope

    return lambda_um

def plot_ttm_fwhm_vs_distance(axes, ttm, fwhm, dist_map, seg_key_to_level,
                               all_keys, run_label,
                               filtered_keys=None,
                               ttm_max=None, fwhm_max=None, fwhm_min=None):
    """
    Three scatter plots (one per row of `axes`):
      Row 0 : Time-to-max (ms)  vs distance from stim site (µm)
      Row 1 : FWHM (ms)         vs distance from stim site (µm)
      Row 2 : TTM vs FWHM — filter diagnostic (passed = coloured, failed = grey)

    Rows 0–1: colour encodes branch order (plasma colormap).
    Row 2: shows the filter thresholds as dashed lines so it is clear
           which segments survived into clustering.
    """
    # ── Collect valid data points ────────────────────────────────────────────
    dists, ttms, fwhms, orders, keys_valid = [], [], [], [], []
    for key in all_keys:
        d  = dist_map.get(key, np.nan)
        tm = ttm.get(key, np.nan)
        fw = fwhm.get(key, np.nan)
        bo = seg_key_to_level.get(key, 1)
        if not (np.isfinite(d) and np.isfinite(tm) and np.isfinite(fw)):
            continue
        dists.append(d);  ttms.append(tm);  fwhms.append(fw)
        orders.append(bo); keys_valid.append(key)

    dists  = np.array(dists)
    ttms   = np.array(ttms)
    fwhms  = np.array(fwhms)
    orders = np.array(orders)

    passed = np.array([k in filtered_keys for k in keys_valid]) \
             if filtered_keys is not None else np.ones(len(keys_valid), dtype=bool)

    max_order = int(orders.max()) if len(orders) else 1
    cmap      = plt.cm.plasma
    norm      = mcolors.Normalize(vmin=1, vmax=max_order)

    # ── Row 0: TTM vs distance ────────────────────────────────────────────────
    ax0 = axes[0]
    sc0 = ax0.scatter(dists, ttms, c=orders, cmap=cmap, norm=norm,
                      s=MS_DOT_DIST, edgecolors='black', linewidths=0.4, alpha=0.85)
    ax0.set_title(f"Time to max vs distance from stim — {run_label}",
                  fontsize=FONT_TITLE, fontweight='bold')
    ax0.set_xlabel("Distance from stim site (µm)", fontsize=FONT_LABEL)
    ax0.set_ylabel("Time to max (ms)", fontsize=FONT_LABEL)
    ax0.tick_params(labelsize=FONT_TICK)
    ax0.grid(True, alpha=0.22)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    cbar0 = plt.colorbar(sc0, ax=ax0, pad=0.02)
    cbar0.set_label("Branch order", fontsize=FONT_CBAR)
    cbar0.set_ticks(range(1, max_order + 1))
    cbar0.ax.tick_params(labelsize=FONT_CBAR)

    # ── Row 1: FWHM vs distance ───────────────────────────────────────────────
    ax1 = axes[1]
    sc1 = ax1.scatter(dists, fwhms, c=orders, cmap=cmap, norm=norm,
                      s=MS_DOT_DIST, edgecolors='black', linewidths=0.4, alpha=0.85)
    ax1.set_title(f"FWHM vs distance from stim — {run_label}",
                  fontsize=FONT_TITLE, fontweight='bold')
    ax1.set_xlabel("Distance from stim site (µm)", fontsize=FONT_LABEL)
    ax1.set_ylabel("FWHM (ms)", fontsize=FONT_LABEL)
    ax1.tick_params(labelsize=FONT_TICK)
    ax1.grid(True, alpha=0.22)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    cbar1 = plt.colorbar(sc1, ax=ax1, pad=0.02)
    cbar1.set_label("Branch order", fontsize=FONT_CBAR)
    cbar1.set_ticks(range(1, max_order + 1))
    cbar1.ax.tick_params(labelsize=FONT_CBAR)

    # ── Row 2: TTM vs FWHM — filter diagnostic ───────────────────────────────
    ax2 = axes[2]
    if (~passed).any():
        ax2.scatter(ttms[~passed], fwhms[~passed],
                    color='#cccccc', s=MS_DOT_DIST - 10, edgecolors='none',
                    alpha=0.6, label='Filtered out', zorder=2)
    if passed.any():
        sc2 = ax2.scatter(ttms[passed], fwhms[passed],
                          c=orders[passed], cmap=cmap, norm=norm,
                          s=MS_DOT_DIST, edgecolors='black', linewidths=0.4,
                          alpha=0.88, label='Passed filter', zorder=3)
        cbar2 = plt.colorbar(sc2, ax=ax2, pad=0.02)
        cbar2.set_label("Branch order", fontsize=FONT_CBAR)
        cbar2.set_ticks(range(1, max_order + 1))
        cbar2.ax.tick_params(labelsize=FONT_CBAR)

    if ttm_max is not None:
        ax2.axvline(ttm_max, color='steelblue', linestyle='--', lw=2.0,
                    label=f"TTM max = {ttm_max:.0f} ms")
    if fwhm_max is not None:
        ax2.axhline(fwhm_max, color='darkorange', linestyle='--', lw=2.0,
                    label=f"FWHM max = {fwhm_max:.0f} ms")
    if fwhm_min is not None:
        ax2.axhline(fwhm_min, color='darkgreen', linestyle='--', lw=2.0,
                    label=f"FWHM min = {fwhm_min:.0f} ms")

    ax2.set_title(f"TTM vs FWHM — filter diagnostic — {run_label}",
                  fontsize=FONT_TITLE, fontweight='bold')
    ax2.set_xlabel("Time to max (ms)", fontsize=FONT_LABEL)
    ax2.set_ylabel("FWHM (ms)", fontsize=FONT_LABEL)
    ax2.tick_params(labelsize=FONT_TICK)
    ax2.grid(True, alpha=0.22)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(fontsize=FONT_LEGEND, loc='best')


def filter_segments_by_shape(att_values, ttm, fwhm, all_keys,
                              ttm_max=None, fwhm_max=None, fwhm_min=None):
    """Keep only segments whose peak-shape satisfies TTM/FWHM criteria."""
    filtered_keys = []
    for key in all_keys:
        if key not in att_values:
            continue
        ttm_val  = ttm.get(key, np.nan)
        fwhm_val = fwhm.get(key, np.nan)
        if np.isnan(ttm_val) or np.isnan(fwhm_val):
            continue
        if ttm_max  is not None and ttm_val  > ttm_max:
            continue
        if fwhm_max is not None and fwhm_val > fwhm_max:
            continue
        if fwhm_min is not None and fwhm_val < fwhm_min:
            continue
        filtered_keys.append(key)
    return filtered_keys


# Compute metrics
corr1 = compute_peak_ratio(v_dict=v1, reference_key=stim_key,
                            all_keys=all_segment_keys, t=t1,
                            t_start=100.0, t_end=250.0)
corr2 = compute_peak_ratio(v_dict=v2, reference_key=stim_key,
                            all_keys=all_segment_keys, t=t2,
                            t_start=100.0, t_end=1100.0)

ttm1, fwhm1 = compute_time_to_max_and_fwhm(v1, all_segment_keys, t1,
                                             t_start=100.0, t_end=250.0)
ttm2, fwhm2 = compute_time_to_max_and_fwhm(v2, all_segment_keys, t2,
                                             t_start=100.0, t_end=1100.0)

filtered_keys1 = filter_segments_by_shape(corr1, ttm1, fwhm1, all_segment_keys,
                                           ttm_max=15.0, fwhm_max=25.0)
filtered_keys2 = filter_segments_by_shape(corr2, ttm2, fwhm2, all_segment_keys,
                                           ttm_max=1000.0, fwhm_min=5.0)

if len(filtered_keys2) == 0:
    print("WARNING: 1 s shape filter removed all segments. Falling back to all segments.")
    filtered_keys2 = list(all_segment_keys)

print(f"Filtered 1 ms segments: {len(filtered_keys1)}")
print(f"Filtered 1 s segments:  {len(filtered_keys2)}")
print(f"\nPeak ratio computed for {len(corr1)} segments (burst run)")
print(f"Peak ratio computed for {len(corr2)} segments (sustained run)")
print(f"  Burst     range: [{min(corr1.values()):.3f}, {max(corr1.values()):.3f}]")
print(f"  Sustained range: [{min(corr2.values()):.3f}, {max(corr2.values()):.3f}]")

# ─── Map every SWC node to its nearest NEURON segment ────────────────────────

def build_node_to_segment_map():
    """For every SWC node, find the NEURON segment whose 3D position is closest."""
    node_to_seg = {}
    for nid in node_order:
        node     = nodes[nid]
        node_xyz = np.array([node['x'], node['y'], node['z']])
        best_key  = None
        best_dist = float('inf')
        for sec in h.allsec():
            n3d_n = sec.n3d()
            if n3d_n == 0:
                continue
            ta = sec.arc3d(n3d_n - 1)
            if ta == 0:
                continue
            af = np.array([sec.arc3d(i) / ta for i in range(n3d_n)])
            for seg in sec:
                key     = f"{sec.name()}({seg.x:.3f})"
                idx     = int(np.argmin(np.abs(af - seg.x)))
                seg_xyz = np.array([sec.x3d(idx), sec.y3d(idx), sec.z3d(idx)])
                dist    = np.linalg.norm(node_xyz - seg_xyz)
                if dist < best_dist:
                    best_dist = dist
                    best_key  = key
        node_to_seg[nid] = best_key
    return node_to_seg


def choose_representative_soma_key(soma_segments, nodes):
    soma_root_xyz = np.array([nodes[1]['x'], nodes[1]['y'], nodes[1]['z']])
    best_key  = None
    best_dist = float('inf')
    for sec, seg in soma_segments:
        key   = f"{sec.name()}({seg.x:.3f})"
        n3d_n = sec.n3d()
        if n3d_n == 0:
            continue
        ta = sec.arc3d(n3d_n - 1)
        if ta == 0:
            continue
        af  = np.array([sec.arc3d(i) / ta for i in range(n3d_n)])
        idx = int(np.argmin(np.abs(af - seg.x)))
        seg_xyz = np.array([sec.x3d(idx), sec.y3d(idx), sec.z3d(idx)])
        d = np.linalg.norm(seg_xyz - soma_root_xyz)
        if d < best_dist:
            best_dist = d
            best_key  = key
    return best_key


print("\nBuilding node-to-segment map for morphology colouring...")
node_to_segment = build_node_to_segment_map()


def build_segment_distance_from_stim():
    seg_xyz = {}
    for sec in h.allsec():
        n3d_n = sec.n3d()
        if n3d_n == 0:
            continue
        ta = sec.arc3d(n3d_n - 1)
        if ta == 0:
            continue
        af = np.array([sec.arc3d(i) / ta for i in range(n3d_n)])
        for seg in sec:
            key     = f"{sec.name()}({seg.x:.3f})"
            idx     = int(np.argmin(np.abs(af - seg.x)))
            seg_xyz[key] = np.array([sec.x3d(idx), sec.y3d(idx), sec.z3d(idx)])

    if stim_key not in seg_xyz:
        raise ValueError(f"Stim segment {stim_key} not found in segment coordinate map.")

    stim_xyz = seg_xyz[stim_key]
    return {key: float(np.linalg.norm(xyz - stim_xyz)) for key, xyz in seg_xyz.items()}


segment_distance_from_stim = build_segment_distance_from_stim()

# ===================================================================================================
# RESULT 1/2 — MORPHOLOGY GRADIENT PLOT
# ===================================================================================================

def plot_morphology_gradient(nodes, node_order, node_to_segment, corr_values,
                              title="Morphology — peak-ratio gradient"):
    fig, ax = plt.subplots(figsize=FIG_GRADIENT)

    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.cm.Reds

    for nid in node_order:
        node   = nodes[nid]
        parent = node['parent_id']
        if parent == -1:
            continue

        seg_key    = node_to_segment.get(nid)
        metric_val = corr_values.get(seg_key, None)
        if metric_val is None and seg_key and ').' in seg_key:
            seg_key    = seg_key.split(').', 1)[1]
            metric_val = corr_values.get(seg_key, None)

        x0, y0 = nodes[parent]['x'], nodes[parent]['y']
        x1, y1 = node['x'],          node['y']

        ax.plot([x0, x1], [y0, y1], color='black', lw=LW_MORPH_OUT, alpha=0.7,
                solid_capstyle='round', zorder=1)

        color = cmap(norm(metric_val)) if metric_val is not None else '#c8c8c8'
        ax.plot([x0, x1], [y0, y1], color=color, lw=LW_MORPH,
                solid_capstyle='round', zorder=2)

    soma_xs = [nodes[n]['x'] for n in nodes if n == 1 or nodes[n]['parent_id'] == 1]
    soma_ys = [nodes[n]['y'] for n in nodes if n == 1 or nodes[n]['parent_id'] == 1]
    ax.scatter(np.mean(soma_xs), np.mean(soma_ys),
               s=MS_SOMA, marker='o', color='gold',
               edgecolors='black', linewidths=1.5, zorder=5, label='Soma')
    ax.scatter(distal_xyz[0], distal_xyz[1],
               s=MS_STIM, marker='*', color='black',
               edgecolors='white', linewidths=1.0, zorder=6, label='Stim site')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=CBAR_FRACTION, pad=0.04)
    cbar.set_label("Peak ratio vs stim site", fontsize=FONT_CBAR)
    cbar.ax.tick_params(labelsize=FONT_CBAR)

    ax.set_aspect('equal')
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold')
    ax.set_xlabel("x (µm)", fontsize=FONT_LABEL)
    ax.set_ylabel("y (µm)", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    # More transparent legend so morphology underneath is easier to see.
    ax.legend(
    fontsize=FONT_LEGEND,
    loc='upper right',
    framealpha=0.55,
    facecolor='white',
    edgecolor='black'
)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("neuron_morphology_gradient.png", dpi=150, bbox_inches='tight')
    print("Saved: neuron_morphology_gradient.png")
    plt.show()


plot_morphology_gradient(nodes, node_order, node_to_segment, corr1,
                         title="Morphology — peak-ratio gradient (Burst 1 ms)")
plot_morphology_gradient(nodes, node_order, node_to_segment, corr2,
                         title="Morphology — peak-ratio gradient (Sustained 1 s)")

# ===================================================================================================
# STEP 8b — SELECT 5 REPRESENTATIVE TRACES
# ===================================================================================================

print("\n--- SOMA DEBUG ---")
print(f"soma_segment_keys[0] = {soma_segment_keys[0] if soma_segment_keys else 'NONE'}")
print("node_to_segment for nodes 1-5:")
for nid in [1, 2, 3, 4, 5]:
    if nid in node_to_segment:
        print(f"  node {nid} -> {node_to_segment[nid]}")
print("---")


def select_five_traces(corr_values, stim_key, soma_segment_keys,
                       non_soma_segment_keys, soma_index, non_soma_index,
                       soma_segments, nodes):
    # Exclude axon segments — traces should represent dendritic compartments only
    candidates  = [k for k in non_soma_segment_keys
                   if k != stim_key and not seg_is_axon.get(k, False)]
    if not candidates:   # fallback if everything is somehow axon-labelled
        candidates = [k for k in non_soma_segment_keys if k != stim_key]
    r_vals      = np.array([corr_values.get(k, 0.0) for k in candidates])
    sorted_idx  = np.argsort(r_vals)
    lowest_key  = candidates[sorted_idx[0]]
    median_key  = candidates[sorted_idx[len(sorted_idx) // 2]]
    highest_key = candidates[sorted_idx[-1]]
    soma_key    = choose_representative_soma_key(soma_segments, nodes) if soma_segments else None

    keys = [stim_key, soma_key, lowest_key, median_key, highest_key]

    def seg_label(key):
        if key == stim_key:
            if key in soma_index:
                return f"Stim (S{soma_index[key]})"
            elif key in non_soma_index:
                return f"Stim ({non_soma_index[key]})"
            return "Stim"
        if key is None:
            return "Soma"
        if key in soma_index:
            return f"Soma (S{soma_index[key]})"
        elif key in non_soma_index:
            return f"Segment {non_soma_index[key]}"
        return "Segment"

    labels = [seg_label(stim_key), seg_label(soma_key),
              seg_label(lowest_key), seg_label(median_key), seg_label(highest_key)]
    colors = ['black', 'crimson', 'steelblue', 'darkorange', 'mediumseagreen']
    return keys, labels, colors


keys1, labels1, trace_colors = select_five_traces(
    corr1, stim_key, soma_segment_keys, non_soma_segment_keys,
    soma_index, non_soma_index, soma_segments, nodes)

keys2, labels2, _ = select_five_traces(
    corr2, stim_key, soma_segment_keys, non_soma_segment_keys,
    soma_index, non_soma_index, soma_segments, nodes)


def plot_five_traces(ax, t, v, keys, labels, colors, title,
                     stim_delay_ms=100, stim_dur_ms=1.0, xlim_s=None):
    t_s   = t / 1000.0
    x_end = xlim_s if xlim_s is not None else t_s[-1]

    for key, label, color in zip(keys, labels, colors):
        if key is None or key not in v:
            continue
        mask = t_s <= x_end
        ax.plot(t_s[mask], v[key][mask], color=color, lw=LW_TRACE, label=label, zorder=3)

    ax.axvline(stim_delay_ms / 1000.0, color='dimgrey', lw=1.5,
               linestyle='--', alpha=0.7, label='Stim onset')
    ax.axhline(-65, linestyle=':', color='k', alpha=0.25, lw=1.2)

    ax.set_xlim(0, x_end)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=FONT_LABEL)
    ax.set_ylabel("Voltage (mV)", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    # More transparent legend so traces are less obstructed.
    ax.legend(
        fontsize=FONT_LEGEND,
        loc='upper right',
        framealpha=0.55,
        facecolor='white',
        edgecolor='black'
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.18, lw=0.5)


# ===================================================================================================
# STEP 8c — SKELETON COLOUR-CODED BY SELECTED TRACES + VOLTAGE + CURRENT PANELS
# ===================================================================================================

print("\n" + "=" * 70)
print("  SOMA SEGMENTS")
print("=" * 70)
for k in soma_segment_keys:
    print(f"  {soma_index[k]:>4}  {k}")

print("\n" + "=" * 70)
print("  NON-SOMA SEGMENTS  (first 30 shown)")
print("=" * 70)
for k in non_soma_segment_keys[:30]:
    print(f"  {non_soma_index[k]:>4}  {k}")
print(f"\n  ... ({len(non_soma_segment_keys)} non-soma segments total)")

print("\n" + "=" * 110)
print(f"  {'#':>4}  {'Class':<9}  {'Segment key':<50}  "
      f"{'x (um)':>10}  {'y (um)':>10}  {'z (um)':>10}  {'diam (um)':>10}")
print("=" * 110)


def seg_centre_xyz_diam(sec, seg):
    n3d_n = sec.n3d()
    if n3d_n == 0:
        return 0.0, 0.0, 0.0, 0.0
    ta = sec.arc3d(n3d_n - 1)
    if ta == 0:
        return 0.0, 0.0, 0.0, 0.0
    af  = np.array([sec.arc3d(i) / ta for i in range(n3d_n)])
    idx = int(np.argmin(np.abs(af - seg.x)))
    return sec.x3d(idx), sec.y3d(idx), sec.z3d(idx), sec.diam3d(idx)


for sec, seg in soma_segments:
    key         = f"{sec.name()}({seg.x:.3f})"
    idx         = soma_index[key]
    x, y, z, d = seg_centre_xyz_diam(sec, seg)
    print(f"  {idx:>4}  {'SOMA':<9}  {key:<50}  {x:>10.1f}  {y:>10.1f}  {z:>10.1f}  {d:>10.3f}")

for sec, seg in non_soma_segments:
    key         = f"{sec.name()}({seg.x:.3f})"
    idx         = non_soma_index[key]
    x, y, z, d = seg_centre_xyz_diam(sec, seg)
    print(f"  {idx:>4}  {'NON-SOMA':<9}  {key:<50}  {x:>10.1f}  {y:>10.1f}  {z:>10.1f}  {d:>10.3f}")

print("=" * 110)
print(f"  Total soma: {len(soma_segments)}   Total non-soma: {len(non_soma_segments)}")
print("=" * 110)


def build_selected_colour_map(keys, colors):
    return {k: c for k, c in zip(keys, colors) if k is not None}


colour_map1        = build_selected_colour_map(keys1, trace_colors)
colour_map2        = build_selected_colour_map(keys2, trace_colors)
colour_map_combined = {**colour_map1, **colour_map2}

soma_key_check = soma_segment_keys[0] if soma_segment_keys else None
if soma_key_check and soma_key_check not in colour_map_combined:
    colour_map_combined[soma_key_check] = 'crimson'
    print(f"  WARNING: soma key was missing from colour map — force-added: {soma_key_check}")


def draw_skeleton_selected(ax, nodes, node_order, node_to_segment,
                            selected_colour_map, soma_node_ids,
                            stim_key, distal_xyz, title="Morphology skeleton"):
    GREY = '#777777'

    selected_seg_xyz = {}
    for sec in h.allsec():
        n3d_n = sec.n3d()
        if n3d_n == 0:
            continue
        ta = sec.arc3d(n3d_n - 1)
        if ta == 0:
            continue
        af = np.array([sec.arc3d(i) / ta for i in range(n3d_n)])
        for seg in sec:
            key = f"{sec.name()}({seg.x:.3f})"
            if key in selected_colour_map:
                idx = int(np.argmin(np.abs(af - seg.x)))
                selected_seg_xyz[key] = np.array([sec.x3d(idx),
                                                   sec.y3d(idx),
                                                   sec.z3d(idx)])

    edge_colours   = {}
    keys_with_edge = set()

    for nid in node_order:
        node   = nodes[nid]
        parent = node['parent_id']
        if parent == -1:
            continue
        seg_key = node_to_segment.get(nid)
        color   = selected_colour_map.get(seg_key, GREY)
        edge_colours[(parent, nid)] = color
        if seg_key in selected_colour_map:
            keys_with_edge.add(seg_key)

    for key, seg_xyz in selected_seg_xyz.items():
        if key in keys_with_edge:
            continue
        best_nid  = None
        best_dist = float('inf')
        for nid in node_order:
            nd = nodes[nid]
            d  = np.linalg.norm(seg_xyz - np.array([nd['x'], nd['y'], nd['z']]))
            if d < best_dist:
                best_dist = d
                best_nid  = nid
        if best_nid is not None:
            parent = nodes[best_nid]['parent_id']
            if parent != -1:
                edge_colours[(parent, best_nid)] = selected_colour_map[key]
            else:
                for child_nid in node_order:
                    if nodes[child_nid]['parent_id'] == best_nid:
                        edge_colours[(best_nid, child_nid)] = selected_colour_map[key]
                        break

    for nid in node_order:
        node   = nodes[nid]
        parent = node['parent_id']
        if parent == -1:
            continue
        color = edge_colours.get((parent, nid), GREY)
        if color != GREY:
            continue
        x0, y0 = nodes[parent]['x'], nodes[parent]['y']
        x1, y1 = node['x'],          node['y']
        # Axon edges drawn in cyan even when not one of the 5 selected segments.
        # Uses seg_is_axon (topology-based) because SWC type codes are all 1.
        seg_key_nid    = node_to_segment.get(nid)
        seg_key_parent = node_to_segment.get(parent)
        is_axon_edge   = (seg_is_axon.get(seg_key_nid,    False) or
                          seg_is_axon.get(seg_key_parent, False))
        draw_color   = 'cyan' if is_axon_edge else GREY
        lw           = LW_SKEL_AXON if is_axon_edge else LW_SKEL_GREY
        ax.plot([x0, x1], [y0, y1], color=draw_color, lw=lw,
                solid_capstyle='round', zorder=2)

    for nid in node_order:
        node   = nodes[nid]
        parent = node['parent_id']
        if parent == -1:
            continue
        color = edge_colours.get((parent, nid), GREY)
        if color == GREY:
            continue
        x0, y0 = nodes[parent]['x'], nodes[parent]['y']
        x1, y1 = node['x'],          node['y']
        lw = LW_SKEL_SOMA if (nid in soma_node_ids or parent in soma_node_ids) else LW_SKEL_SEL
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw,
                solid_capstyle='round', zorder=5)

    stim_node      = None
    stim_node_dist = float('inf')
    for nid, seg_key in node_to_segment.items():
        if seg_key == stim_key:
            candidate_xyz = np.array([nodes[nid]['x'], nodes[nid]['y'], nodes[nid]['z']])
            d = np.linalg.norm(candidate_xyz - distal_xyz)
            if d < stim_node_dist:
                stim_node_dist = d
                stim_node      = nid

    if stim_node is not None:
        ax.scatter(nodes[stim_node]['x'], nodes[stim_node]['y'],
                   s=MS_STIM, marker='*', color='black', zorder=6,
                   label='Stim site', edgecolors='white', linewidths=1.5)

    legend_handles = []
    seen_colors    = set()
    for key, label, color in zip(keys1, labels1, trace_colors):
        if color in seen_colors:
            continue
        seen_colors.add(color)
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=LW_SKEL_SEL, label=label))
    legend_handles.append(
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
                   markersize=18, label='Stim site'))
    legend_handles.append(
        plt.Line2D([0], [0], color='cyan', lw=LW_SKEL_AXON, label='Axon (detected)'))

    ax.set_aspect('equal')
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold')
    ax.set_xlabel("x (µm)", fontsize=FONT_LABEL)
    ax.set_ylabel("y (µm)", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    # Put the morphology/tree legend in the bottom-left corner
    # and make the legend box more transparent so data behind it stays visible.
    ax.legend(
        handles=legend_handles,
        fontsize=FONT_LEGEND,
        loc='lower right',
        framealpha=0.55,
        facecolor='white',
        edgecolor='black'
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_current_waveform(ax, t, stim_delay_ms, stim_dur_ms, amp_nA, title):
    t_s             = t / 1000.0
    i_wave          = np.zeros_like(t)
    on_mask         = (t >= stim_delay_ms) & (t <= stim_delay_ms + stim_dur_ms)
    i_wave[on_mask] = amp_nA

    ax.fill_between(t_s, i_wave, step='post',
                    color='steelblue', alpha=0.35, label='Current (nA)')
    ax.plot(t_s, i_wave, color='steelblue', lw=1.5, drawstyle='steps-post')

    ax.axvline(stim_delay_ms / 1000.0, color='dimgrey',
               lw=1.0, linestyle='--', alpha=0.7, label='Stim onset')
    ax.axvline((stim_delay_ms + stim_dur_ms) / 1000.0, color='salmon',
               lw=1.0, linestyle='--', alpha=0.7, label='Stim offset')

    ax.set_ylim(-amp_nA * 0.15, amp_nA * 1.30)
    # FIX #11: removed the internal ax.set_xlim(0, t_s[-1]) that was immediately
    # overridden by the caller; the caller's xlim is the authoritative one.

    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=FONT_LABEL)
    ax.set_ylabel("Current (nA)", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    # More transparent legend so the current waveform remains visible behind it.
    ax.legend(
        fontsize=FONT_LEGEND,
        loc='upper right',
        framealpha=0.55,
        facecolor='white',
        edgecolor='black'
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.18, lw=0.5)


fig = plt.figure(figsize=FIG_COMPOSITE)
# Make the tree, current plots, and voltage plots sit closer together.
# - Slightly widen the tree panel
# - Reduce horizontal and vertical spacing between panels
gs  = fig.add_gridspec(
    nrows=2, ncols=3,
    width_ratios=[1.55, 1, 1],
    height_ratios=[1, 2.2],
    hspace=0.22,
    wspace=0.16
)

ax_morph = fig.add_subplot(gs[:, 0])
ax_curr1 = fig.add_subplot(gs[0, 1])
ax_curr2 = fig.add_subplot(gs[0, 2])
ax_volt1 = fig.add_subplot(gs[1, 1])
ax_volt2 = fig.add_subplot(gs[1, 2])

draw_skeleton_selected(
    ax_morph, nodes, node_order, node_to_segment,
    selected_colour_map = colour_map_combined,
    soma_node_ids       = soma_node_ids,
    stim_key            = stim_key,
    distal_xyz          = distal_xyz,
    title               = "Morphology skeleton\n(highlighted: 5 selected segments)"
)

plot_current_waveform(ax_curr1, t1, stim_delay_ms=100, stim_dur_ms=1.0,
                      amp_nA=I_AMP, title="Current injection — Burst (1 ms)")
ax_curr1.set_xlim(0.08, 0.15)

plot_current_waveform(ax_curr2, t2, stim_delay_ms=100, stim_dur_ms=1000.0,
                      amp_nA=I_AMP, title="Current injection — Sustained (1 s)")
ax_curr2.set_xlim(0, 1.5)

plot_five_traces(ax_volt1, t1, v1, keys=keys1, labels=labels1, colors=trace_colors,
                 title="Voltage responses — Burst (1 ms)",
                 stim_delay_ms=100, stim_dur_ms=1.0)
ax_volt1.set_xlim(0.08, 0.15)

plot_five_traces(ax_volt2, t2, v2, keys=keys2, labels=labels2, colors=trace_colors,
                 title="Voltage responses — Sustained (1 s)",
                 stim_delay_ms=100, stim_dur_ms=1000.0, xlim_s=1.5)

fig.suptitle(
    "Neuron morphology with selected recording sites, current injection, and voltage responses",
    fontsize=FONT_SUPTITLE, fontweight='bold', y=1.01
)

# Tighten the outer figure margins so the full composite looks less spread out.
fig.subplots_adjust(top=0.92, left=0.06, right=0.98, bottom=0.07)

plt.savefig("neuron_morphology_selected_traces.png", dpi=150, bbox_inches='tight')
print("\nSaved: neuron_morphology_selected_traces.png")
plt.show()


# ===================================================================================================
# STEP 9 — TTM/FWHM VISUALIZATION, SHAPE FILTERING, CLUSTERING, FILTERED PLOTS, AND JSON EXPORT
# ===================================================================================================

# FIX #2/#3/#4/#5/#6: All duplicate function definitions that were repeated in Steps 9-13
# have been removed. The single canonical definitions above are used throughout.

# ---------------------------------------------------------------------------------------------------
# Helper: build trace feature matrix for clustering
# ---------------------------------------------------------------------------------------------------
def build_trace_feature_matrix(v_dict, all_keys, t, t_start, t_end):
    mask       = (t >= t_start) & (t <= t_end)
    X          = []
    valid_keys = []
    for key in all_keys:
        if key not in v_dict:
            continue
        trace    = v_dict[key]
        baseline = np.mean(trace[t < t_start])
        X.append(trace[mask] - baseline)
        valid_keys.append(key)
    return valid_keys, np.array(X)


# ---------------------------------------------------------------------------------------------------
# Helper: cluster filtered segments
# ---------------------------------------------------------------------------------------------------
def cluster_segments_by_trace_similarity(v_dict, all_keys, t, t_start, t_end, n_clusters=5):
    valid_keys, X = build_trace_feature_matrix(v_dict, all_keys, t, t_start, t_end)

    if len(valid_keys) == 0 or X.size == 0:
        print("WARNING: No valid segments available for clustering in this run.")
        return {}, 0

    n_clusters = min(n_clusters, len(valid_keys))
    if len(valid_keys) == 1:
        return {valid_keys[0]: 0}, 1

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    km     = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = km.fit_predict(Xs)
    return {k: int(c) for k, c in zip(valid_keys, labels)}, n_clusters


# ---------------------------------------------------------------------------------------------------
# Helper: reorder clusters so cluster 0 is most stim-like
# ---------------------------------------------------------------------------------------------------
def reorder_clusters_by_similarity_to_stim(cluster_ids, att_values, k):
    if k == 0 or len(cluster_ids) == 0:
        return {}
    cluster_means = []
    for c in range(k):
        vals     = [att_values[key] for key, cc in cluster_ids.items()
                    if cc == c and key in att_values]
        mean_val = np.mean(vals) if vals else -np.inf
        cluster_means.append((c, mean_val))
    cluster_means.sort(key=lambda x: x[1], reverse=True)
    old_to_new = {old: new for new, (old, _) in enumerate(cluster_means)}
    return {key: old_to_new[c] for key, c in cluster_ids.items()}


# ---------------------------------------------------------------------------------------------------
# Helper: red gradient for clusters
# ---------------------------------------------------------------------------------------------------
def build_cluster_red_gradient(k):
    if k <= 0:
        return {}
    cmap = plt.cm.Reds
    vals = np.linspace(0.35, 0.95, k)[::-1]
    return {i: cmap(vals[i]) for i in range(k)}


# ---------------------------------------------------------------------------------------------------
# Helper: filtered voltage plot
# ---------------------------------------------------------------------------------------------------
def plot_voltage(ax, t, v, keys_to_plot, cluster_ids, cluster_colors,
                 stim_key, segment_labels, title):
    valid_keys = [k for k in keys_to_plot if k in v and k in cluster_ids]

    if len(valid_keys) == 0:
        ax.text(0.5, 0.5, "No clustered segments to plot",
                ha='center', va='center', transform=ax.transAxes, fontsize=FONT_TICK)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (mV)")
        ax.grid(True, alpha=0.25)
        return

    keys_by_cluster = sorted(valid_keys, key=lambda k: cluster_ids[k])
    cluster_handle  = {}

    for key in keys_by_cluster:
        c     = cluster_ids[key]
        color = cluster_colors.get(c, 'lightcoral')
        ax.plot(t / 1000.0, v[key], color=color, lw=0.6, alpha=0.7)
        if c not in cluster_handle:
            cluster_handle[c] = plt.Line2D([], [], color=color, lw=2, label=f"Cluster {c}")

    if stim_key in v and stim_key in cluster_ids:
        ax.plot(t / 1000.0, v[stim_key], color='black', lw=2,
                label=f"Stimulated: {segment_labels.get(stim_key, stim_key)}")
        cluster_handle['stim'] = plt.Line2D([], [], color='black', lw=2, label="Stimulated")

    ax.axhline(-65, linestyle='--', color='k', alpha=0.3)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=FONT_LABEL)
    ax.set_ylabel("Voltage (mV)", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(True)

    handles = [cluster_handle[c] for c in sorted(c for c in cluster_handle if c != 'stim')]
    if 'stim' in cluster_handle:
        handles.append(cluster_handle['stim'])
    ax.legend(handles=handles, loc='upper right', fontsize=FONT_LEGEND)


# ---------------------------------------------------------------------------------------------------
# Helper: cluster similarity / distance panel
# ---------------------------------------------------------------------------------------------------
def plot_cluster_similarity_distance_panel(
    att_values, cluster_ids, cluster_colors, all_keys,
    soma_index, non_soma_index, dist_map, k, ax, title
):
    if k <= 0:
        ax.text(0.5, 0.5, "No clustered segments available",
                ha='center', va='center', transform=ax.transAxes, fontsize=FONT_TICK)
        ax.set_title(title)
        ax.set_xlabel("Cluster number")
        ax.set_ylabel("Peak ratio vs stimulated segment")
        ax.grid(True, alpha=0.25)
        return

    cluster_to_keys = {c: [] for c in range(k)}
    for key in all_keys:
        c = cluster_ids.get(key, None)
        if c is not None:
            cluster_to_keys[c].append(key)

    cluster_centers = np.arange(k)
    local_halfwidth = 0.28
    axis_y  = -0.06
    tick_y0 = axis_y - 0.012
    tick_y1 = axis_y + 0.012

    for c in range(k):
        keys_c = cluster_to_keys.get(c, [])
        if not keys_c:
            continue

        dvals = np.array([dist_map.get(key, np.nan) for key in keys_c], dtype=float)
        valid = np.isfinite(dvals)
        keys_c = [key for key, ok in zip(keys_c, valid) if ok]
        dvals  = dvals[valid]

        if len(keys_c) == 0:
            continue

        dmax = float(np.max(dvals))
        if dmax <= 0:
            dmax = 1.0

        xc      = cluster_centers[c]
        x_left  = xc - local_halfwidth
        x_right = xc + local_halfwidth

        ax.plot([x_left, x_right], [axis_y, axis_y],
                color='black', lw=0.8, zorder=1, clip_on=False)

        # Draw ONLY 3 labels: 0, mid, max (clean and readable)
        for frac, d in zip([0.0, 0.5, 1.0], [0.0, 0.5 * dmax, dmax]):
            xt = x_left + frac * (x_right - x_left)

            # Tick mark
            ax.plot([xt, xt], [tick_y0, tick_y1],
                    color='black', lw=0.9, zorder=1, clip_on=False)

            # Label (rounded, fewer digits, slightly smaller font)
            ax.text(
                xt,
                axis_y - 0.03,                 # push further DOWN (key fix)
                f"{d:.0f}" if frac < 1.0 else f"{d:.0f} µm",                  # INTEGER → no clutter
                ha='center',
                va='top',
                fontsize=FONT_TICK - 6,       # smaller text → readable
                clip_on=False
            )

        for key, d in zip(keys_c, dvals):
            frac = d / dmax
            x    = x_left + frac * (x_right - x_left)
            y    = att_values.get(key, 0.0)
            marker = 'D' if key in soma_index else 'o'
            size   = 60  if key in soma_index else 42
            txt    = f"S{soma_index[key]}" if key in soma_index \
                     else f"{non_soma_index.get(key, '?')}"

            ax.scatter(x, y, color=cluster_colors.get(c, 'lightcoral'),
                       marker=marker, s=MS_DOT_CLUST, edgecolors='white', linewidths=0.8, zorder=3)
            ax.annotate(txt, (x, y), fontsize=FONT_TICK - 2, ha='center', va='bottom',
                        xytext=(0, 4), textcoords='offset points')

    ax.set_xlim(-0.5, k - 0.5)
    ax.set_ylim(-0.14, 1.05)
    ax.set_xticks(cluster_centers)
    ax.set_xticklabels([f"{c}" for c in range(k)], fontsize=FONT_TICK)
    ax.set_xlabel(
    "Branch order (within each order, horizontal position = distance from stim site [µm])",
    fontsize=FONT_LABEL
    )
    ax.set_ylabel("Peak ratio vs stimulated segment", fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(alpha=0.25, axis='y')

    cluster_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cluster_colors.get(c, 'lightcoral'),
                   markersize=12, label=f"Cluster {c}")
        for c in range(k)
    ]
    soma_handle = plt.Line2D([0], [0], marker='D', color='w',
                             markerfacecolor='grey', markersize=12, label='Soma segment')
    ax.legend(handles=cluster_handles + [soma_handle], loc='upper right',
              fontsize=FONT_LEGEND, framealpha=0.9)
    ax.text(-0.48, axis_y - 0.005, "Distance (µm)",
            ha='left', va='center', fontsize=FONT_LEGEND, fontweight='bold', clip_on=False)


# ---------------------------------------------------------------------------------------------------
# Helper: print cluster table (single canonical definition — FIX #4)
# ---------------------------------------------------------------------------------------------------
def print_cluster_table(att_values, cluster_ids, all_keys,
                        soma_index, non_soma_index, segment_labels,
                        dist_map, title="Cluster Table"):
    print("\n" + "=" * 110)
    print(f"  {title}")
    print("=" * 110)
    print(f"  {'Rank':>4}  {'Segment label':<35}  {'Class':<9}  "
          f"{'Peak ratio':>10}  {'Cluster':>10}  {'Dist (um)':>10}")
    print("-" * 110)

    sorted_keys = sorted(all_keys, key=lambda k: att_values.get(k, 0.0), reverse=True)

    for rank, key in enumerate(sorted_keys, 1):
        att = att_values.get(key, 0.0)
        c   = cluster_ids.get(key, -1)
        d   = dist_map.get(key, np.nan)
        lbl = segment_labels.get(key, key)
        cls = ("SOMA"     if key in soma_index     else
               "NON-SOMA" if key in non_soma_index else "UNKNOWN")
        cluster_txt = f"Cluster {c}" if c >= 0 else "Filtered"
        print(f"  {rank:>4}  {lbl:<35}  {cls:<9}  {att:>10.4f}  {cluster_txt:<10}  {d:>10.2f}")

    print("=" * 110)


# ---------------------------------------------------------------------------------------------------
# PART B — Run clustering
# NOTE: The TTM/FWHM filter scatter is deferred to the very end of the script (Step 15)
# so it appears AFTER all analysis plots rather than before them.
# FIX #1: clust1/clust2/k1/k2 are now ASSIGNED HERE before any code tries to use them.
# The original code called reorder_clusters_by_similarity_to_stim(clust1, ...) before
# cluster_segments_by_trace_similarity() had ever been invoked, causing a NameError.
# ---------------------------------------------------------------------------------------------------
clust1_raw, k1 = cluster_segments_by_trace_similarity(
    v1, filtered_keys1, t1, t_start=100.0, t_end=250.0, n_clusters=5)
clust2_raw, k2 = cluster_segments_by_trace_similarity(
    v2, filtered_keys2, t2, t_start=100.0, t_end=1100.0, n_clusters=5)

clust1 = reorder_clusters_by_similarity_to_stim(clust1_raw, corr1, k1)
clust2 = reorder_clusters_by_similarity_to_stim(clust2_raw, corr2, k2)

cluster_colors1 = build_cluster_red_gradient(k1)
cluster_colors2 = build_cluster_red_gradient(k2)

print(f"1 ms clustered segments: {len(clust1)}, k1={k1}, colors={len(cluster_colors1)}")
print(f"1 s  clustered segments: {len(clust2)}, k2={k2}, colors={len(cluster_colors2)}")

# ---------------------------------------------------------------------------------------------------
# PART D — print cluster tables
# ---------------------------------------------------------------------------------------------------
print_cluster_table(corr1, clust1, all_segment_keys, soma_index, non_soma_index,
                    segment_labels, segment_distance_from_stim,
                    title="Peak-ratio / cluster table — 1 ms stimulus")
print_cluster_table(corr2, clust2, all_segment_keys, soma_index, non_soma_index,
                    segment_labels, segment_distance_from_stim,
                    title="Peak-ratio / cluster table — 1 s stimulus")

# ---------------------------------------------------------------------------------------------------
# PART E — filtered voltage traces
# ---------------------------------------------------------------------------------------------------
fig_v, axes_v = plt.subplots(2, 1, figsize=FIG_VOLTAGE)

plot_voltage(axes_v[0], t1, v1, filtered_keys1, clust1, cluster_colors1,
             stim_key, segment_labels,
             f"Voltage traces — filtered/clustered — 1 ms stimulus (k={k1})")
plot_voltage(axes_v[1], t2, v2, filtered_keys2, clust2, cluster_colors2,
             stim_key, segment_labels,
             f"Voltage traces — filtered/clustered — 1 s stimulus (k={k2})")

plt.tight_layout()
plt.savefig("neuron_voltage_clustered.png", dpi=150)
print("Saved: neuron_voltage_clustered.png")
plt.show()

# ---------------------------------------------------------------------------------------------------
# PART F — cluster similarity / distance panel removed:
#           the branch-order plot (Step 14) is the primary spatial view.
# ---------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------
# build_seg_key_to_branch_level — defined here so it is available for the JSON
# export in Part G below AND for the branch-order plot in Step 14.
# ---------------------------------------------------------------------------------------------------
def build_seg_key_to_branch_level(nodes, node_order, children, node_to_segment):
    """
    BFS over the SWC node tree.  Assigns an integer branch order to every node:
      - The root (parent == -1) is order 1.
      - Traversing a bifurcation node (2+ children) increments the order for all
        children of that fork.  Non-branching pass-through nodes keep the same order.

    Then each NEURON segment key is assigned the branch order of the SWC node that
    node_to_segment maps to it (nearest-neighbour mapping already computed earlier).

    Returns
    -------
    seg_key_to_level : dict  segment_key -> int branch order (1-based)
    node_level       : dict  node_id     -> int branch order (for diagnostics)
    """
    from collections import deque

    root = next(nid for nid in node_order if nodes[nid]['parent_id'] == -1)

    node_level = {}
    queue      = deque()
    queue.append((root, 1))

    while queue:
        nid, level = queue.popleft()
        node_level[nid] = level

        kids = children[nid]
        if len(kids) >= 2:
            next_level = level + 1
        else:
            next_level = level

        for child in kids:
            if child not in node_level:
                queue.append((child, next_level))

    seg_key_to_level = {}
    for nid, seg_key in node_to_segment.items():
        if seg_key is None:
            continue
        lvl = node_level.get(nid, 1)
        if seg_key not in seg_key_to_level or lvl < seg_key_to_level[seg_key]:
            seg_key_to_level[seg_key] = lvl

    return seg_key_to_level, node_level


# ---------------------------------------------------------------------------------------------------
# Pre-compute branch-order map here so it is available for the JSON export below.
# ---------------------------------------------------------------------------------------------------
seg_key_to_branch_level, node_level_map = build_seg_key_to_branch_level(
    nodes, node_order, children, node_to_segment)

max_level_found = max(seg_key_to_branch_level.values()) if seg_key_to_branch_level else 1
print(f"\nBranch order range: 1 – {max_level_found}")
print(f"Segments with branch-order assignment: {len(seg_key_to_branch_level)}")

# ---------------------------------------------------------------------------------------------------
# PART G — JSON export
# FIX #8 / #9: corrected two broken dict literals where a closing brace was missing,
# causing the next key to be parsed as part of the preceding value.
# ---------------------------------------------------------------------------------------------------
export = {
    "node_to_segment": {str(k): v for k, v in node_to_segment.items()},
    "nodes": [
        {
            "id":           nid,
            "x":            nodes[nid]["x"],
            "y":            nodes[nid]["y"],
            "z":            nodes[nid]["z"],
            "type":         nodes[nid]["type"],
            "parent":       nodes[nid]["parent_id"],
            "radius":       nodes[nid]["radius"],
            # topology flags — needed by the 3D viewer
            "is_axon":      (node_to_segment.get(nid) is not None and
                             seg_is_axon.get(node_to_segment.get(nid), False)),
            "branch_order": node_level_map.get(nid, 1),
        }
        for nid in node_order
    ],
    # Exact SWC node nearest to the stim section's distal 3D tip.
    # Computed here once so the 3D viewer doesn't have to guess.
    "stim_node_id": int(min(
        node_order,
        key=lambda nid: (
            (nodes[nid]["x"] - float(distal_xyz[0]))**2 +
            (nodes[nid]["y"] - float(distal_xyz[1]))**2 +
            (nodes[nid]["z"] - float(distal_xyz[2]))**2
        )
    )),
    "stim_key": stim_key,
    "runs": {
        "1ms": {
            "corr":          {k: float(corr1[k])           for k in all_segment_keys},
            "cluster":       {k: int(clust1.get(k, -1))    for k in all_segment_keys},
            "peak":          {k: float(np.max(v1[k]))       for k in v1},   # FIX #8: was missing closing brace
            "k":             int(k1),
            "filtered_keys": filtered_keys1,
        },
        "1s": {
            "corr":          {k: float(corr2[k])           for k in all_segment_keys},
            "cluster":       {k: int(clust2.get(k, -1))    for k in all_segment_keys},
            "peak":          {k: float(np.max(v2[k]))       for k in v2},   # FIX #9: was missing closing brace
            "k":             int(k2),
            "filtered_keys": filtered_keys2,
        },
    },
    "segment_labels": segment_labels,
    "soma_keys":       soma_segment_keys,
    "non_soma_keys":   non_soma_segment_keys,
    "soma_index":      soma_index,
    "non_soma_index":  {k: int(v) for k, v in non_soma_index.items()},
}

with open("neuron_morphology_data.json", "w") as f:
    json.dump(export, f)
print("Exported: neuron_morphology_data.json")


# ===================================================================================================
# STEP 11 — NODE SUMMARY
# FIX #7: The original rebuilt node_to_segment from scratch with an expensive O(N²) loop,
# discarding the correct result already computed by build_node_to_segment_map() earlier.
# We now reuse the existing node_to_segment dict and only add the peak-voltage annotation.
# ===================================================================================================
print("\nNode summary:\n")

# Pre-compute peak voltage for every segment in the sustained run
seg_peak = {k: np.max(v2[k]) for k in v2}

for nid in node_order:
    node     = nodes[nid]
    best_key = node_to_segment[nid]   # FIX #7: use pre-computed mapping, not a new O(N²) search
    peak     = seg_peak.get(best_key, None)

    if best_key in soma_index:
        seg_class = "SOMA"
        seg_lbl   = f"Soma segment {soma_index[best_key]}"
    elif best_key in non_soma_index:
        seg_class = "NON-SOMA"
        seg_lbl   = f"Non-soma segment {non_soma_index[best_key]}"
    else:
        seg_class = "UNKNOWN"
        seg_lbl   = "Unknown segment"

    print(f"{nid:>4}  "
          f"{node['x']:>8.1f} {node['y']:>8.1f} {node['z']:>8.1f}  "
          f"{node['radius']:>6.2f}  "
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
def plot_branch_correlation_dotplot(corr_values, cluster_ids, cluster_colors,
                                     seg_key_to_level, all_keys, k,
                                     stim_key, segment_labels, ax, title, v_dict,
                                     legend_loc='upper left'):
    """
    Dot plot:
      y = peak ratio vs stimulated segment
      major x grouping = branch order
      local x offset inside each branch-order group = distance from stim site

    For each branch order:
      - a short horizontal scale line is drawn
      - left side of that line represents 0 um from stim
      - right side represents the maximum distance observed in that branch order
      - each dot is placed along that local scale
    """

    # Keep only keys that actually have voltage data
    keys_present = [k for k in all_keys if k in v_dict]
    if not keys_present:
        ax.text(0.5, 0.5, "No branch-level data", ha='center', va='center',
                transform=ax.transAxes, fontsize=FONT_TICK)
        ax.set_title(title)
        return

    # Determine all branch orders present
    levels_present = sorted({seg_key_to_level.get(k, 1) for k in keys_present})
    max_lvl = max(levels_present)

    # Width of the mini distance ruler inside each branch-order column
    local_halfwidth = 0.28

    # Vertical position for the local distance rulers and labels
    axis_y  = -0.06
    tick_y0 = axis_y - 0.012
    tick_y1 = axis_y + 0.012

    # Draw one local distance ruler per branch-order group
    for lvl in levels_present:
        keys_lvl = [k for k in keys_present if seg_key_to_level.get(k, 1) == lvl]

        # Collect valid distances for this branch order
        dvals = np.array([segment_distance_from_stim.get(k, np.nan) for k in keys_lvl], dtype=float)
        valid = np.isfinite(dvals)
        keys_lvl = [k for k, ok in zip(keys_lvl, valid) if ok]
        dvals = dvals[valid]

        if len(keys_lvl) == 0:
            continue

        # Maximum distance within this branch-order group
        dmax = float(np.max(dvals))
        if dmax <= 0:
            dmax = 1.0

        # Left/right ends of the local ruler for this branch-order column
        x_center = lvl
        x_left   = x_center - local_halfwidth
        x_right  = x_center + local_halfwidth

        # Draw the horizontal ruler line
        ax.plot([x_left, x_right], [axis_y, axis_y],
                color='black', lw=0.9, zorder=1, clip_on=False)

        # Draw only 3 labels per local ruler so they do not overlap:
        # left = 0, middle = half-max, right = max.
        # Show the unit only once, on the rightmost label.
        for frac, d in zip([0.0, 0.5, 1.0], [0.0, 0.5 * dmax, dmax]):
            xt = x_left + frac * (x_right - x_left)

            # Short ruler tick
            ax.plot([xt, xt], [tick_y0, tick_y1],
                    color='black', lw=0.9, zorder=1, clip_on=False)

            # Compact integer labels; only the final label shows the unit.
            label_txt = f"{d:.0f}" if frac < 1.0 else f"{d:.0f} µm"
            ax.text(
                xt, axis_y - 0.032, label_txt,
                ha='center', va='top',
                fontsize=FONT_TICK - 2,
                clip_on=False
            )

        # Plot each segment in this branch-order group using its distance-based local x-position
        for key, d in zip(keys_lvl, dvals):
            r       = corr_values.get(key, 0.0)
            c       = cluster_ids.get(key, None)
            is_axon = seg_is_axon.get(key, False)

            # Map distance to a local position on the ruler
            frac = d / dmax
            x    = x_left + frac * (x_right - x_left)

            # Choose marker style by segment class
            if is_axon:
                marker = 's'
                color  = 'cyan'
                size   = MS_DOT_BRANCH
            elif key in soma_index:
                marker = 'D'
                color  = cluster_colors.get(c, '#888888') if c is not None else '#888888'
                size   = MS_DOT_BRANCH
            else:
                marker = 'o'
                color  = cluster_colors.get(c, '#cccccc') if c is not None else '#cccccc'
                size   = MS_DOT_BRANCH

            ax.scatter(x, r, color=color, marker=marker, s=size,
                       alpha=0.80, edgecolors='white', linewidths=0.5, zorder=3)

    # Draw the stimulated segment as a black star
    if stim_key in v_dict:
        r_stim = corr_values.get(stim_key, 1.0)
        lvl_s  = seg_key_to_branch_level.get(stim_key, 1)

        # Place stim star using the same local distance ruler logic
        keys_lvl = [k for k in keys_present if seg_key_to_branch_level.get(k, 1) == lvl_s]
        dvals = np.array([segment_distance_from_stim.get(k, np.nan) for k in keys_lvl], dtype=float)
        dvals = dvals[np.isfinite(dvals)]
        dmax = float(np.max(dvals)) if len(dvals) else 1.0
        if dmax <= 0:
            dmax = 1.0

        x_left  = lvl_s - local_halfwidth
        x_right = lvl_s + local_halfwidth
        d_stim  = segment_distance_from_stim.get(stim_key, 0.0)
        frac_s  = d_stim / dmax
        x_stim  = x_left + frac_s * (x_right - x_left)

        ax.scatter(x_stim, r_stim, color='black', marker='*', s=MS_STIM,
                   zorder=6, label='Stimulated segment',
                   edgecolors='white', linewidths=1.0)

    # Optional axon annotation
    axon_orders = [seg_key_to_level.get(key, 1)
                   for key in keys_present if seg_is_axon.get(key, False)]
    if axon_orders:
        axon_ratios = [corr_values.get(key, 0.0)
                       for key in keys_present if seg_is_axon.get(key, False)]
        ann_x = float(np.median(axon_orders))
        ann_y = float(np.median(axon_ratios))

        ax.annotate(
            "Axon",
            xy=(ann_x, ann_y),
            xytext=(ann_x + 0.6, ann_y + 0.12),
            ha='left', va='bottom',
            fontsize=FONT_ANNOT, color='darkcyan', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkcyan', lw=1.8),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='darkcyan', alpha=0.9)
        )

    # Baseline
    ax.axhline(0, color='k', lw=1.0, linestyle='--', alpha=0.4)

    # Main axes formatting
    ax.set_xticks(range(1, max_lvl + 1))
    ax.set_xticklabels([f"{i}" for i in range(1, max_lvl + 1)], fontsize=FONT_TICK + 2)
    # Horizontal position inside each branch-order group represents local distance from
    # the stimulated site; the local ruler labels carry the unit in µm.
    ax.set_xlabel(
        "Branch order (within each order, horizontal position shows distance from stim site)",
        fontsize=FONT_LABEL + 2
    )
    ax.set_ylabel("Peak ratio vs stimulated segment", fontsize=FONT_LABEL + 2)
    ax.set_xlim(0.3, max_lvl + 0.7)
    ax.set_ylim(-0.14, 1.05)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(alpha=0.25, axis='y')

    # Legend
    cluster_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cluster_colors.get(c, 'lightcoral'),
                   markersize=13, label=f"Cluster {c}")
        for c in range(k)
    ]
    stim_handle = plt.Line2D([0], [0], marker='*', color='w',
                             markerfacecolor='black', markersize=18, label='Stimulated')
    soma_handle = plt.Line2D([0], [0], marker='D', color='w',
                             markerfacecolor='grey', markersize=13, label='Soma segment')
    axon_handle = plt.Line2D([0], [0], marker='s', color='w',
                             markerfacecolor='cyan', markersize=13, label='Axon (detected)')

   # Legend entry explaining the local ruler beneath each branch-order group.
    distance_handle = plt.Line2D(
        [0, 1], [0, 0],
        color='black', lw=1.2,
        label='Local ruler: 0 → max distance [µm]'
    )
    ax.legend(
        handles=cluster_handles + [stim_handle, soma_handle, axon_handle, distance_handle],
        loc=legend_loc,
        fontsize=FONT_LEGEND,
        framealpha=0.55,
        facecolor='white',
        edgecolor='black',
        ncol=2
    )


fig_b, axes_b = plt.subplots(2, 1, figsize=FIG_BRANCH)

# =====================================================================
# Compute space constant (λ) BEFORE plotting branch-order figures
# =====================================================================

lambda_1ms = compute_space_constant(
    corr1,
    segment_distance_from_stim,
    all_segment_keys
)

lambda_1s = compute_space_constant(
    corr2,
    segment_distance_from_stim,
    all_segment_keys
)

print("\n================ SPACE CONSTANT =================")
print(f"λ (1 ms stimulus) = {lambda_1ms:.2f} µm")
print(f"λ (1 s stimulus)  = {lambda_1s:.2f} µm")
print("================================================\n")


# Plot the 1 ms branch-order panel.
# Title no longer shows k because k refers to KMeans clusters, not branch order.
plot_branch_correlation_dotplot(
    corr1, clust1, cluster_colors1,
    seg_key_to_branch_level, all_segment_keys, k1,
    stim_key, segment_labels, axes_b[0],
    "Peak ratio by Branch Order — 1 ms stimulus",
    v_dict=v1,
    legend_loc='upper left'
)

# Plot the 1 s branch-order panel.
# Keep the legend in the bottom-right as requested.
plot_branch_correlation_dotplot(
    corr2, clust2, cluster_colors2,
    seg_key_to_branch_level, all_segment_keys, k2,
    stim_key, segment_labels, axes_b[1],
    "Peak ratio by Branch Order — 1 s stimulus",
    v_dict=v2,
    legend_loc='center right'
)

plt.tight_layout()
plt.savefig("neuron_branch_order.png", dpi=150)
print("Saved: neuron_branch_order.png")
plt.show()

# ===================================================================================================
# STEP 15 — TTM AND FWHM vs DISTANCE FROM STIM + FILTER DIAGNOSTIC
#
# Each figure has three panels:
#   Row 0 : Time-to-max (ms) vs distance from stim site (µm), coloured by branch order
#   Row 1 : FWHM (ms)        vs distance from stim site (µm), coloured by branch order
#   Row 2 : TTM vs FWHM — filter diagnostic (passed = coloured, failed = grey)
# ===================================================================================================
fig_feat1, axes_feat1 = plt.subplots(3, 1, figsize=FIG_DISTANCE)
plot_ttm_fwhm_vs_distance(
    axes_feat1, ttm1, fwhm1,
    segment_distance_from_stim, seg_key_to_branch_level,
    all_segment_keys, run_label="1 ms stimulus",
    filtered_keys=filtered_keys1,
    ttm_max=15.0, fwhm_max=25.0
)
plt.tight_layout()
plt.savefig("neuron_ttm_fwhm_vs_distance_1ms.png", dpi=150, bbox_inches='tight')
print("Saved: neuron_ttm_fwhm_vs_distance_1ms.png")
plt.show()

fig_feat2, axes_feat2 = plt.subplots(3, 1, figsize=FIG_DISTANCE)
plot_ttm_fwhm_vs_distance(
    axes_feat2, ttm2, fwhm2,
    segment_distance_from_stim, seg_key_to_branch_level,
    all_segment_keys, run_label="1 s stimulus",
    filtered_keys=filtered_keys2,
    ttm_max=1000.0, fwhm_min=5.0
)
plt.tight_layout()
plt.savefig("neuron_ttm_fwhm_vs_distance_1s.png", dpi=150, bbox_inches='tight')
print("Saved: neuron_ttm_fwhm_vs_distance_1s.png")
plt.show()

# input("Press Enter to exit...")