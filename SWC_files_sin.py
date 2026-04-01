# ===================================================================================================
# NEURON + SWC PIPELINE (SEGMENT-LEVEL ANALYSIS) — WITH SPEARMAN CORRELATION CLUSTERING
# ---------------------------------------------------------------------------------------------------
# Changes vs original:
#   - compute_correlation_clusters(): Spearman r of every segment vs stimulated segment,
#     auto-selects k via silhouette score (k=2..8), assigns each segment to exactly one cluster
#   - print_correlation_table(): sorted console table per stimulus run
#   - plot_correlation_panel(): dedicated figure with sorted bar charts + cluster legend
#   - Voltage plots recolored by cluster membership (same color = same cluster)
# ===================================================================================================

# =========================
# Imports
# =========================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import vtk
import os
import sys

from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict

from neuron import h, gui
from neuron.units import ms, mV, um

print(f'VTK version: {vtk.VTK_VERSION}')

# =========================
# Load NEURON tools
# =========================
h.load_file("stdrun.hoc")
h.load_file("import3d.hoc")

# =========================
# Load SWC file
# =========================
swc_path = '/home/aksay_lab/NeuronProject/CloudvolNeuron_Test/SWC_files/76182_reRoot_reSample_5000.swc'

print("File exists?", os.path.exists(swc_path))
print("Loading SWC file:", swc_path)

reader = h.Import3d_SWC_read()
reader.input(swc_path)

importer = h.Import3d_GUI(reader, 1)

# =========================
# Cell container
# =========================
class Cell:
    def __init__(self):
        self.sl = h.SectionList()

cell = Cell()
importer.instantiate(cell)

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
            'node_id': node_id, 'type': node_type,
            'x': x, 'y': y, 'z': z,
            'radius': radius, 'parent_id': parent_id
        }
        node_order.append(node_id)

print(f"Expected SWC edges: {len(nodes) - 1}")

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

sections = []
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
SOMA_THRESHOLD = 10.0

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
candidate_leaves  = [sec for sec in leaf_sections if sec in non_soma_sections] or leaf_sections

leaf_distances = sorted(
    [(path_distance_from_soma(sec, soma_secs), sec) for sec in candidate_leaves],
    key=lambda x: x[0]
)
max_dist, stim_section = leaf_distances[-1]

n3d = stim_section.n3d()
distal_idx  = n3d - 1
distal_xyz  = np.array([stim_section.x3d(distal_idx),
                        stim_section.y3d(distal_idx),
                        stim_section.z3d(distal_idx)])
total_arc   = stim_section.arc3d(n3d - 1)
arc_fracs   = np.array([stim_section.arc3d(i) / total_arc for i in range(n3d)])

best_seg = None
best_dist_seg = float('inf')
for seg in stim_section:
    idx = int(np.argmin(np.abs(arc_fracs - seg.x)))
    seg_xyz = np.array([stim_section.x3d(idx),
                        stim_section.y3d(idx),
                        stim_section.z3d(idx)])
    d = np.linalg.norm(seg_xyz - distal_xyz)
    if d < best_dist_seg:
        best_dist_seg = d
        best_seg = seg

stim_seg  = best_seg
stim_x    = float(stim_seg.x)
stim_key  = f"{stim_section.name()}({stim_x:.3f})"

print(f"\nStim site: {stim_section.name()}({stim_x:.3f})")
print(f"Path distance from soma ≈ {max_dist:.2f} µm")

# ===================================================================================================
# STEP 6 — Simulation function
# ===================================================================================================
T_STOP = 5000.0
DT     = 0.025
I_AMP  = 75

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
# STEP 8 — CORRELATION + CLUSTERING
# ===================================================================================================

def compute_correlation_clusters(v_dict, reference_key, all_keys): 
# v_dict: is the dictionary of voltage traces for all segments; reference_key: is the key of the stimulated segment;
# reference_key: is the key of the stimulated segment; we will compute Spearman r of every other segment vs this reference segment, to see how similar their voltage traces are to the stimulated segment.
    """
    Computes Spearman r for every segment in all_keys vs reference_key.
    Auto-selects k (2..8) by silhouette score on the 1-D correlation values.
    Returns:
        corr_values : dict  key -> float  (Spearman r)
        cluster_ids : dict  key -> int    (0-based cluster label)
        k_chosen    : int
        cluster_centers : sorted list of (center_value, cluster_id)
    """
    ref = v_dict[reference_key]

    corr_values = {}
    for key in all_keys:
        if key not in v_dict:
            corr_values[key] = 0.0
            continue
        r, _ = spearmanr(ref, v_dict[key])
        corr_values[key] = float(r) if not np.isnan(r) else 0.0

    X = np.array([corr_values[k] for k in all_keys]).reshape(-1, 1)

    # Auto-select k via silhouette score
    # what is silhouette score? it measures how well-separated the clusters are; ranges from -1 to 1, higher is better
    # why is it called silhouette score? because it looks at how similar each point is to its own cluster (cohesion) vs other clusters (separation), like a silhouette in the distance

    best_k     = 2 # why is the best k, 2? because silhouette score is only defined for k >= 2, and we want to allow up to 8 clusters but not more than the number of segments
    best_score = -1.0 # why is the best score, -1? because silhouette score ranges from -1 to 1, and higher is better, so we start with the lowest possible score

    for k in range(2, min(9, len(all_keys))):
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = km.fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k     = k

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
    raw_labels = km_final.fit_predict(X)

    # Re-order cluster IDs so cluster 0 = lowest mean corr, cluster k-1 = highest
    centers = km_final.cluster_centers_.flatten()
    order   = np.argsort(centers)          # sorted by center value
    remap   = {old: new for new, old in enumerate(order)}
    cluster_ids = {k: remap[raw_labels[i]] for i, k in enumerate(all_keys)}

    sorted_centers = [(centers[order[c]], c) for c in range(best_k)]

    return corr_values, cluster_ids, best_k, sorted_centers


# Compute for both runs
corr1, clust1, k1, centers1 = compute_correlation_clusters(v1, stim_key, all_segment_keys)
corr2, clust2, k2, centers2 = compute_correlation_clusters(v2, stim_key, all_segment_keys)

print(f"\nAuto-selected k (1 ms run)  : {k1}")
print(f"Auto-selected k (1 s  run)  : {k2}")


# ===================================================================================================
# STEP 9 — CLUSTER COLOR MAPS
# ===================================================================================================
# One distinct color per cluster; use qualitative colormap
def build_cluster_colors(k):
    base = mcm.get_cmap('tab10', k)
    return {c: base(c) for c in range(k)}

cluster_colors1 = build_cluster_colors(k1)
cluster_colors2 = build_cluster_colors(k2)


# ===================================================================================================
# STEP 10 — PRINT SORTED CORRELATION TABLE
# ===================================================================================================

def print_correlation_table(corr_values, cluster_ids, all_keys,
                             soma_index, non_soma_index, segment_labels,
                             title="Correlation Table"):
    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90)
    print(f"  {'Rank':>4}  {'Segment label':<35}  {'Class':<9}  {'Spearman r':>10}  {'Cluster':>7}")
    print("-" * 90)

    # Sort descending by correlation
    sorted_keys = sorted(all_keys, key=lambda k: corr_values.get(k, 0.0), reverse=True)

    for rank, key in enumerate(sorted_keys, 1):
        r   = corr_values.get(key, 0.0)
        c   = cluster_ids.get(key, -1)
        lbl = segment_labels.get(key, key)

        if key in soma_index:
            cls = "SOMA"
        elif key in non_soma_index:
            cls = "NON-SOMA"
        else:
            cls = "UNKNOWN"

        print(f"  {rank:>4}  {lbl:<35}  {cls:<9}  {r:>10.4f}  Cluster {c}")

    print("=" * 90)


print_correlation_table(corr1, clust1, all_segment_keys,
                        soma_index, non_soma_index, segment_labels,
                        title="Spearman Correlation — 1 ms stimulus (vs stimulated segment)")

print_correlation_table(corr2, clust2, all_segment_keys,
                        soma_index, non_soma_index, segment_labels,
                        title="Spearman Correlation — 1 s stimulus (vs stimulated segment)")


# ===================================================================================================
# STEP 11 — NODE SUMMARY
# ===================================================================================================
print("\nNode summary:\n")
node_to_segment = {} # node ID -> best matching segment key (e.g. "SectionName(x.xx)")
seg_peak = {k: np.max(v2[k]) for k in v2}

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
            key = f"{sec.name()}({seg.x:.3f})"
            idx = int(np.argmin(np.abs(af - seg.x)))
            seg_xyz = np.array([sec.x3d(idx), sec.y3d(idx), sec.z3d(idx)])
            dist = np.linalg.norm(node_xyz - seg_xyz)
            if dist < best_dist:
                best_dist = dist
                best_key  = key

    node_to_segment[nid] = best_key
    peak = seg_peak.get(best_key, None)

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
# STEP 12 — VOLTAGE PLOTS (colored by cluster)
# ===================================================================================================

def plot_voltage(ax, t, v, all_keys, cluster_ids, cluster_colors,
                 stim_key, segment_labels, title):
    """Plot voltage traces; each cluster shares one color."""

    # Sort keys by cluster so legend groups nicely
    keys_by_cluster = sorted(all_keys, key=lambda k: cluster_ids.get(k, -1))

    cluster_handle = {}   # one legend handle per cluster
    for key in keys_by_cluster:
        if key not in v:
            continue
        c     = cluster_ids.get(key, 0)
        color = cluster_colors[c]
        ax.plot(t / 1000.0, v[key], color=color, lw=0.6, alpha=0.7)
        if c not in cluster_handle:
            cluster_handle[c] = plt.Line2D([], [], color=color, lw=2,
                                           label=f"Cluster {c}")

    # Stimulated segment on top
    if stim_key in v:
        ax.plot(t / 1000.0, v[stim_key], color='black', lw=2,
                label=f"Stimulated: {segment_labels.get(stim_key, stim_key)}")
        stim_handle = plt.Line2D([], [], color='black', lw=2,
                                 label=f"Stimulated")
        cluster_handle['stim'] = stim_handle

    ax.axhline(-65, linestyle='--', color='k', alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (mV)")
    ax.grid(True)

    handles = [cluster_handle[c] for c in sorted(
        [c for c in cluster_handle if c != 'stim'])] + \
        ([cluster_handle['stim']] if 'stim' in cluster_handle else [])
    ax.legend(handles=handles, loc='upper right', fontsize=8)


fig_v, axes_v = plt.subplots(2, 1, figsize=(13, 9))

plot_voltage(axes_v[0], t1, v1, all_segment_keys, clust1, cluster_colors1,
             stim_key, segment_labels, f"Voltage traces — 1 ms stimulus  (k={k1} clusters)")
plot_voltage(axes_v[1], t2, v2, all_segment_keys, clust2, cluster_colors2,
             stim_key, segment_labels, f"Voltage traces — 1 s stimulus  (k={k2} clusters)")

plt.tight_layout()
plt.savefig("neuron_voltage_clustered.png", dpi=150)
print("\nSaved: neuron_voltage_clustered.png")


# ===================================================================================================
# STEP 13 — DEDICATED CORRELATION PLOT
# ===================================================================================================

def plot_correlation_panel(corr_values, cluster_ids, cluster_colors, all_keys,
                           soma_index, non_soma_index, segment_labels, k, ax, title,
                           v_dict):
    """
    Scatter dot plot:
      X = Spearman r vs stimulated segment
      Y = peak voltage (max mV) of that segment
      Dot color  = cluster membership
      Dot marker = circle (non-soma) / diamond (soma)
      Dot label  = segment number (soma or non-soma index)
    """
    for key in all_keys:
        if key not in v_dict:
            continue

        r     = corr_values.get(key, 0.0)
        peak  = float(np.max(v_dict[key]))
        c     = cluster_ids.get(key, 0)
        color = cluster_colors[c]

        if key in soma_index:
            marker = 'D'
            size   = 60
            num    = soma_index[key]
            label_txt = f"S{num}"
        else:
            marker = 'o'
            size   = 40
            num    = non_soma_index[key]
            label_txt = f"{num}"

        ax.scatter(r, peak, color=color, marker=marker, s=size,
                   zorder=3, edgecolors='white', linewidths=0.4)

        ax.annotate(label_txt, xy=(r, peak),
                    fontsize=4, ha='center', va='bottom',
                    color='black', zorder=4,
                    xytext=(0, 3), textcoords='offset points')

    ax.axvline(0, color='k', lw=0.8, linestyle='--', alpha=0.4)
    ax.set_xlabel("Spearman r vs. stimulated segment")
    ax.set_ylabel("Peak voltage (mV)")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    # ── legend ───────────────────────────────────────────────────────────
    cluster_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cluster_colors[c], markersize=8,
                   label=f"Cluster {c}")
        for c in range(k)
    ]
    soma_handle = plt.Line2D([0], [0], marker='D', color='w',
                             markerfacecolor='grey', markersize=9,
                             label="Soma (S#) / Non-soma (#)")
    ax.legend(handles=cluster_handles + [soma_handle],
              loc='upper left', fontsize=8, framealpha=0.9)


fig_c, axes_c = plt.subplots(2, 1, figsize=(14, 8))

plot_correlation_panel(
    corr1, clust1, cluster_colors1, all_segment_keys,
    soma_index, non_soma_index, segment_labels, k1,
    axes_c[0],
    f"Spearman r vs Peak Voltage — 1 ms stimulus  (auto k={k1})",
    v_dict=v1
)
plot_correlation_panel(
    corr2, clust2, cluster_colors2, all_segment_keys,
    soma_index, non_soma_index, segment_labels, k2,
    axes_c[1],
    f"Spearman r vs Peak Voltage — 1 s stimulus  (auto k={k2})",
    v_dict=v2
)

plt.tight_layout()
plt.savefig("neuron_correlation_clusters.png", dpi=150)
print("Saved: neuron_correlation_clusters.png")

plt.show()

# ── Export SWC_files_sin.py ──────────────
import json

export = {
    "node_to_segment": {str(k): v for k, v in node_to_segment.items()}, 
    "nodes": [
        {
            "id":      nid,
            "x":       nodes[nid]["x"],
            "y":       nodes[nid]["y"],
            "z":       nodes[nid]["z"],
            "type":    nodes[nid]["type"],
            "parent":  nodes[nid]["parent_id"],
            "radius":  nodes[nid]["radius"],
        }
        for nid in node_order
    ],
    "stim_key": stim_key,
    "runs": {
        "1ms": {
            "corr":    {k: float(corr1[k])  for k in all_segment_keys},
            "cluster": {k: int(clust1[k])   for k in all_segment_keys},
            "peak":    {k: float(np.max(v1[k])) for k in v1},
            "k":       int(k1),
        },
        "1s": {
            "corr":    {k: float(corr2[k])  for k in all_segment_keys},
            "cluster": {k: int(clust2[k])   for k in all_segment_keys},
            "peak":    {k: float(np.max(v2[k])) for k in v2},
            "k":       int(k2),
        },
    },
    "segment_labels":    segment_labels,
    "soma_keys":         soma_segment_keys,
    "non_soma_keys":     non_soma_segment_keys,
    "soma_index":        soma_index,
    "non_soma_index":    {k: int(v) for k, v in non_soma_index.items()},
}

with open("neuron_morphology_data.json", "w") as f:
    json.dump(export, f)
print("Exported: neuron_morphology_data.json")

#input("Press Enter to exit...")