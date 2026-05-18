# ===================================================================================================
# NEURON 3D MORPHOLOGY VIEWER — attenuation gradient + axon identification
# ===================================================================================================
# Reads neuron_morphology_data.json (produced by the main pipeline) and generates
# two self-contained interactive HTML files:
#
#   neuron_3d_1ms.html  — peak-ratio attenuation for the 1 ms burst stimulus
#   neuron_3d_1s.html   — peak-ratio attenuation for the 1 s sustained stimulus
#
# Visual encoding:
#   Edge colour : Reds colorscale, white (low) → dark red (high peak ratio)
#   Edge width  : scales with node radius so thick dendrites look thicker
#   Axon        : drawn in cyan regardless of peak ratio
#   Soma        : gold sphere marker (all soma-adjacent nodes averaged)
#   Stim site   : black star marker at the exact SWC node nearest to the
#                 stimulated segment's distal tip (stim_node_id from JSON)
#   Hover text  : node ID, segment key, peak ratio, branch order, cluster ID
#
# Changes vs the original script:
#   - Stim site identified via stim_node_id (exact) rather than first-match loop
#   - Axon edges drawn in cyan using is_axon flag from JSON (topology-based)
#   - Edge width driven by node radius for anatomical realism
#   - Soma marker covers all soma nodes (parent == -1 or parent is root), not just node 1
#   - Hover shows branch_order and cluster assignment
#   - Black background + subtle grid for better depth perception in 3D
#   - Colorbar anchored properly with a visible gradient from 0 → 1
# ===================================================================================================

import json
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale


# ─── Colour helpers ──────────────────────────────────────────────────────────

def ratio_to_red(val: float) -> str:
    """
    Map a peak-ratio in [0, 1] to a colour on the Reds colorscale.
    val=0 → very pale pink;  val=1 → deep red.
    We clamp the lower end to 0.08 so even zero-ratio edges are faintly visible
    against a dark background (pure white would disappear on white backgrounds too).
    """
    v = max(0.15, min(1.0, float(val)))
    return sample_colorscale("Reds", [v])[0]


EDGE_WIDTH = 8   # flat line width (px) for all tree edges — increase to taste

def lw_from_radius(radius_um: float,
                   r_min: float = 0.05, r_max: float = 3.0,
                   lw_min: float = 6.0, lw_max: float = 14.0) -> float:
    """
    Radius-to-width mapping kept for reference, but the actual draw width
    is overridden by EDGE_WIDTH so the tree is uniformly thick regardless
    of how small the converted radii are.
    """
    r = max(r_min, min(r_max, float(radius_um)))
    return lw_min + (r - r_min) / (r_max - r_min) * (lw_max - lw_min)


# ─── Load data ───────────────────────────────────────────────────────────────

with open("neuron_morphology_data.json", "r") as f:
    data = json.load(f)

node_list        = data["nodes"]               # list of dicts, one per SWC node
node_to_segment  = data["node_to_segment"]     # str(node_id) -> segment key
stim_key         = data["stim_key"]            # e.g. "soma[42](0.917)"

# stim_node_id was added to the JSON export in a later pipeline run.
# If it is missing (JSON produced by an older run), compute it here:
# find the SWC node whose segment key matches stim_key, then among all
# matching nodes pick the one that appears last in node_list (most distal).
if "stim_node_id" in data:
    stim_node_id = int(data["stim_node_id"])
else:
    # Collect every node whose mapped segment key equals stim_key
    candidates = [
        n for n in node_list
        if node_to_segment.get(str(n["id"])) == stim_key
    ]
    if candidates:
        # Among candidates, prefer the one with the highest branch_order
        # (most distal), then fall back to the last one in file order.
        stim_node_id = max(
            candidates,
            key=lambda n: (n.get("branch_order", 1), n["id"])
        )["id"]
    else:
        # No node maps to stim_key — use node 1 as a safe fallback and warn
        stim_node_id = 1
        print(f"WARNING: no node maps to stim_key '{stim_key}'. "
              f"Re-run the pipeline to regenerate the JSON with stim_node_id. "
              f"Falling back to node 1 for the stim site marker.")

# Build fast lookup: node_id (int) → node dict
node_map = {n["id"]: n for n in node_list}

# Identify soma nodes: the root (parent == -1) and its immediate children
root_id    = next(n["id"] for n in node_list if n["parent"] == -1)
soma_ids   = {root_id} | {n["id"] for n in node_list if n["parent"] == root_id}


# ─── Generate one figure per stimulus condition ───────────────────────────────

for run_key, title_suffix, fname in [
    ("1ms", "Burst 1 ms stimulus",    "neuron_3d_1ms.html"),
    ("1s",  "Sustained 1 s stimulus", "neuron_3d_1s.html"),
]:
    corr_values  = data["runs"][run_key]["corr"]    # seg_key → peak ratio [0,1]
    cluster_ids  = data["runs"][run_key]["cluster"] # seg_key → cluster int or -1
    k            = data["runs"][run_key]["k"]

    fig = go.Figure()

    # ── 1. Draw all edges ────────────────────────────────────────────────────
    # Each SWC edge is drawn as two overlapping Scatter3d line traces:
    #   (a) a slightly thicker black outline at low opacity → depth cue
    #   (b) the coloured edge on top
    # Drawing thousands of 2-point traces is the standard Plotly approach for
    # variable-colour 3D lines (Plotly has no native multi-colour 3D line).

    for node in node_list:
        if node["parent"] == -1:
            continue                      # root has no parent edge

        parent = node_map[node["parent"]]

        # Determine colour: axon → cyan, otherwise Reds by peak ratio
        seg_key   = node_to_segment.get(str(node["id"]))
        # is_axon and branch_order were added in a later pipeline run;
        # fall back to False / 1 if the JSON was produced by an older run.
        is_axon   = node.get("is_axon", False)
        b_order   = node.get("branch_order", 1)
        val       = corr_values.get(seg_key, 0.0) if seg_key else 0.0
        cluster   = cluster_ids.get(seg_key, -1)  if seg_key else -1

        edge_color = "#008080" if is_axon else ratio_to_red(val)  # teal axon on white bg

        # Use the flat EDGE_WIDTH constant — radius-based scaling is too small
        # after nm→µm conversion and produces hairline edges.
        lw_col = EDGE_WIDTH
        lw_out = EDGE_WIDTH + 3   # outline slightly thicker for edge separation

        hover = (
            f"Node {node['id']} → parent {parent['id']}<br>"
            f"Segment: {seg_key or '—'}<br>"
            f"Peak ratio: {val:.3f}<br>"
            f"Branch order: {b_order}<br>"
            f"Cluster: {cluster if cluster >= 0 else 'filtered'}<br>"
            f"{'AXON' if is_axon else ''}"
            f"<extra></extra>"
        )

        xs = [parent["x"], node["x"]]
        ys = [parent["y"], node["y"]]
        zs = [parent["z"], node["z"]]

        # (a) light grey outline — provides edge separation on white background
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(width=lw_out, color="rgba(180,180,180,0.35)"),
            hoverinfo="skip",
            showlegend=False
        ))

        # (b) coloured edge
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(width=lw_col, color=edge_color),
            hovertemplate=hover,
            showlegend=False
        ))

    # ── 2. Soma marker ───────────────────────────────────────────────────────
    soma_xs = [node_map[nid]["x"] for nid in soma_ids if nid in node_map]
    soma_ys = [node_map[nid]["y"] for nid in soma_ids if nid in node_map]
    soma_zs = [node_map[nid]["z"] for nid in soma_ids if nid in node_map]
    cx, cy, cz = np.mean(soma_xs), np.mean(soma_ys), np.mean(soma_zs)

    fig.add_trace(go.Scatter3d(
        x=[cx], y=[cy], z=[cz],
        mode="markers",
        marker=dict(size=11, color="gold",
                    line=dict(color="black", width=2)),
        name="Soma",
        hovertemplate="Soma<br>"
                      f"x={cx:.1f}, y={cy:.1f}, z={cz:.1f}<extra></extra>"
    ))

    # ── 3. Stim site marker ──────────────────────────────────────────────────
    # Use the exact stim_node_id written into the JSON by the pipeline
    # (nearest SWC node to the stim section's distal 3D tip).
    stim_node = node_map.get(stim_node_id)
    if stim_node:
        stim_order   = stim_node.get("branch_order", "?")
        stim_val     = corr_values.get(stim_key, 1.0)
        stim_cluster = cluster_ids.get(stim_key, -1)
        fig.add_trace(go.Scatter3d(
            x=[stim_node["x"]],
            y=[stim_node["y"]],
            z=[stim_node["z"]],
            mode="markers",
            marker=dict(size=14, color="crimson", symbol="diamond",
                        line=dict(color="black", width=2)),
            name="Stim site",
            hovertemplate=(
                f"Stim site (node {stim_node_id})<br>"
                f"Segment: {stim_key}<br>"
                f"Peak ratio: {stim_val:.3f}<br>"
                f"Branch order: {stim_order}<br>"
                f"Cluster: {stim_cluster if stim_cluster >= 0 else 'filtered'}"
                f"<extra></extra>"
            )
        ))

    # ── 4. Invisible dummy trace → drives the colorbar ───────────────────────
    # Plotly requires a marker/line trace with colorscale to show a colorbar.
    # We place two invisible points at [0, 1] which anchors the full scale.
    fig.add_trace(go.Scatter3d(
        x=[None, None], y=[None, None], z=[None, None],
        mode="markers",
        marker=dict(
            size=0.01,
            color=[0.0, 1.0],
            colorscale="Reds",
            cmin=0.0, cmax=1.0,
            showscale=True,
            colorbar=dict(
                title=dict(text="Peak ratio vs stim", side="right", font=dict(size=12)),
                tickmode="array",
                tickvals=[0.0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["0", "0.25", "0.5", "0.75", "1"],
                len=0.6,
                thickness=16,
                x=1.0,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="grey",
                borderwidth=1,
            )
        ),
        hoverinfo="none",
        showlegend=False
    ))

    

    # ── 6. Layout ─────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"Neuron 3D attenuation — {title_suffix}",
            font=dict(size=16, color="black"),
            x=0.5, xanchor="center"
        ),
        scene=dict(
            xaxis=dict(title="X (µm)", showgrid=True, gridcolor="#dddddd",
                       backgroundcolor="white", color="black",
                       zerolinecolor="#cccccc"),
            yaxis=dict(title="Y (µm)", showgrid=True, gridcolor="#dddddd",
                       backgroundcolor="white", color="black",
                       zerolinecolor="#cccccc"),
            zaxis=dict(title="Z (µm)", showgrid=True, gridcolor="#dddddd",
                       backgroundcolor="white", color="black",
                       zerolinecolor="#cccccc"),
            bgcolor="white",
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.4, y=1.4, z=0.7),
                up=dict(x=0, y=0, z=1)
            )
        ),
        paper_bgcolor="white",
        font=dict(color="black"),
        margin=dict(l=0, r=120, b=0, t=50),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#aaaaaa",
            borderwidth=1,
            font=dict(size=12, color="black")
        )
    )

    fig.write_html(fname)
    print(f"Saved: {fname}")