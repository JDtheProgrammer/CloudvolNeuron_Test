import json
import plotly.graph_objects as go
import plotly.io as pio

# =========================
# LOAD DATA
# =========================
with open("neuron_morphology_data.json", "r") as f:
    data = json.load(f)

nodes = data["nodes"]
node_to_segment = data["node_to_segment"]
clusters = data["runs"]["1s"]["cluster"]

node_map = {n["id"]: n for n in nodes}

# =========================
# BUILD LINES
# =========================
x_lines = []
y_lines = []
z_lines = []
colors  = []

for n in nodes:
    if n["parent"] == -1:
        continue

    n1 = n
    n2 = node_map[n["parent"]]

    seg = node_to_segment.get(str(n1["id"]), None)
    cluster_id = clusters.get(seg, 0)

    x_lines += [n1["x"], n2["x"], None]
    y_lines += [n1["y"], n2["y"], None]
    z_lines += [n1["z"], n2["z"], None]

    colors += [cluster_id, cluster_id, cluster_id]

# =========================
# PLOT
# =========================
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=x_lines,
    y=y_lines,
    z=z_lines,
    mode='lines',
    line=dict(
        width=4,
        color=colors,
        colorscale='Viridis',
        colorbar=dict(title="Cluster")
    )
))

fig.update_layout(
    title="Interactive Neuron Morphology (Clusters)",
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.write_html("neuron_3d_interactive.html")
print("Saved: neuron_3d_interactive.html")