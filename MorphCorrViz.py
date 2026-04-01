import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# =========================
# Load data
# =========================
with open("neuron_morphology_data.json") as f:
    data = json.load(f)

nodes = data["nodes"]
clusters = data["runs"]["1ms"]["cluster"]   # or "1s"
node_to_segment = data["node_to_segment"]

# Build quick lookup
node_dict = {n["id"]: n for n in nodes}

# Colormap
cmap = cm.get_cmap("tab10")

# =========================
# Plot
# =========================
plt.figure(figsize=(8, 8))

for n in nodes:
    parent_id = n["parent"]
    if parent_id == -1:
        continue
    if n["id"] == 1:
        plt.scatter(x1, y1, color="black", s=30, zorder=5)

    parent = node_dict[parent_id]

    x1, y1 = n["x"], n["y"]
    x2, y2 = parent["x"], parent["y"]

    # Get cluster
    seg_key = node_to_segment[str(n["id"])]
    cluster = clusters.get(seg_key, 0)

    color = cmap(cluster)

    plt.plot([x1, x2], [y1, y2],
             color=color,
             linewidth=1.5,
             alpha=0.9)

plt.title("Neuron Morphology (Colored by Correlation Cluster)")
plt.xlabel("X")
plt.ylabel("Y")
#plt.gca().invert_yaxis()
plt.axis("equal")
plt.grid(alpha=0.2)

plt.show()

plt.savefig("morphology_clusters_2d.png", dpi=300)

