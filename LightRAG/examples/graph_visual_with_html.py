import pipmaster as pm

if not pm.is_installed("pyvis"):
    pm.install("pyvis")
if not pm.is_installed("networkx"):
    pm.install("networkx")

import os
import sys
from pathlib import Path

import networkx as nx
from pyvis.network import Network
import random

# Locate the GraphML file written by NetworkXStorage.
# Usage: python graph_visual_with_html.py [path/to/graph_chunk_entity_relation.graphml]
# Without an argument, WORKING_DIR (same variable as the server's .env) is used,
# then the repository default ./rag_storage, then the legacy ./dickens example dir.
GRAPHML_NAME = "graph_chunk_entity_relation.graphml"
if len(sys.argv) > 1:
    graphml_path = Path(sys.argv[1])
else:
    repo_root = Path(__file__).resolve().parent.parent
    candidates = []
    if os.getenv("WORKING_DIR"):
        candidates.append(Path(os.environ["WORKING_DIR"]) / GRAPHML_NAME)
    candidates += [
        repo_root / "rag_storage" / GRAPHML_NAME,
        Path("./dickens") / GRAPHML_NAME,
    ]
    graphml_path = next((c for c in candidates if c.is_file()), candidates[0])

if not graphml_path.is_file():
    sys.exit(f"GraphML file not found: {graphml_path}")

G = nx.read_graphml(graphml_path)
print(
    f"Loaded {graphml_path}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
)

# Create a Pyvis network
net = Network(height="100vh", notebook=True)

# Convert NetworkX graph to Pyvis network
net.from_nx(G)


# Add colors and title to nodes
for node in net.nodes:
    node["color"] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    if "description" in node:
        node["title"] = node["description"]

# Add title to edges
for edge in net.edges:
    if "description" in edge:
        edge["title"] = edge["description"]

# Save and display the network
net.show("knowledge_graph.html")
