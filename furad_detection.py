import polars as pl
import networkx as nx
from pyvis.network import Network

# ==========================================
# STEP 1: TABULAR INGESTION (Polars Engine)
# ==========================================
# Creating dummy banking transactions with a multi-hop fraud loop (A -> B -> C -> A)
raw_data = {
    "Transaction_ID": ["T101", "T102", "T103", "T104", "T105", "T106", "T107"],
    "Sender_Acc":     ["Acc_A", "Acc_B", "Acc_C", "Acc_D", "Acc_E", "Acc_F", "Acc_G"],
    "Receiver_Acc":   ["Acc_B", "Acc_C", "Acc_A", "Acc_B", "Acc_B", "Acc_C", "Acc_D"],
    "Amount":         [45000,   45000,   45000,   1200,    3500,    8000,    15000]
}

# Polars processes transaction datasets with high-performance multi-threading
df = pl.DataFrame(raw_data)
print("--- Ingested Transaction Logs via Polars ---")
print(df)

# ==========================================
# STEP 2: GRAPH TRANSFORMATION (NetworkX)
# ==========================================
# Instantiate a directed graph to capture source-to-target capital flow
G = nx.DiGraph()

# Add relationships natively from the Polars DataFrame
for row in df.iter_rows(named=True):
    G.add_edge(
        row["Sender_Acc"], 
        row["Receiver_Acc"], 
        tx_id=row["Transaction_ID"], 
        amount=row["Amount"]
    )

# ==========================================
# STEP 3: CENTRALITY SCORING & LOOP DETECTION
# ==========================================
# Calculate PageRank Centrality to catch suspicious "mule hubs" or loop nodes
pagerank_scores = nx.pagerank(G, weight="amount")

# Extract closed-loop cycles (money laundering circular topology)
detected_cycles = list(nx.simple_cycles(G))

print("\n--- Graph Analytics Results ---")
print("PageRank Metrics (Higher score = More suspicious hub):")
for account, score in sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {account}: {score:.4f}")

print("\nDetected Closed Financial Fraud Loops:")
for cycle in detected_cycles:
    print(f"  🚨 Loop Found: {' -> '.join(cycle)} -> {cycle[0]}")

# ==========================================
# STEP 4: INTERACTIVE VISUALIZATION (Pyvis)
# ==========================================
# Initialize Pyvis interactive layout with physics enabled
net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", directed=True)

# Generate node styling based on threat anomalies (PageRank scores)
for node in G.nodes():
    # Scale node size based on PageRank importance
    node_size = int(pagerank_scores[node] * 100) + 15 
    
    # Flag nodes involved in loops with a distinct color signature
    in_loop = any(node in cycle for cycle in detected_cycles)
    node_color = "#ff4d4d" if in_loop else "#4da6ff"
    
    net.add_node(node, label=node, size=node_size, color=node_color, title=f"PageRank: {pagerank_scores[node]:.4f}")

# Populate visual edges
for source, target, data in G.edges(data=True):
    net.add_edge(source, target, value=data["amount"], title=f"TX: {data['tx_id']} | ₹{data['amount']}")

# Save physics-driven graph view to local file
output_file = "fraud_network.html"
net.write_html(output_file)
print(f"\n--- Process Complete ---\nInteractive fraud map successfully saved to: '{output_file}'")