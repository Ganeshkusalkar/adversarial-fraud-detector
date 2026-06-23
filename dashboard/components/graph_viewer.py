import streamlit as st
import networkx as nx
from typing import List, Tuple, Dict, Any

# Fallback visualization options
try:
    from pyvis.network import Network
    import streamlit.components.v1 as components
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False

def render_graph_viewer(flagged_paths: List[Tuple[int, int]], node_risk_scores: Dict[int, float]) -> None:
    """
    Renders an interactive topological graph visualizer of flagged transaction paths.
    """
    st.subheader("Interactive Suspect Transaction Paths")
    
    if not flagged_paths:
        st.info("No active suspect paths flagged in the current stream window.")
        return

    st.markdown("Visualizing accounts (nodes) and high-risk transfer flow pathways (edges).")

    if HAS_PYVIS:
        try:
            # Construct pyvis network graph
            net = Network(height="400px", width="100%", bgcolor="#0f1116", font_color="#e0e0e0", directed=True)
            
            # Populate nodes and risk highlights
            for node_id, risk in node_risk_scores.items():
                # Color code based on risk scale: Red for high risk, Green for low risk
                color = "#ff4b4b" if risk >= 0.80 else "#2ca02c"
                size = 15 + int(risk * 15)
                net.add_node(
                    node_id, 
                    label=f"Acct:{node_id}\nRisk:{risk:.2f}",
                    color=color,
                    size=size,
                    title=f"Account ID: {node_id}\nFraud Risk Score: {risk * 100:.1f}%"
                )
                
            # Populate edges
            for src, dst in flagged_paths:
                net.add_edge(src, dst, color="#d3d3d3", width=2)
                
            net.set_options("""
            var options = {
              "physics": {
                "barnesHut": {
                  "gravitationalConstant": -4000,
                  "centralGravity": 0.3,
                  "springLength": 95
                }
              }
            }
            """)
            
            # Save and embed output HTML in streamlit iframe
            net.save("temp_graph.html")
            with open("temp_graph.html", "r", encoding="utf-8") as f:
                html_content = f.read()
                
            components.html(html_content, height=420)
            return
        except Exception as e:
            st.error(f"Failed to render interactive graph: {e}")
            
    # Simple static fallback when Pyvis is missing
    st.warning("Install `pyvis` package for full interactive graph viewer functionality.")
    
    # Render static table structure of the connections
    data_list = []
    for src, dst in flagged_paths:
        data_list.append({
            "Source Account": f"C{src}",
            "Destination Account": f"C{dst}",
            "Source Risk Score": f"{node_risk_scores.get(src, 0.0) * 100:.1f}%",
            "Destination Risk Score": f"{node_risk_scores.get(dst, 0.0) * 100:.1f}%",
            "Status": "HIGH RISK ALERT" if (node_risk_scores.get(src, 0.0) >= 0.80 or node_risk_scores.get(dst, 0.0) >= 0.80) else "Monitoring"
        })
    st.table(data_list)
