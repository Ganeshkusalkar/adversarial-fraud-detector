import streamlit as st
from typing import Dict, Any

def render_metrics_cards(metrics_data: Dict[str, Any]) -> None:
    """
    Renders business metrics and Crore savings dashboard cards.
    """
    st.subheader("Financial ROI & System Health")
    
    # Calculate savings in Crores (1 Crore = 10,000,000 Rupees or general unit)
    total_blocked_amt = metrics_data.get("total_blocked_amount", 0.0)
    savings_crores = total_blocked_amt / 10_000_000.0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Blocked Volume", 
            value=f"₹{savings_crores:.3f} Cr", 
            delta=f"+₹{(metrics_data.get('recent_blocked', 0.0) / 10_000_000.0):.4f} Cr"
        )
    with col2:
        st.metric(
            label="Flagged Transactions", 
            value=metrics_data.get("flag_count", 0), 
            delta=f"+{metrics_data.get('recent_flags', 0)}"
        )
    with col3:
        st.metric(
            label="Active TPS", 
            value=f"{metrics_data.get('tps', 0.0):.2f}",
            delta=f"{metrics_data.get('tps_delta', 0.0):+.1f} TPS"
        )
    with col4:
        st.metric(
            label="Avg API Latency", 
            value=f"{metrics_data.get('latency_ms', 0.0):.1f} ms",
            delta=None
        )
        
    st.markdown("---")
