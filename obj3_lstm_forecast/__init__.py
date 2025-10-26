"""
Objective 3: Hybrid LSTM Demand Forecasting

This module handles:
- Rule-based sales data simulation (60 data points)
- CLIP embeddings extraction (768-dim)
- Hybrid LSTM architecture (2-layer, hidden_dim=128)
- GRU baseline comparison
- Model training & evaluation (R² > 0.7)

Target: Predict sales for 28 generated designs
"""

__version__ = "1.0.0"
