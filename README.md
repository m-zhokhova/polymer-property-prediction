# Polymer Property Prediction: GNN vs Tree-Based Models

A comparison of **Graph Neural Networks** and **Tree-Based Models** (ExtraTreesRegressor) for predicting polymer properties from molecular structure (SMILES).

Based on the [NeurIPS Open Polymer Prediction 2025](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025) Kaggle competition.

---

## 🎯 Problem

Predict 5 physical properties of polymers from their SMILES representation:

| Property | Description | Unit |
|----------|-------------|------|
| **Tg** | Glass Transition Temperature | Kelvin |
| **FFV** | Fractional Free Volume | - |
| **Tc** | Thermal Conductivity | W/(m·K) |
| **Density** | Mass per Volume | g/cm³ |
| **Rg** | Radius of Gyration | Å |

---

## 🔬 Two Approaches Compared

### 1. Graph Neural Network (GNN)
- Treats molecules as graphs (atoms = nodes, bonds = edges)
- 4-layer GINE architecture with residual connections
- 18 atom features + 7 bond features
- Multi-task learning (predicts all 5 targets simultaneously)

### 2. Tree-Based Models (ExtraTrees + ensemble)
- Uses RDKit molecular descriptors (~200 features)
- Morgan fingerprints (512 bits)
- Ensemble of ExtraTrees, GradientBoosting, RandomForest
- Separate model per target

---

## 📊 Results

| Target | GNN MAE | Tree MAE | Winner | Improvement |
|--------|---------|----------|--------|-------------|
| **Tg** | 46.05 K | 53.31 K | GNN | +13.6% |
| **FFV** | 0.0049 | 0.0069 | GNN | +28.6% |
| **Tc** | 0.0296 | 0.0293 | Tree | +1.1% |
| **Density** | 0.0294 | 0.0325 | GNN | +9.5% |
| **Rg** | 1.40 | 1.73 | GNN | +18.9% |

**GNN wins 4 out of 5 targets**, but tree models remain competitive — especially considering they train in seconds vs. minutes for GNN.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/polymer-property-prediction.git
cd polymer-property-prediction

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebook

```bash
jupyter notebook polymer_gnn_vs_trees.ipynb
```

Or run directly on [Kaggle](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025) where the data is available.

---

## 📁 Project Structure

```
polymer-property-prediction/
│
├── polymer_gnn_vs_trees.ipynb   # Main comparison notebook
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
└── data/                        # Data directory (not included)
    ├── train.csv
    └── test.csv
```

### Data

The dataset is from the Kaggle competition. To run locally:
1. Download from [Kaggle](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/data)
2. Place `train.csv` and `test.csv` in the `data/` folder
3. Update file paths in the notebook

---

## 🔑 Key Findings

1. **GNN excels at structural properties** — Tg and Rg depend heavily on molecular topology, which GNNs capture naturally through message passing.

2. **Tree models are surprisingly competitive** — With good feature engineering (RDKit + Morgan fingerprints), they achieve ~80-90% of GNN performance.

3. **Data scarcity hurts both** — Only 667 samples for Tg resulted in ~46K MAE for both approaches. More data would help significantly.

4. **Ensemble is best** — Combining GNN (40%) + Trees (60%) gives more robust predictions than either alone.

---

## 📚 Methods Explained

### ExtraTreesRegressor

Extremely Randomized Trees — a variant of Random Forest that uses random splits instead of optimal splits:

```python
from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
```

**Why it works well here:**
- Handles high-dimensional sparse features (Morgan fingerprints)
- Robust to outliers
- Fast training
- No need for feature scaling

### GNN Architecture

```
Input → NodeEmbed → [GINE + LayerNorm + Residual] × 4 → MultiPooling → MLP → Output
```

- **GINE**: Graph Isomorphism Network with Edge features
- **Multi-pooling**: Concatenates mean, max, and sum pooling
- **Residual connections**: Helps with gradient flow

---

## 🛠️ Possible Improvements

- [ ] Add external datasets for Tg and Tc
- [ ] Try XGBoost / LightGBM
- [ ] Implement 5-fold cross-validation
- [ ] Use pretrained molecular embeddings (ChemBERTa)
- [ ] Hyperparameter tuning with Optuna
- [ ] Add 3D molecular coordinates

---

## 📖 Related Article

*Coming soon on Medium — a deep dive into ExtraTreesRegressor and when "boring" models win.*

---

## 📜 License

MIT License — feel free to use, modify, and share.

---

## 👩‍💻 Author

**Maria Zhokhova**  
Data Science MSc @ FCUP/INESC-TEC  
[LinkedIn](https://www.linkedin.com/in/m-zhokhova/)

---

## 🙏 Acknowledgments

- [NeurIPS Open Polymer Prediction 2025](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025) competition organizers
- [RDKit](https://www.rdkit.org/) for molecular descriptors
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) for GNN implementation
