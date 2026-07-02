# Structural Topology of the Spanish Day-Ahead Electricity Market (OMIE)

**Author:** Eoin Gallagher  
**Institution:** Universidad Carlos III de Madrid (UC3M)  
**Supervisor:** Dr. Andrés M. Alonso Fernández  

## 📌 Project Overview
This repository contains the complete computational pipeline for my Master's thesis. The research introduces a novel, unsupervised machine learning framework to analyze the structural topology of the Iberian Day-Ahead Market (OMIE). 

Instead of traditional price-point forecasting, this codebase treats aggregate supply and demand bidding curves as continuous, high-dimensional geometric objects. By applying a Directed Hausdorff distance metric and Hierarchical Clustering, it identifies deterministic market regimes and maps them against physical grid constraints (VRE integration, demand loads) sourced from the ESIOS platform.

## 🏗️ Pipeline Architecture and Repository Structure

The methodology is split into a sequential 5-phase execution pipeline:

### Phase 1: Curve Simplification & Origin Rectification (`Phase_1.py`)
* Ingests raw OMIE auction data.
* Compresses bidding curves using the Ramer-Douglas-Peucker algorithm, achieving a ~55% reduction in data points without violating economic or monotonicity constraints.
* Injects strict origin points to ensure downstream geometric algorithms capture true economic willingness-to-pay/sell.

### Phase 2: Non-Euclidean Distance Space Assembly (`Phase_2.py`)
* Computes the relational topologies of the curves using the **Directed Hausdorff metric**.
* Due to the $O(N^2)$ computational complexity, this phase includes specific scripts designed for distributed High-Performance Computing (HPC) environments (`stage1_supply.py`, `stage1_supply_resume.py`).

### Phase 3: Unsupervised Spatial Clustering (`Phase_3.py`)
* Executes Agglomerative Hierarchical Clustering (AHC) on the distance matrices.
* Evaluates partition stability via Silhouette scoring to isolate optimal structural medoids ($k=3$ distinct archetypes for both supply and demand).

### Phase 4: Exogenous Market Characterisation (`Phase_4.py`)
* Aligns the discrete cluster categorical time series with continuous physical grid forecasts (Wind MW, Solar PV MW, Demand Forecasts) using ESIOS data.
* Computes joint-probability intersection matrices to isolate natural market equilibriums and structural incompatibilities.

## ⚙️ Computational Environment
To reproduce this pipeline, ensure the following dependencies are installed:
* `pandas`, `numpy`, `scipy`, `scikit-learn`
* `seaborn`, `matplotlib` (Headless mode supported for HPC execution)
* `joblib` (for parallelization and matrix caching)
* `holidays` (for calendar effect isolation)

## 📊 Data Availability Note
While the full algorithmic pipeline is open-source, the raw foundational datasets (OMIE bidding curves and ESIOS grid variables) are subject to platform usage limits and size constraints. Researchers must pull their own raw queries from the respective OMIE and ESIOS API portals to run the `Phase_1.py` initialization.
