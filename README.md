# 🌊 FloodNet Impact Mapper

**INTELLATHON-26 · Team CODEVERSE**

> A GIS decision-support prototype that links flood exposure, critical-service dependencies, road-network access, and population impact into one explainable risk score and a ranked intervention list.

> ⚠️ **ALL DATA IS SIMULATED.** Terrain, population, flood scenarios, road network, dependency topology and asset locations are synthetic. This is a hackathon prototype — not for operational emergency use.

---

## 🚀 Quick Start (one command)

```bash
# 1. Clone / enter the project folder
cd codeverse

# 2. Install dependencies (Python 3.11+ required)
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**

---

## 📁 File Structure

```
codeverse/
├── app.py              ← Main Streamlit application (single file)
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
└── sample_assets.csv   ← Upload-ready demo CSV with 14 critical assets
```

---

## 🗺️ Study Area

**Perambalur, Tamil Nadu, India** (11.2342°N, 78.8808°E)  
~10 km × 10 km synthetic study area

---

## 🔧 Features

| Feature | Description |
|---------|-------------|
| **Flood Model** | Bathtub model + BFS hydraulic connectivity from river |
| **Exposure Analysis** | Site depth + population within 1.5 km radius |
| **Road Access** | NetworkX graph; speed degrades with depth; Dijkstra routing |
| **Dependency Graph** | Directed power/water dependency cascade (ASSUMED topology) |
| **Risk Score (0–100)** | Weighted sum of 5 normalised factors; adjustable weights |
| **"Why" Explanation** | Plain-English per-asset reason text |
| **Intervention Ranking** | Sorted table + factor breakdown chart + download |
| **Scenario Sweep** | Pre-computed tipping-point chart across 0.5–6 m rise |

---

## 🎛️ UI Controls

### Sidebar
- **Scenario Presets**: Custom / 1-in-10 yr / 1-in-50 yr / 1-in-100 yr (all simulated)
- **Water Rise Slider**: 0–6 m
- **Road Closure Depth**: depth at which a road segment becomes impassable
- **5 Weight Sliders**: adjust relative importance of risk factors
- **CSV Upload**: upload your own assets (format: `name, type, lat, lon`)

### Tabs
1. 🗺️ **Impact Map** — Folium interactive map with flood overlay, road colours, asset markers
2. 📊 **Intervention Ranking** — Sorted table with factor bars + CSV/GeoJSON download
3. 🕸️ **Dependencies** — Plotly dependency network graph
4. 📈 **Scenario Sweep** — Tipping-point line chart
5. 📖 **Methodology** — Full pipeline documentation

---

## 📊 Custom Asset CSV Format

```csv
name,type,lat,lon
My Hospital,hospital,11.2380,78.8820
Town Shelter,shelter,11.2350,78.8770
```

**Allowed types:** `hospital`, `shelter`, `school`, `substation`, `water_plant`

---

## 🔬 Simulation Details

| Component | Method |
|-----------|--------|
| DEM | Gaussian river valley + smooth random noise |
| Population | Gaussian clusters (town + floodplain + villages) |
| Flood model | Bathtub fill + BFS connected-component analysis |
| Road network | 9×9 regular lattice (simulated Perambalur road grid) |
| Dependencies | Nearest-source heuristic (NOT real utility topology) |
| Assets | 14 representative facilities (manually placed) |

---

## 🔌 Real-Data Integration Points

The code is commented with swap-in points:

```python
# terrain()  → load real GeoTIFF DEM + WorldPop raster
# build_roads() → osmnx.graph_from_place("Perambalur, Tamil Nadu")
# demo_assets() → OSM query for hospitals, shelters, schools
```

---

## 🏆 Team CODEVERSE — INTELLATHON-26

Built with: Python 3.11 · Streamlit · Folium · Plotly · NetworkX · NumPy · Pandas · SciPy
