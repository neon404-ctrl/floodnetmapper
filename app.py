"""
FloodNet Impact Mapper
======================
INTELLATHON-26 | Team CODEVERSE
A GIS decision-support tool linking flood exposure, service dependencies,
road access, and population impact into an explainable risk score and
ranked intervention list.

⚠️  ALL TERRAIN, POPULATION, FLOOD, DEPENDENCY AND ROAD DATA ARE SIMULATED.
    This is a prototype. Do not use for real emergency decisions.
"""

import io
import json
import math
import textwrap
from collections import deque

import folium
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from scipy.ndimage import gaussian_filter

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & STUDY AREA
# ─────────────────────────────────────────────────────────────────────────────
CENTER_LAT = 11.2342
CENTER_LON = 78.8808
GRID_N = 120                       # DEM resolution (120 × 120 cells ≈ 83 m/cell)
CELL_KM = 10.0 / GRID_N            # ~0.083 km per cell
DEG_PER_KM_LAT = 1.0 / 111.0
DEG_PER_KM_LON = 1.0 / (111.0 * math.cos(math.radians(CENTER_LAT)))
LAT0 = CENTER_LAT - 5 * DEG_PER_KM_LAT   # SW corner
LON0 = CENTER_LON - 5 * DEG_PER_KM_LON

ASSET_TYPES = ["hospital", "shelter", "school", "substation", "water_plant"]

# Criticality weight per asset type (0-1)
CRITICALITY = {
    "hospital":   1.00,
    "shelter":    0.85,
    "water_plant":0.90,
    "substation": 0.80,
    "school":     0.55,
}

# Direct-failure depth threshold (m): asset is considered failed if depth > this
FAIL_DEPTH = {
    "hospital":    0.30,
    "shelter":     0.50,
    "water_plant": 0.20,
    "substation":  0.10,
    "school":      0.60,
}

# Scenario presets  [water_rise_m]
PRESETS = {
    "Custom":                 None,
    "1-in-10 yr (simulated)": 1.2,
    "1-in-50 yr (simulated)": 2.5,
    "1-in-100 yr (simulated)":4.0,
}

ROAD_NODES = 9   # 9 × 9 lattice

# ─────────────────────────────────────────────────────────────────────────────
# 1.  TERRAIN  (SIMULATED — swap terrain() for a real GeoTIFF DEM + WorldPop)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def build_terrain():
    """
    Returns (dem, pop_grid) each of shape (GRID_N, GRID_N).
    dem  : elevation in metres above an arbitrary datum
    pop  : population density (people per cell)

    SWAP-IN POINT:
      Replace this function body with:
        dem  = read_geotiff("perambalur_dem_30m.tif", GRID_N)
        pop  = read_worldpop("worldpop_ind_2020.tif", GRID_N)
    """
    rng = np.random.default_rng(42)
    x = np.linspace(0, 1, GRID_N)
    y = np.linspace(0, 1, GRID_N)
    xx, yy = np.meshgrid(x, y)

    # --- River valley: a meandering trough running roughly NW→SE
    t = np.linspace(0, 1, 200)
    river_x = t
    river_y = 0.5 + 0.12 * np.sin(2 * np.pi * t * 1.5) + 0.06 * np.sin(2 * np.pi * t * 4)
    # Distance from each cell to river centre-line
    dist_to_river = np.min(
        np.sqrt((xx[:, :, None] - river_x[None, None, :]) ** 2
                + (yy[:, :, None] - river_y[None, None, :]) ** 2),
        axis=2,
    )
    valley_depth = 8.0   # metres below surrounding ground
    valley_width = 0.08  # normalised units
    valley = valley_depth * np.exp(-0.5 * (dist_to_river / valley_width) ** 2)

    # Base slope (gentle W→E rise)
    base = 40.0 + 12.0 * xx

    # Smooth random noise (local topography)
    noise = gaussian_filter(rng.uniform(-3, 3, (GRID_N, GRID_N)), sigma=6)

    dem = base - valley + noise
    dem = dem - dem.min() + 1.0   # shift so minimum = 1 m (above datum)

    # --- Population density
    # Town cluster near centre
    town = np.exp(-0.5 * (((xx - 0.48) / 0.12) ** 2 + ((yy - 0.50) / 0.12) ** 2))
    # Floodplain population (higher risk area)
    flood_pop = 0.6 * np.exp(-0.5 * (dist_to_river / 0.10) ** 2)
    # Scattered villages
    v1 = 0.35 * np.exp(-0.5 * (((xx - 0.25) / 0.08) ** 2 + ((yy - 0.30) / 0.08) ** 2))
    v2 = 0.25 * np.exp(-0.5 * (((xx - 0.75) / 0.07) ** 2 + ((yy - 0.70) / 0.07) ** 2))
    pop_raw = town + flood_pop + v1 + v2 + 0.05 * rng.uniform(0, 1, (GRID_N, GRID_N))
    pop_grid = np.round(pop_raw * 200).astype(float)  # scale to people per cell

    return dem, pop_grid


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ASSETS  (SIMULATED — swap demo_assets() for OSM / real dataset)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def demo_assets():
    """
    Returns a DataFrame with columns: name, type, lat, lon

    SWAP-IN POINT:
      Replace with OSM query via osmnx:
        tags = {"amenity": ["hospital","school"], "emergency": "shelter", ...}
        gdf = ox.features_from_place("Perambalur, Tamil Nadu", tags)
    """
    data = [
        ("Perambalur District Hospital",  "hospital",    11.2380, 78.8820),
        ("Government General Hospital",   "hospital",    11.2290, 78.8750),
        ("Primary Health Centre East",    "hospital",    11.2420, 78.8950),
        ("Town Relief Shelter A",         "shelter",     11.2350, 78.8770),
        ("Flood Camp Shelter B",          "shelter",     11.2260, 78.8830),
        ("High-Ground Shelter C",         "shelter",     11.2450, 78.8700),
        ("Perambalur Town School",        "school",      11.2340, 78.8810),
        ("Government Higher Secondary",   "school",      11.2310, 78.8760),
        ("East Zone Primary School",      "school",      11.2400, 78.8900),
        ("Central Substation",            "substation",  11.2360, 78.8790),
        ("North Substation",              "substation",  11.2430, 78.8720),
        ("South Substation",              "substation",  11.2270, 78.8870),
        ("Water Treatment Plant A",       "water_plant", 11.2300, 78.8800),
        ("Water Treatment Plant B",       "water_plant", 11.2410, 78.8860),
    ]
    return pd.DataFrame(data, columns=["name", "type", "lat", "lon"])


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ROAD NETWORK  (SIMULATED — swap build_roads() for OSMnx graph)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def build_roads():
    """
    Returns (G, node_positions, hub_node) where:
      G              : NetworkX Graph with edge attribute 'length_km'
      node_positions : dict {node_id: (lat, lon)}
      hub_node       : id of the emergency-response hub (high ground)

    SWAP-IN POINT:
      G = ox.graph_from_place("Perambalur, Tamil Nadu", network_type="drive")
      G = ox.add_edge_speeds(G); G = ox.add_edge_travel_times(G)
    """
    n = ROAD_NODES
    G = nx.Graph()
    pos = {}
    lat_step = 10.0 / (n - 1) * DEG_PER_KM_LAT
    lon_step = 10.0 / (n - 1) * DEG_PER_KM_LON

    # Build 9×9 lattice
    for i in range(n):
        for j in range(n):
            node = i * n + j
            lat = LAT0 + i * lat_step
            lon = LON0 + j * lon_step
            pos[node] = (lat, lon)
            G.add_node(node, lat=lat, lon=lon)

    for i in range(n):
        for j in range(n):
            node = i * n + j
            # horizontal edge
            if j + 1 < n:
                G.add_edge(node, node + 1, length_km=CELL_KM * 10.0 / (n - 1))
            # vertical edge
            if i + 1 < n:
                G.add_edge(node, node + n, length_km=CELL_KM * 10.0 / (n - 1))

    # Hub: top-left quadrant (high ground, west side)
    hub = (n - 1) * n + 0   # top-left corner node
    return G, pos, hub


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: coordinate ↔ grid index
# ─────────────────────────────────────────────────────────────────────────────
def latlon_to_grid(lat, lon):
    """Convert geographic coordinates to (row, col) in the DEM grid."""
    row = int((lat - LAT0) / (10.0 * DEG_PER_KM_LAT) * (GRID_N - 1))
    col = int((lon - LON0) / (10.0 * DEG_PER_KM_LON) * (GRID_N - 1))
    row = max(0, min(GRID_N - 1, row))
    col = max(0, min(GRID_N - 1, col))
    return row, col


def grid_to_latlon(row, col):
    lat = LAT0 + row / (GRID_N - 1) * 10.0 * DEG_PER_KM_LAT
    lon = LON0 + col / (GRID_N - 1) * 10.0 * DEG_PER_KM_LON
    return lat, lon


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FLOOD MODEL — Bathtub + BFS hydraulic connectivity
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def compute_flood(dem: np.ndarray, water_rise: float):
    """
    Bathtub flood model with BFS connectivity to the river.

    Returns:
        flood_depth : (GRID_N, GRID_N) array, metres; 0 where not flooded
        flood_mask  : bool array

    SIMULATED: This is a bathtub model, not a hydrodynamic simulation.
    A real implementation would use HEC-RAS, LISFLOOD-FP, or similar.
    """
    if water_rise <= 0:
        return np.zeros_like(dem), np.zeros(dem.shape, dtype=bool)

    # River base level: minimum elevation + a small offset
    river_base = dem.min() + 1.0
    flood_level = river_base + water_rise

    # Seed BFS from all river cells (cells very close to minimum elevation)
    river_threshold = river_base + 0.5
    seed_mask = dem <= river_threshold

    # BFS flood fill: a cell is flooded if it is reachable from a seed cell
    # while remaining below flood_level
    flooded = np.zeros(dem.shape, dtype=bool)
    queue = deque()
    for r in range(GRID_N):
        for c in range(GRID_N):
            if seed_mask[r, c] and dem[r, c] < flood_level:
                flooded[r, c] = True
                queue.append((r, c))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_N and 0 <= nc < GRID_N:
                if not flooded[nr, nc] and dem[nr, nc] < flood_level:
                    flooded[nr, nc] = True
                    queue.append((nr, nc))

    depth = np.where(flooded, flood_level - dem, 0.0)
    depth = np.maximum(depth, 0.0)
    return depth, flooded


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ROAD ACCESS — speed degradation + Dijkstra travel times
# ─────────────────────────────────────────────────────────────────────────────
def latlon_to_road_node(lat, lon, node_pos):
    """Find nearest road node by Euclidean distance in lat/lon space."""
    best_node, best_d = None, float("inf")
    for nid, (nlat, nlon) in node_pos.items():
        d = (lat - nlat) ** 2 + (lon - nlon) ** 2
        if d < best_d:
            best_d = d
            best_node = nid
    return best_node


def sample_depth_at_edge(u, v, node_pos, flood_depth, factor=0.5):
    """Sample flood depth at the midpoint of a road edge."""
    (la, lo_a) = node_pos[u]
    (lb, lo_b) = node_pos[v]
    mlat = (la + lb) / 2
    mlon = (lo_a + lo_b) / 2
    r, c = latlon_to_grid(mlat, mlon)
    return float(flood_depth[r, c])


def compute_road_access(G, node_pos, hub_node, flood_depth, closure_depth,
                        assets_df):
    """
    Build a weighted road graph where:
      - Edges above closure_depth are removed (impassable)
      - Edges with depth 0 → closure_depth get speed penalty (speed ∝ 1/depth)
    Returns:
      edge_status   : dict {(u,v): 'open'|'slow'|'closed'}
      travel_times  : dict {asset_idx: travel_time_minutes or None}
      baseline_times: dict {asset_idx: dry_travel_time_minutes}
      cut_off_nodes : set of asset indices that are unreachable
      links_cut     : int
    """
    BASE_SPEED_KMH = 40.0   # dry road speed

    Gw = nx.Graph()
    Gw.add_nodes_from(G.nodes())
    edge_status = {}
    links_cut = 0

    for u, v, data in G.edges(data=True):
        depth = sample_depth_at_edge(u, v, node_pos, flood_depth)
        length_km = data.get("length_km", CELL_KM)

        if depth >= closure_depth:
            edge_status[(u, v)] = "closed"
            links_cut += 1
            continue   # edge removed from Gw

        if depth > 0.1:
            # Speed degrades linearly to 5 km/h at closure_depth
            fraction = depth / closure_depth
            speed = BASE_SPEED_KMH * (1 - 0.875 * fraction)  # 40→5 km/h
            edge_status[(u, v)] = "slow"
        else:
            speed = BASE_SPEED_KMH
            edge_status[(u, v)] = "open"

        travel_min = (length_km / speed) * 60.0
        Gw.add_edge(u, v, weight=travel_min, length_km=length_km)

    # Dry baseline graph (no flood)
    Gdry = nx.Graph()
    Gdry.add_nodes_from(G.nodes())
    for u, v, data in G.edges(data=True):
        length_km = data.get("length_km", CELL_KM)
        travel_min = (length_km / BASE_SPEED_KMH) * 60.0
        Gdry.add_edge(u, v, weight=travel_min)

    travel_times = {}
    baseline_times = {}
    cut_off_nodes = set()

    for idx, row in assets_df.iterrows():
        node = latlon_to_road_node(row["lat"], row["lon"], node_pos)
        # Dry baseline
        try:
            t_dry = nx.shortest_path_length(Gdry, hub_node, node, weight="weight")
        except nx.NetworkXNoPath:
            t_dry = None
        baseline_times[idx] = t_dry

        # Flooded network
        if hub_node not in Gw or node not in Gw:
            travel_times[idx] = None
            cut_off_nodes.add(idx)
        else:
            try:
                t = nx.shortest_path_length(Gw, hub_node, node, weight="weight")
                travel_times[idx] = t
            except nx.NetworkXNoPath:
                travel_times[idx] = None
                cut_off_nodes.add(idx)

    return edge_status, travel_times, baseline_times, cut_off_nodes, links_cut


# ─────────────────────────────────────────────────────────────────────────────
# 6.  DEPENDENCY GRAPH — power + water cascade failures
# ─────────────────────────────────────────────────────────────────────────────
def build_dependency_graph(assets_df):
    """
    Directed dependency graph (ASSUMED TOPOLOGY — nearest-source rule).

    Rules (labelled ASSUMED in the UI):
      - Each non-substation asset gets power from its nearest substation.
      - Hospitals and shelters get water from their nearest water_plant.
      - Each water_plant gets power from its nearest substation.

    SWAP-IN POINT: replace with real utility network topology data.
    """
    DG = nx.DiGraph()
    DG.add_nodes_from(assets_df.index)

    substations = assets_df[assets_df["type"] == "substation"]
    water_plants = assets_df[assets_df["type"] == "water_plant"]

    def nearest(src_lat, src_lon, candidates):
        if candidates.empty:
            return None
        dists = np.sqrt((candidates["lat"] - src_lat) ** 2 +
                        (candidates["lon"] - src_lon) ** 2)
        return int(dists.idxmin())

    for idx, row in assets_df.iterrows():
        atype = row["type"]
        if atype == "substation":
            continue  # substations are sources (no upstream power dependency)
        # Power edge: substation → asset
        sub_idx = nearest(row["lat"], row["lon"], substations)
        if sub_idx is not None:
            DG.add_edge(sub_idx, idx, dep_type="power")

        # Water edge: water_plant → hospital/shelter
        if atype in ("hospital", "shelter"):
            wp_idx = nearest(row["lat"], row["lon"], water_plants)
            if wp_idx is not None:
                DG.add_edge(wp_idx, idx, dep_type="water")

    # Water plants need power from nearest substation
    for idx, row in assets_df[assets_df["type"] == "water_plant"].iterrows():
        sub_idx = nearest(row["lat"], row["lon"], substations)
        if sub_idx is not None:
            DG.add_edge(sub_idx, idx, dep_type="power")

    return DG


def compute_cascade(assets_df, dep_graph, direct_failed: set):
    """
    Propagate failures through the dependency graph.
    Returns dict {idx: cascade_fraction} = share of upstream deps that failed.
    """
    cascade_frac = {}
    for idx in assets_df.index:
        preds = list(dep_graph.predecessors(idx))
        if not preds:
            cascade_frac[idx] = 0.0
            continue
        n_failed = sum(1 for p in preds if p in direct_failed)
        cascade_frac[idx] = n_failed / len(preds)
    return cascade_frac


# ─────────────────────────────────────────────────────────────────────────────
# 7.  EXPOSURE — depth at site + exposed population within 1.5 km
# ─────────────────────────────────────────────────────────────────────────────
def compute_exposure(assets_df, flood_depth, pop_grid):
    """
    Returns:
      site_depth   : {idx: depth_m}
      exposed_pop  : {idx: people}
    """
    RADIUS_KM = 1.5
    radius_cells = int(RADIUS_KM / CELL_KM)

    site_depth = {}
    exposed_pop = {}

    for idx, row in assets_df.iterrows():
        r, c = latlon_to_grid(row["lat"], row["lon"])
        site_depth[idx] = float(flood_depth[r, c])

        # Sum population in flooded cells within radius
        r0 = max(0, r - radius_cells)
        r1 = min(GRID_N, r + radius_cells + 1)
        c0 = max(0, c - radius_cells)
        c1 = min(GRID_N, c + radius_cells + 1)
        patch_depth = flood_depth[r0:r1, c0:c1]
        patch_pop = pop_grid[r0:r1, c0:c1]
        exposed_pop[idx] = int(np.sum(patch_pop[patch_depth > 0]))

    return site_depth, exposed_pop


# ─────────────────────────────────────────────────────────────────────────────
# 8.  IMPACT RISK SCORE + "WHY" EXPLANATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_scores(assets_df, site_depth, exposed_pop, travel_times,
                   baseline_times, cut_off_set, cascade_frac, weights):
    """
    Impact Risk Score (0-100) = weighted sum of five normalised factors.
    weights: dict with keys hazard, criticality, population, access, cascade
             (values in 0-100; normalised internally to sum to 1)
    """
    MAX_POP = max(exposed_pop.values()) if exposed_pop and max(exposed_pop.values()) > 0 else 1
    MAX_TT  = max((v for v in travel_times.values() if v is not None), default=1) or 1

    # Normalise weights (guard against all-zero)
    w = np.array([weights["hazard"], weights["criticality"],
                  weights["population"], weights["access"], weights["cascade"]], dtype=float)
    if w.sum() == 0:
        w = np.ones(5)
    w = w / w.sum()

    results = []
    for idx, row in assets_df.iterrows():
        atype = row["type"]
        depth = site_depth[idx]

        # F1: Hazard (depth / 2m, capped at 1)
        f_hazard = min(depth / 2.0, 1.0)

        # F2: Criticality (asset-type based)
        f_crit = CRITICALITY.get(atype, 0.5)

        # F3: Population exposed (normalised)
        f_pop = min(exposed_pop[idx] / MAX_POP, 1.0)

        # F4: Access loss
        tt = travel_times[idx]
        bl = baseline_times[idx]
        if idx in cut_off_set:
            f_access = 1.0
        elif tt is not None and bl is not None and bl > 0:
            f_access = min((tt - bl) / (bl * 5), 1.0)   # 5× slowdown = max
            f_access = max(f_access, 0.0)
        else:
            f_access = 0.0

        # F5: Cascade failure fraction
        f_cascade = cascade_frac.get(idx, 0.0)

        factors = np.array([f_hazard, f_crit, f_pop, f_access, f_cascade])
        score = float(np.dot(w, factors) * 100)

        # Direct failure?
        threshold = FAIL_DEPTH.get(atype, 0.5)
        direct_fail = depth > threshold

        # Status label
        if direct_fail:
            status = "Direct Failure"
        elif f_cascade > 0:
            status = "Cascade Failure"
        elif idx in cut_off_set:
            status = "Cut Off"
        elif f_access > 0:
            status = "Degraded Access"
        else:
            status = "Operational"

        # --- WHY text
        why_parts = []
        if depth > 0.05:
            why_parts.append(f"{depth:.1f} m water at site")
        if idx in cut_off_set:
            why_parts.append("no passable road to hub")
        elif f_access > 0.3:
            why_parts.append(f"access time {tt:.0f} min vs {bl:.0f} min dry")
        if f_cascade > 0:
            failed_preds = [assets_df.loc[p, "name"]
                            for p in assets_df.index
                            if p in cut_off_set or site_depth[p] > FAIL_DEPTH.get(assets_df.loc[p, "type"], 0.5)]
            # filter to actual predecessors
            from networkx import DiGraph  # local ref
            # we'll fill this after building dep graph outside; placeholder
            why_parts.append(f"{f_cascade*100:.0f}% upstream dependencies failed")
        if exposed_pop[idx] > 0:
            why_parts.append(f"{exposed_pop[idx]:,} people in flooded 1.5 km radius")
        why_text = "; ".join(why_parts) if why_parts else "No significant impact detected"

        # --- Suggested action
        if direct_fail and idx in cut_off_set:
            action = "🚨 Immediate evacuation + emergency access route"
        elif direct_fail:
            action = "🔴 Flood-proof facility or pre-position resources"
        elif idx in cut_off_set:
            action = "🟠 Establish alternate access / airlift capability"
        elif f_cascade > 0.5:
            action = "⚡ Pre-position backup power/water generator"
        elif f_cascade > 0:
            action = "⚡ Backup power/water supply recommended"
        elif score >= 50:
            action = "🔶 Deploy flood barriers; pre-position emergency team"
        elif score >= 25:
            action = "🟡 Monitor closely; prepare contingency plan"
        else:
            action = "✅ Continue routine monitoring"

        tt_display = f"{tt:.1f} min" if tt is not None else "Unreachable"
        bl_display = f"{bl:.1f} min" if bl is not None else "—"

        results.append({
            "idx": idx,
            "name": row["name"],
            "type": atype,
            "lat": row["lat"],
            "lon": row["lon"],
            "score": round(score, 1),
            "status": status,
            "depth_m": round(depth, 2),
            "exposed_pop": exposed_pop[idx],
            "travel_time": tt_display,
            "baseline_time": bl_display,
            "travel_time_raw": tt,
            "baseline_time_raw": bl,
            "f_hazard": round(f_hazard, 3),
            "f_criticality": round(f_crit, 3),
            "f_population": round(f_pop, 3),
            "f_access": round(f_access, 3),
            "f_cascade": round(f_cascade, 3),
            "direct_fail": direct_fail,
            "cut_off": idx in cut_off_set,
            "why": why_text,
            "action": action,
        })

    return pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  SCENARIO SWEEP (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def scenario_sweep(dem, pop_grid, _G, node_pos, hub_node, closure_depth,
                   assets_df):
    """Pre-compute summary statistics across a range of water rise levels."""
    rises = np.arange(0.5, 6.1, 0.5)
    records = []
    for rise in rises:
        fd, _ = compute_flood(dem, rise)
        sd, ep = compute_exposure(assets_df, fd, pop_grid)
        es, tt, bt, co, lc = compute_road_access(
            _G, node_pos, hub_node, fd, closure_depth, assets_df)
        dep = build_dependency_graph(assets_df)
        dir_fail = {i for i, row in assets_df.iterrows()
                    if sd[i] > FAIL_DEPTH.get(row["type"], 0.5)}
        cf = compute_cascade(assets_df, dep, dir_fail)
        casc_fail = sum(1 for i in assets_df.index
                        if i not in dir_fail and cf.get(i, 0) > 0)
        records.append({
            "water_rise": rise,
            "direct_failures": len(dir_fail),
            "cascade_failures": casc_fail,
            "cut_off": len(co),
            "links_cut": lc,
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 10. FOLIUM MAP
# ─────────────────────────────────────────────────────────────────────────────
def make_folium_map(dem, flood_depth, flood_mask, results_df, assets_df,
                    edge_status, node_pos, water_rise):
    """Build a Folium map with flood overlay, roads, and asset markers."""
    m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=13,
                   tiles="CartoDB dark_matter")

    # --- Flood depth raster overlay (down-sampled to 30×30 rectangles)
    step = GRID_N // 30
    color_map = {
        0: "transparent",
        1: "#bde0fe88",
        2: "#5390d988",
        3: "#1d3d7088",
    }
    flood_group = folium.FeatureGroup(name="🌊 Flood Depth", show=True)
    lat_res = 10.0 * DEG_PER_KM_LAT / GRID_N
    lon_res = 10.0 * DEG_PER_KM_LON / GRID_N
    for r in range(0, GRID_N, step):
        for c in range(0, GRID_N, step):
            d = flood_depth[r, c]
            if d <= 0:
                continue
            if d < 0.5:
                col = "#bde0fe99"
            elif d < 1.5:
                col = "#5390d9aa"
            elif d < 3.0:
                col = "#1d3d70bb"
            else:
                col = "#0a1628cc"
            lat1 = LAT0 + r * lat_res
            lon1 = LON0 + c * lon_res
            folium.Rectangle(
                bounds=[[lat1, lon1], [lat1 + step * lat_res, lon1 + step * lon_res]],
                color=None, fill=True, fill_color=col, fill_opacity=0.7,
                tooltip=f"Depth: {d:.1f} m",
            ).add_to(flood_group)
    flood_group.add_to(m)

    # --- Roads
    road_group = folium.FeatureGroup(name="🛣️ Roads", show=True)
    status_colors = {"open": "#22c55e", "slow": "#f59e0b", "closed": "#ef4444"}
    for (u, v), status in edge_status.items():
        la, lo_a = node_pos[u]
        lb, lo_b = node_pos[v]
        folium.PolyLine(
            locations=[[la, lo_a], [lb, lo_b]],
            color=status_colors[status],
            weight=2.5 if status != "closed" else 1.5,
            opacity=0.8 if status != "closed" else 0.4,
            tooltip=f"Road: {status}",
        ).add_to(road_group)
    road_group.add_to(m)

    # --- Hub marker
    hub_lat, hub_lon = node_pos[max(node_pos.keys())]  # top-left = (n-1)*n + 0
    # Recalculate
    n = ROAD_NODES
    hub_node = (n - 1) * n + 0
    hub_lat, hub_lon = node_pos[hub_node]
    folium.Marker(
        location=[hub_lat, hub_lon],
        popup="🚑 Emergency Hub (High Ground)",
        icon=folium.Icon(color="blue", icon="home", prefix="fa"),
    ).add_to(m)

    # --- Asset markers
    asset_group = folium.FeatureGroup(name="🏥 Critical Assets", show=True)
    type_icons = {
        "hospital": ("hospital", "fa"),
        "shelter": ("house", "fa"),
        "school": ("graduation-cap", "fa"),
        "substation": ("bolt", "fa"),
        "water_plant": ("tint", "fa"),
    }
    for _, row in results_df.iterrows():
        score = row["score"]
        if score >= 50:
            color = "red"
        elif score >= 25:
            color = "orange"
        else:
            color = "green"

        icon_name, prefix = type_icons.get(row["type"], ("circle", "fa"))
        crit = CRITICALITY.get(row["type"], 0.5)
        radius = int(8 + crit * 8)

        popup_html = f"""
        <div style='font-family:sans-serif;min-width:220px;'>
          <b style='font-size:14px;'>{row['name']}</b><br>
          <span style='background:{"#ef4444" if score>=50 else "#f59e0b" if score>=25 else "#22c55e"};
                color:white;padding:2px 6px;border-radius:4px;font-weight:bold;'>
            Score: {score:.0f}/100</span><br><br>
          <b>Status:</b> {row['status']}<br>
          <b>Water depth at site:</b> {row['depth_m']} m<br>
          <b>Exposed population:</b> {row['exposed_pop']:,}<br>
          <b>Travel time:</b> {row['travel_time']} (dry: {row['baseline_time']})<br><br>
          <b>Why:</b> {row['why']}<br><br>
          <b>Action:</b> {row['action']}
        </div>"""

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color, fill=True, fill_color=color, fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row['name']} | Score: {score:.0f}",
        ).add_to(asset_group)
    asset_group.add_to(m)

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:rgba(15,23,42,0.92);color:white;padding:12px 16px;
                border-radius:10px;font-family:sans-serif;font-size:12px;
                border:1px solid rgba(255,255,255,0.15);">
      <b>Risk Score</b><br>
      🔴 High (≥50)&nbsp;&nbsp;🟠 Medium (25-49)&nbsp;&nbsp;🟢 Low (&lt;25)<br><br>
      <b>Road Status</b><br>
      <span style="color:#22c55e">▬</span> Open &nbsp;
      <span style="color:#f59e0b">▬</span> Slow &nbsp;
      <span style="color:#ef4444">▬</span> Closed<br><br>
      <b>Flood Depth</b><br>
      <span style="background:#bde0fe;padding:0 8px">&nbsp;</span> &lt;0.5m &nbsp;
      <span style="background:#5390d9;padding:0 8px">&nbsp;</span> 0.5-1.5m &nbsp;
      <span style="background:#1d3d70;padding:0 8px">&nbsp;</span> &gt;1.5m
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ─────────────────────────────────────────────────────────────────────────────
# 11. DEPENDENCY GRAPH (Plotly)
# ─────────────────────────────────────────────────────────────────────────────
def make_dep_graph(assets_df, dep_graph, results_df):
    """Build a Plotly figure for the dependency network."""
    status_color = {
        "Direct Failure":  "#ef4444",
        "Cascade Failure": "#f97316",
        "Cut Off":         "#a855f7",
        "Degraded Access": "#f59e0b",
        "Operational":     "#22c55e",
    }
    score_map = dict(zip(results_df["name"], results_df["status"]))

    # Node positions: use lat/lon directly
    lat_min = assets_df["lat"].min()
    lat_max = assets_df["lat"].max()
    lon_min = assets_df["lon"].min()
    lon_max = assets_df["lon"].max()

    def norm(v, vmin, vmax):
        return (v - vmin) / (vmax - vmin + 1e-9)

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    idx_to_pos = {}
    for idx, row in assets_df.iterrows():
        x = norm(row["lon"], lon_min, lon_max)
        y = norm(row["lat"], lat_min, lat_max)
        idx_to_pos[idx] = (x, y)
        node_x.append(x)
        node_y.append(y)
        status = score_map.get(row["name"], "Operational")
        node_color.append(status_color.get(status, "#22c55e"))
        crit = CRITICALITY.get(row["type"], 0.5)
        node_size.append(20 + int(crit * 20))
        node_text.append(f"{row['name']}<br>({row['type']})<br>Status: {status}")

    # Edges
    power_ex, power_ey = [], []
    water_ex, water_ey = [], []
    for u, v, data in dep_graph.edges(data=True):
        x0, y0 = idx_to_pos[u]
        x1, y1 = idx_to_pos[v]
        if data.get("dep_type") == "power":
            power_ex += [x0, x1, None]
            power_ey += [y0, y1, None]
        else:
            water_ex += [x0, x1, None]
            water_ey += [y0, y1, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=power_ex, y=power_ey, mode="lines",
        line=dict(color="#f59e0b", width=2, dash="dot"),
        name="⚡ Power dependency (ASSUMED)",
        hoverinfo="none"))
    fig.add_trace(go.Scatter(
        x=water_ex, y=water_ey, mode="lines",
        line=dict(color="#60a5fa", width=2, dash="dash"),
        name="💧 Water dependency (ASSUMED)",
        hoverinfo="none"))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(color=node_color, size=node_size,
                    line=dict(color="white", width=1.5)),
        text=[a["name"].split()[0] for _, a in assets_df.iterrows()],
        textposition="top center",
        hovertext=node_text, hoverinfo="text",
        name="Assets"))

    fig.update_layout(
        title=dict(text="⚠️ ASSUMED Dependency Topology (nearest-source rule)",
                   font=dict(color="#f59e0b", size=14)),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="white"),
        showlegend=True,
        legend=dict(bgcolor="rgba(30,41,59,0.8)", bordercolor="#334155"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 12.  MAIN STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────
def validate_csv(df):
    """Validate user-uploaded CSV. Returns (df_clean, error_msg)."""
    required = {"name", "type", "lat", "lon"}
    missing = required - set(df.columns.str.strip().str.lower())
    if missing:
        return None, f"Missing columns: {', '.join(missing)}"
    df.columns = df.columns.str.strip().str.lower()
    df = df[["name", "type", "lat", "lon"]].copy()
    # Guard: empty CSV before .str accessor (Pandas infers float for empty col)
    if df.empty:
        return None, "CSV contains no data rows."
    # Cast to str to handle any Pandas type inference artefacts
    df["type"] = df["type"].astype(str).str.strip().str.lower()
    invalid_types = set(df["type"].dropna().unique()) - set(ASSET_TYPES)
    if invalid_types:
        return None, (f"Unknown asset types: {', '.join(sorted(invalid_types))}. "
                      f"Allowed: {', '.join(ASSET_TYPES)}")
    try:
        df["lat"] = pd.to_numeric(df["lat"])
        df["lon"] = pd.to_numeric(df["lon"])
    except Exception:
        return None, "lat/lon columns must contain numeric values."
    lat_range = (10.5, 12.0)
    lon_range = (78.0, 80.0)
    out = df[(df["lat"] < lat_range[0]) | (df["lat"] > lat_range[1]) |
             (df["lon"] < lon_range[0]) | (df["lon"] > lon_range[1])]
    if not out.empty:
        names = ", ".join(out["name"].tolist()[:3])
        st.warning(f"⚠️ Some assets appear outside the study area: {names}. "
                   "Results may be unreliable for those assets.")
    return df.reset_index(drop=True), None


def main():
    st.set_page_config(
        page_title="FloodNet Impact Mapper | INTELLATHON-26",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Global dark theme injection
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #0f172a; color: #e2e8f0; }
    .stApp { background: #0f172a; }
    section[data-testid="stSidebar"] { background: #1e293b; border-right: 1px solid #334155; }
    .kpi-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-2px); }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #60a5fa; line-height: 1.2; }
    .kpi-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-value.danger { color: #f87171; }
    .kpi-value.warning { color: #fbbf24; }
    .kpi-value.safe { color: #4ade80; }
    .warning-banner {
        background: linear-gradient(90deg, #451a03, #7c2d12);
        border: 1px solid #f97316;
        border-radius: 10px;
        padding: 12px 20px;
        margin-bottom: 20px;
        font-size: 0.85rem;
        color: #fed7aa;
    }
    .score-bar-outer {
        background: #334155; border-radius: 6px; height: 8px; overflow: hidden;
    }
    .score-bar-inner {
        height: 8px; border-radius: 6px;
        background: linear-gradient(90deg, #22c55e, #f59e0b, #ef4444);
    }
    h1, h2, h3 { color: #e2e8f0 !important; }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #60a5fa !important; border-bottom-color: #60a5fa !important; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

    # ─── Load base data (cached) ───────────────────────────────────────────
    dem, pop_grid = build_terrain()
    G, node_pos, hub_node = build_roads()
    default_assets = demo_assets()

    # ─── SIDEBAR ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🌊 FloodNet Mapper")
        st.markdown("**INTELLATHON-26 · Team CODEVERSE**")
        st.divider()

        # Scenario preset
        preset_key = st.selectbox("📋 Scenario Preset", list(PRESETS.keys()), index=0)
        preset_val = PRESETS[preset_key]

        water_rise = st.slider(
            "🌡️ Water Rise (m)",
            min_value=0.0, max_value=6.0,
            value=preset_val if preset_val is not None else 2.0,
            step=0.1,
            help="Height above river base level (simulated bathtub model)",
        )

        closure_depth = st.slider(
            "🚧 Road Closure Depth (m)",
            min_value=0.2, max_value=2.0,
            value=0.5, step=0.1,
            help="Flood depth at which a road segment becomes impassable",
        )

        st.divider()
        st.markdown("### ⚖️ Risk Weight Sliders")
        st.caption("Weights auto-normalise to 100%")
        w_hazard    = st.slider("💧 Hazard (depth)",     0, 100, 25)
        w_crit      = st.slider("🏥 Criticality (type)", 0, 100, 25)
        w_pop       = st.slider("👥 Population",         0, 100, 20)
        w_access    = st.slider("🛣️ Access loss",        0, 100, 20)
        w_cascade   = st.slider("⚡ Cascade failure",    0, 100, 10)
        weights = dict(hazard=w_hazard, criticality=w_crit, population=w_pop,
                       access=w_access, cascade=w_cascade)

        total_w = sum(weights.values())
        if total_w > 0:
            pcts = {k: round(v / total_w * 100) for k, v in weights.items()}
            st.caption(f"Effective: H={pcts['hazard']}% C={pcts['criticality']}% "
                       f"P={pcts['population']}% A={pcts['access']}% X={pcts['cascade']}%")
        else:
            st.warning("⚠️ All weights are zero — using equal weighting.")
            weights = {k: 20 for k in weights}

        st.divider()
        st.markdown("### 📂 Upload Custom Assets")
        uploaded = st.file_uploader(
            "CSV: name, type, lat, lon", type=["csv"],
            help="Allowed types: hospital, shelter, school, substation, water_plant",
        )

        assets_df = default_assets.copy()
        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded)
                cleaned, err = validate_csv(raw)
                if err:
                    st.error(f"❌ CSV Error: {err}")
                else:
                    assets_df = cleaned
                    st.success(f"✅ Loaded {len(assets_df)} assets from CSV")
            except Exception as e:
                st.error(f"❌ Failed to read CSV: {e}")

        st.divider()
        st.caption("📌 Study area: Perambalur, Tamil Nadu, India")
        st.caption("⚠️ All data is SIMULATED — not for operational use")

    # ─── Simulated warning banner ──────────────────────────────────────────
    st.markdown("""
    <div class='warning-banner'>
    ⚠️ <strong>SIMULATED PROTOTYPE — INTELLATHON-26</strong> &nbsp;|&nbsp;
    All terrain (synthetic DEM), population (modelled grid), flood scenarios (bathtub model),
    road network (9×9 lattice), dependency links (nearest-source heuristic), and asset locations
    are <strong>SIMULATED</strong> and do <strong>NOT</strong> represent real conditions.
    <strong>Not for operational emergency use.</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("# 🌊 FloodNet Impact Mapper")
    st.markdown("*GIS Decision-Support for Flood-Impacted Critical Infrastructure*")

    # ─── COMPUTE (cached where possible) ──────────────────────────────────
    flood_depth, flood_mask = compute_flood(dem, water_rise)
    site_depth, exposed_pop = compute_exposure(assets_df, flood_depth, pop_grid)
    edge_status, travel_times, baseline_times, cut_off_set, links_cut = compute_road_access(
        G, node_pos, hub_node, flood_depth, closure_depth, assets_df)
    dep_graph = build_dependency_graph(assets_df)
    direct_failed = {
        i for i, row in assets_df.iterrows()
        if site_depth[i] > FAIL_DEPTH.get(row["type"], 0.5)
    }
    cascade_frac = compute_cascade(assets_df, dep_graph, direct_failed)
    cascade_failed = sum(
        1 for i in assets_df.index
        if i not in direct_failed and cascade_frac.get(i, 0) > 0
    )
    results_df = compute_scores(assets_df, site_depth, exposed_pop, travel_times,
                                baseline_times, cut_off_set, cascade_frac, weights)

    # KPIs
    flooded_km2 = float(flood_mask.sum()) * CELL_KM ** 2
    total_flooded_pop = int(pop_grid[flood_mask].sum())
    n_cut = len(cut_off_set)
    n_direct = len(direct_failed)
    n_cascade = cascade_failed

    # ─── KPI METRICS ──────────────────────────────────────────────────────
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    kpis = [
        (col1, f"{water_rise:.1f} m", "Water Rise", ""),
        (col2, f"{flooded_km2:.1f} km²", "Flooded Area", "danger" if flooded_km2 > 5 else "warning" if flooded_km2 > 1 else "safe"),
        (col3, f"{total_flooded_pop:,}", "People in Flood Zone", "danger" if total_flooded_pop > 5000 else "warning" if total_flooded_pop > 1000 else "safe"),
        (col4, f"{n_direct}D / {n_cascade}C", "Assets Failed (D/C)", "danger" if n_direct > 3 else "warning" if n_direct > 0 else "safe"),
        (col5, f"{n_cut}", "Assets Cut Off", "danger" if n_cut > 3 else "warning" if n_cut > 0 else "safe"),
        (col6, f"{links_cut}", "Road Links Cut", "danger" if links_cut > 20 else "warning" if links_cut > 5 else "safe"),
    ]
    for col, val, label, cls in kpis:
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
              <div class='kpi-value {cls}'>{val}</div>
              <div class='kpi-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── TABS ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Impact Map",
        "📊 Intervention Ranking",
        "🕸️ Dependencies",
        "📈 Scenario Sweep",
        "📖 Methodology",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1: IMPACT MAP
    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("### 🗺️ Flood Impact Map")
        st.caption("Use the layer control (top-right) to toggle flood depth, roads, and assets. Click markers for details.")

        fmap = make_folium_map(
            dem, flood_depth, flood_mask, results_df, assets_df,
            edge_status, node_pos, water_rise,
        )
        map_html = fmap._repr_html_()
        components.html(map_html, height=600, scrolling=False)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2: INTERVENTION RANKING
    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### 📊 Intervention Priority Ranking")
        st.caption("Sorted by Impact Risk Score. Higher score = higher intervention priority.")

        # Ranked table with score bars
        for rank, row in results_df.iterrows():
            score = row["score"]
            score_color = "#ef4444" if score >= 50 else "#f59e0b" if score >= 25 else "#22c55e"
            status_badge = {
                "Direct Failure":  "🔴",
                "Cascade Failure": "🟠",
                "Cut Off":         "🟣",
                "Degraded Access": "🟡",
                "Operational":     "🟢",
            }.get(row["status"], "⚪")

            with st.expander(
                f"#{rank+1}  {row['name']}  |  Score: {score:.0f}/100  {status_badge} {row['status']}",
                expanded=(rank < 3),
            ):
                c1, c2, c3 = st.columns([2, 2, 3])
                with c1:
                    st.metric("Risk Score", f"{score:.0f}/100")
                    st.metric("Flood Depth", f"{row['depth_m']} m")
                    st.metric("Exposed Pop.", f"{row['exposed_pop']:,}")
                with c2:
                    st.metric("Travel Time", row["travel_time"])
                    st.metric("Dry Baseline", row["baseline_time"])
                    st.metric("Asset Type", row["type"].replace("_", " ").title())
                with c3:
                    st.markdown(f"**Why:** {row['why']}")
                    st.markdown(f"**Action:** {row['action']}")

                # Factor breakdown bar
                factors = {
                    "Hazard": row["f_hazard"],
                    "Criticality": row["f_criticality"],
                    "Population": row["f_population"],
                    "Access": row["f_access"],
                    "Cascade": row["f_cascade"],
                }
                fig_bar = go.Figure(go.Bar(
                    x=list(factors.keys()),
                    y=list(factors.values()),
                    marker_color=["#60a5fa", "#a78bfa", "#34d399", "#f59e0b", "#f87171"],
                    text=[f"{v:.2f}" for v in factors.values()],
                    textposition="outside",
                ))
                fig_bar.update_layout(
                    paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
                    font=dict(color="white", size=11),
                    yaxis=dict(range=[0, 1.1], title="Normalised Factor (0-1)"),
                    xaxis_title="Factor",
                    margin=dict(t=20, b=20, l=10, r=10),
                    height=220,
                    showlegend=False,
                )
                st.plotly_chart(
                    fig_bar,
                    use_container_width=True,
                    key=f"intervention_factor_chart_{rank}",
                )

        st.divider()
        st.markdown("#### ⬇️ Download Results")
        dl1, dl2 = st.columns(2)
        with dl1:
            csv_out = results_df.drop(columns=["idx"], errors="ignore").to_csv(index=False)
            st.download_button("📥 Download CSV", csv_out,
                               file_name="floodnet_results.csv", mime="text/csv")
        with dl2:
            features = []
            for _, row in results_df.iterrows():
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [row["lon"], row["lat"]]},
                    "properties": {k: v for k, v in row.items() if k not in ("lat", "lon")},
                })
            geojson = json.dumps({"type": "FeatureCollection", "features": features}, indent=2)
            st.download_button("📥 Download GeoJSON", geojson,
                               file_name="floodnet_results.geojson", mime="application/json")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3: DEPENDENCY GRAPH
    # ══════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### 🕸️ Critical Infrastructure Dependency Network")
        st.warning("⚠️ Dependency topology is ASSUMED using a nearest-source heuristic. Not based on real utility network data.")

        fig_dep = make_dep_graph(assets_df, dep_graph, results_df)
        st.plotly_chart(
            fig_dep,
            use_container_width=True,
            key="dependency_graph_chart",
        )

        # Legend table
        st.markdown("#### Status Legend")
        st.markdown("""
        | Symbol | Status | Meaning |
        |--------|--------|---------|
        | 🔴 | Direct Failure | Site flooded above failure threshold |
        | 🟠 | Cascade Failure | Upstream dependency (power/water) failed |
        | 🟣 | Cut Off | No passable road from emergency hub |
        | 🟡 | Degraded Access | Accessible but significantly slower |
        | 🟢 | Operational | No significant impact detected |
        """)

        # Show dependency edges as a table
        st.markdown("#### ASSUMED Dependency Edges")
        dep_rows = []
        for u, v, data in dep_graph.edges(data=True):
            dep_rows.append({
                "Source": assets_df.loc[u, "name"],
                "Source Type": assets_df.loc[u, "type"],
                "Target": assets_df.loc[v, "name"],
                "Dependency": data.get("dep_type", "?"),
                "Source Status": results_df[results_df["name"] == assets_df.loc[u, "name"]]["status"].values[0]
                                 if not results_df[results_df["name"] == assets_df.loc[u, "name"]].empty else "?",
            })
        st.dataframe(pd.DataFrame(dep_rows), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4: SCENARIO SWEEP
    # ══════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("### 📈 Scenario Sweep: Tipping Points Analysis")
        st.caption("Simulated outcomes across all water-rise levels (0.5–6.0 m). Current scenario highlighted.")

        sweep_df = scenario_sweep(
            dem, pop_grid, G, node_pos, hub_node,
            closure_depth, assets_df,
        )

        fig_sweep = go.Figure()
        traces = [
            ("direct_failures", "#ef4444", "🔴 Direct Failures"),
            ("cascade_failures", "#f97316", "🟠 Cascade Failures"),
            ("cut_off",          "#a855f7", "🟣 Cut-Off Assets"),
            ("links_cut",        "#f59e0b", "🟡 Road Links Cut"),
        ]
        for col, color, name in traces:
            fig_sweep.add_trace(go.Scatter(
                x=sweep_df["water_rise"], y=sweep_df[col],
                mode="lines+markers", name=name,
                line=dict(color=color, width=2.5),
                marker=dict(size=7),
            ))

        # Vertical line at current scenario
        fig_sweep.add_vline(
            x=water_rise, line_dash="dash", line_color="#60a5fa",
            annotation_text=f"Current: {water_rise}m",
            annotation_font_color="#60a5fa",
        )

        fig_sweep.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            font=dict(color="white"),
            xaxis=dict(title="Water Rise (m)", gridcolor="#334155"),
            yaxis=dict(title="Count", gridcolor="#334155"),
            legend=dict(bgcolor="rgba(30,41,59,0.8)", bordercolor="#334155"),
            height=450,
        )
        st.plotly_chart(
            fig_sweep,
            use_container_width=True,
            key="scenario_sweep_chart",
        )

        st.markdown("#### Tipping Point Table")
        st.dataframe(sweep_df.rename(columns={
            "water_rise": "Rise (m)",
            "direct_failures": "Direct Failures",
            "cascade_failures": "Cascade Failures",
            "cut_off": "Cut Off",
            "links_cut": "Links Cut",
        }), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 5: METHODOLOGY
    # ══════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown("### 📖 Methodology & Data Sources")
        st.markdown("""
## 🌊 FloodNet Impact Mapper — Technical Overview

> **PROTOTYPE ONLY** — All data is simulated. Not validated for operational use.

---

### Pipeline

```
Synthetic DEM  ──┐
                 ├── Flood Model (Bathtub + BFS) ──┐
Scenario Preset──┘                                 │
                                                   ├── Exposure ──┐
Synthetic Pop. ────────────────────────────────────┘             │
                                                                  │
Road Lattice ── Road Access Model (NetworkX Dijkstra) ───────────┼── Impact Risk Score (0–100)
                                                                  │
Dependency Graph (nearest-source heuristic) ─────────────────────┘
                                                                  │
                                                        Ranked Intervention List
```

---

### What Is Simulated

| Component | Simulation Method | Real-World Replacement |
|-----------|------------------|----------------------|
| **Terrain (DEM)** | Gaussian river valley + smooth noise | SRTM 30 m / Bhuvan DEM / Survey of India |
| **Population** | Gaussian clusters around town centre | WorldPop 100 m / Census ward data |
| **Flood model** | Bathtub fill + BFS connectivity | HEC-RAS 2D / LISFLOOD-FP / IITB-CYRIL |
| **Road network** | 9×9 regular lattice | OSMnx real road graph |
| **Asset locations** | Manually placed, representative | OSM + health facility GIS databases |
| **Dependency graph** | Nearest-source heuristic | Utility operator SCADA / network GIS |
| **Bridge flooding** | Lattice edge depth threshold | Bridge deck elevation survey |

---

### Impact Risk Score Formula

```
Score = 100 × Σ(wᵢ × fᵢ)  where Σwᵢ = 1

f₁ (Hazard)      = min(depth_m / 2.0,  1.0)
f₂ (Criticality) = lookup by asset type (hospital=1.0, water_plant=0.9, ...)
f₃ (Population)  = exposed_pop / max_exposed_pop
f₄ (Access)      = min((t_flood - t_dry) / (5 × t_dry), 1.0)   [1.0 if cut off]
f₅ (Cascade)     = failed_upstream_deps / total_upstream_deps
```

---

### Road Access Model
- **Base speed:** 40 km/h (dry)
- **Speed degradation:** linear 40 → 5 km/h as depth goes 0 → closure threshold
- **Closure:** edge removed from graph if depth ≥ closure threshold slider
- **Routing:** NetworkX Dijkstra shortest path (weighted by travel time minutes)
- **Hub:** top-left node of the lattice (representing high-ground emergency base)

---

### Where to Plug In Real Data

1. **`build_terrain()`** → load GeoTIFF DEM + WorldPop raster
2. **`build_roads()`** → replace with `osmnx.graph_from_place()`
3. **`demo_assets()`** → query OSM / NRHM facility database
4. Dependency graph → import utility network GIS topology
5. Flood model → call HEC-RAS output raster

---

### References
- Bathtub flood model: Tate et al. (2015), *Flood Exposure Assessment*
- Dependency analysis: Rinaldi & Peerenboom (2001), *Critical Infrastructure Interdependencies*
- WorldPop: worldpop.org
- OSM: openstreetmap.org

---
*Built for INTELLATHON-26 · Team CODEVERSE*
""")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
