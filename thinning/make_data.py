import json
import os
import random

import cv2
import geopandas as gpd
import numpy as np
import osmnx as ox
import rasterio
from PIL import Image
from osmnx.projection import project_gdf
from rasterio.features import rasterize
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString, mapping
from tqdm import tqdm

# Keep parameters bundled
CONFIG = 'oxford-town'

CONFIGS = {
    "oxford-town": {
        "place": "Oxford, Ohio, USA",
        "image_size": (256, 256),
        "n_samples": 500,
        "network_type": "drive",
        'data_dir': '../data/thinning',
        # Selected road types to include
        'road_types': [
            'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
            'unclassified', 'residential', 'motorway_link', 'trunk_link',
            'primary_link', 'secondary_link', 'tertiary_link'
        ],
        # Distortion parameters
        'distortion': {
            'blur_prob': 0.7,
            'blur_radius': (0.5, 2.0),
            'noise_prob': 0.8,
            'noise_level': (5, 25),
            'dropout_prob': 0.4,
            'dropout_amount': (0.01, 0.05)
        }
    },
}

for c in CONFIGS:
    CONFIGS[c]['name'] = c

# Default thickness by road type (in meters)
# formatted as (lower, upper)
DEFAULT_THICKNESS = {
    'motorway': (20, 35),
    'trunk': (15, 30),
    'primary': (10, 25),
    'secondary': (7, 15),
    'tertiary': (5, 10),
    'unclassified': (3, 7),
    'residential': (4, 8),
    'living_street': (3, 6),
    'service': (2.5, 6),
    'pedestrian': (2, 8),
    'track': (2, 4),
    'footway': (1, 3),
    'cycleway': (1.5, 4),
    'bridleway': (1.5, 3),
    'steps': (1, 3),
    'path': (0.5, 2.5),
    'motorway_link': (7, 15),
    'trunk_link': (6, 12),
    'primary_link': (6, 12),
    'secondary_link': (5, 10),
    'tertiary_link': (4, 8),
    'bus_guideway': (6, 8),
    'raceway': (10, 20),
    'road': (3, 8),
    'busway': (6, 8),
    'corridor': (1, 3),
    'via_ferrata': (0.5, 1.5),
    'sidewalk': (1, 3),
    'crossing': (2, 6),
    'traffic_island': (1, 3),
}


def get_default_thickness(row):
    tag = row.get('highway', 'residential')
    if isinstance(tag, list):
        tag = tag[0]  # pick the first tag if it's a list

    # Base thickness range from the default mapping
    lower, upper = DEFAULT_THICKNESS.get(tag, (2, 4))

    # Adjust thickness based on attributes if available
    lanes = row.get('lanes')
    if lanes and str(lanes).isdigit():
        # Add 1-2 meters per lane
        lane_count = int(lanes)
        lower += (lane_count - 1) * 1.0
        upper += (lane_count - 1) * 2.0

    # Adjust for width if explicitly specified
    width = row.get('width')
    if width and str(width).replace('.', '', 1).isdigit():
        # Use the specified width with some variation
        width_val = float(width)
        return random.uniform(width_val * 0.8, width_val * 1.2)

    # Adjust for importance
    if tag in ['motorway', 'trunk', 'primary']:
        # Important roads might be wider
        lower *= 1.2
        upper *= 1.2

    return random.uniform(lower, upper)


def download_osm_data(config, cache_dir="./data/cache"):
    config_name = config.get("name", "default")
    place = config["place"]
    network_type = config.get("network_type", "all")
    road_types = config.get("road_types", None)

    config_cache_dir = os.path.join(cache_dir, config_name)
    os.makedirs(config_cache_dir, exist_ok=True)

    # Create a unique cache filename based on config parameters
    cache_suffix = f"_{'-'.join(road_types)}" if road_types else ""
    edges_fp = os.path.join(config_cache_dir, f"edges{cache_suffix}.gpkg")
    nodes_fp = os.path.join(config_cache_dir, f"nodes{cache_suffix}.gpkg")

    if os.path.exists(edges_fp) and os.path.exists(nodes_fp):
        print(f"Loading cached OSM data for {place}...")
        gdf_edges = gpd.read_file(edges_fp)
        gdf_nodes = gpd.read_file(nodes_fp)
    else:
        print(f"Fetching roads from {place}...")
        if road_types:
            print(f"Filtering for road types: {', '.join(road_types)}")
            custom_filter = f"['highway'~'^({'|'.join(road_types)})$']"
            G = ox.graph_from_place(place, network_type=network_type, custom_filter=custom_filter)
        else:
            G = ox.graph_from_place(place, network_type=network_type)

        print("Converting to dataframes...")
        gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        gdf_nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)

        print("Projecting to UTM ...")
        gdf_edges = project_gdf(gdf_edges)  # Automatically choose a UTM zone
        gdf_nodes = project_gdf(gdf_nodes)

        print(f"Fetched {len(gdf_edges)} edges and {len(gdf_nodes)} nodes.")
        gdf_edges.to_file(edges_fp, driver="GPKG")
        gdf_nodes.to_file(nodes_fp, driver="GPKG")

    return gdf_edges, gdf_nodes


def pick_random_intersections(nodes, n=500):
    return nodes.sample(n if n < len(nodes) else len(nodes))


def apply_distortions(image, config):
    """
    Apply realistic distortions to the image based on configuration parameters.
    """
    distortion = config.get('distortion', {})
    img_array = np.array(image)

    # Apply Gaussian blur
    if random.random() < distortion.get('blur_prob', 0.5):
        blur_radius = random.uniform(*distortion.get('blur_radius', (0.5, 2.0)))
        img_array = gaussian_filter(img_array, sigma=blur_radius)

    # Add noise
    if random.random() < distortion.get('noise_prob', 0.5):
        noise_level = random.randint(*distortion.get('noise_level', (5, 20)))
        noise = np.random.normal(0, noise_level, img_array.shape).astype(np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Random dropouts (simulate occlusions or data gaps)
    if random.random() < distortion.get('dropout_prob', 0.3):
        dropout_amount = random.uniform(*distortion.get('dropout_amount', (0.01, 0.05)))
        mask = np.random.random(img_array.shape) > dropout_amount
        img_array = img_array * mask

    # Apply slight perspective distortion
    if random.random() < distortion.get('perspective_prob', 0.3):
        h, w = img_array.shape
        # Create slight perspective change
        src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        # Random displacement of corners (up to 5% of image size)
        max_disp = int(min(h, w) * 0.05)
        dst_points = src_points + np.random.randint(-max_disp, max_disp + 1, src_points.shape)
        # Apply perspective transform
        M = cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points.astype(np.float32))
        img_array = cv2.warpPerspective(img_array, M, (w, h))

    return Image.fromarray(img_array)


def rasterize_roads(roads, bounds, size, thickness_fn, selected_road_types=None):
    """
    Rasterize roads into a binary mask using rasterio.
    Thickness is applied by buffering each LineString in meters.
    """
    # Create transform (affine mapping from pixel coords to geographic coords)
    transform = rasterio.transform.from_bounds(*bounds, width=size[0], height=size[1])

    # Buffer each line to simulate thickness and generate (geometry, value) tuples
    shapes = []
    for _, row in roads.iterrows():
        geom = row.geometry
        if geom is None or not isinstance(geom, LineString):
            continue

        # Filter by road type if specified
        if selected_road_types:
            tag = row.get('highway', 'residential')
            if isinstance(tag, list):
                tag = tag[0]
            if tag not in selected_road_types:
                continue

        thickness = thickness_fn(row) / 2.0  # buffer radius in meters
        if thickness > 0:
            buffered = geom.buffer(thickness)
            shapes.append((buffered, 255))  # Burn value = 255 for road
        else:
            shapes.append((geom, 255))

    mask = rasterize(
        shapes=shapes,
        out_shape=size,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    return mask


def create_one_pixel_skeleton(roads, bounds, size, selected_road_types=None):
    """
    Create a one-pixel-width ground truth skeleton from the road network.
    """
    # Create transform (affine mapping from pixel coords to geographic coords)
    transform = rasterio.transform.from_bounds(*bounds, width=size[0], height=size[1])

    # Create shapes without buffering for one-pixel lines
    shapes = []
    for _, row in roads.iterrows():
        geom = row.geometry
        if geom is None or not isinstance(geom, LineString):
            continue

        # Filter by road type if specified
        if selected_road_types:
            tag = row.get('highway', 'residential')
            if isinstance(tag, list):
                tag = tag[0]
            if tag not in selected_road_types:
                continue

        shapes.append((geom, 255))  # Burn value = 255 for road

    mask = rasterize(
        shapes=shapes,
        out_shape=size,
        transform=transform,
        fill=0,
        all_touched=True,  # Ensure connectivity for one-pixel lines
        dtype=np.uint8
    )
    return mask


def generate_samples(gdf_edges, gdf_nodes, config, n_samples=500, out_dir="./data/thinning", image_size=(256, 256)):
    os.makedirs(out_dir, exist_ok=True)

    # Get configuration parameters
    road_types = config.get("road_types", None)

    print(f"Generating {n_samples} samples...")
    selected_nodes = pick_random_intersections(gdf_nodes, n=n_samples)

    # Make subdirectories for organized output
    input_dir = os.path.join(out_dir, "input")
    gt_dir = os.path.join(out_dir, "ground_truth")
    geojson_dir = os.path.join(out_dir, "geojson")
    metadata_dir = os.path.join(out_dir, "metadata")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(geojson_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    for i, (_, node) in enumerate(tqdm(selected_nodes.iterrows(), total=n_samples)):
        pt = node.geometry
        buffer_m = 128  # meters around the point
        bounds = (pt.x - buffer_m, pt.y - buffer_m, pt.x + buffer_m, pt.y + buffer_m)
        clip = gdf_edges.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]].copy()
        if clip.empty:
            continue

        # Generate the thicker roads with varying thickness based on attributes
        image_array = rasterize_roads(clip, bounds, image_size, get_default_thickness, road_types)

        # Generate the one-pixel-wide ground truth
        target_array = create_one_pixel_skeleton(clip, bounds, image_size, road_types)

        # Convert to PIL Images
        input_image = Image.fromarray(image_array)
        gt_image = Image.fromarray(target_array)

        # Apply distortions to the input image
        input_image = apply_distortions(input_image, config)

        # Save input image with distortions
        input_path = os.path.join(input_dir, f"image_{i:05d}.png")
        input_image.save(input_path)

        # Save ground truth (one-pixel-wide skeleton)
        gt_path = os.path.join(gt_dir, f"target_{i:05d}.png")
        gt_image.save(gt_path)

        # Save geojson for reference
        geojson_path = os.path.join(geojson_dir, f"roads_{i:05d}.geojson")
        features = [{
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": {
                "highway": row.get("highway", "unknown"),
                "lanes": row.get("lanes", None),
                "width": row.get("width", None),
                "name": row.get("name", None)
            }
        } for _, row in clip.iterrows() if isinstance((geom := row.geometry), LineString)]

        geojson = {"type": "FeatureCollection", "features": features}
        with open(geojson_path, 'w') as f:
            json.dump(geojson, f)

        # Create a metadata file with sample information
        metadata_path = os.path.join(metadata_dir, f"metadata_{i:05d}.json")
        metadata = {
            "sample_id": i,
            "location": {"x": pt.x, "y": pt.y},
            "bounds": bounds,
            "road_count": len(clip),
            "distortions_applied": config.get("distortion", {})
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)


if __name__ == "__main__":
    config = CONFIGS[CONFIG]
    print(f"Using configuration: {CONFIG}")

    # Make sure data directories exist
    os.makedirs(config.setdefault('data_dir', '../data/thinning'), exist_ok=True)

    # Download OSM data with road type filtering
    edges, nodes = download_osm_data(config, cache_dir="../data/cache")

    # Generate samples with all new features
    generate_samples(
        edges,
        nodes,
        config=config,
        n_samples=config['n_samples'],
        out_dir=config.setdefault('data_dir', '../data/thinning'),
        image_size=config['image_size']
    )

    print(f"Data generation complete! Generated samples are stored in: {config['data_dir']}")
