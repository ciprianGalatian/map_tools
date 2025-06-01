# Copyright 2025 Ciprian Dumitrache and Andreea Stratulat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np
import yaml
from pathlib import Path
from PIL import Image
import matplotlib.cm as cm
import random

class Measurements:
    def __init__(
        self,
        zones_yaml_path: str,
        heatmap_yaml_path: str,
        measurement_fn=None
    ):
        """
        :param zones_yaml_path: path to zones_waypoints.yaml
        :param heatmap_yaml_path:  path where measurements.yaml will be saved
        :param measurement_fn:  optional function zone_id -> measurements_value
                               default: random.uniform(1,10)
        """
        self.zones_path = Path(zones_yaml_path)
        self.heatmap_yaml_path   = Path(heatmap_yaml_path)
        # user-provided measurement function, or default:
        self.measure_zone = measurement_fn or self._default_measurement
        self.measurements_values = {}  # zone_id -> measured value

    def _default_measurement(self, zone_id: int) -> float:
        """Default measurement reading: random float in [1,10)."""
        return random.uniform(1.0, 10.0)

    def load_zones(self) -> list[int]:
        """Reads zones_waypoints.yaml and returns a list of zone IDs."""
        with open(self.zones_path, 'r') as f:
            data = yaml.safe_load(f)
        waypoints = data.get('waypoints', [])
        return [wp['zone'] for wp in waypoints if 'zone' in wp]

    def measure_all(self):
        """Populates self.measurements_values by measuring each zone."""
        zones = self.load_zones()
        self.measurements_values = {}
        for zone_id in zones:
            self.measurements_values[zone_id] = float(self.measure_zone(zone_id))
        print(f"Measured values for {len(self.measurements_values)} zones")

    def save_measurements_yaml(self):
        """
        Writes self.measurements_values into YAML:
        {'measurements': [ {'zone': zone, 'power': power}, ... ]}
        """
        entries = [
            {'zone': zone, 'power': power}
            for zone, power in sorted(self.measurements_values.items())
        ]
        with open(self.heatmap_yaml_path, 'w') as f:
            yaml.dump({'measurements': entries}, f)
        print(f"Saved {len(entries)} entries to {self.heatmap_yaml_path}")

def convert_pgm_to_png(
    pgm_path: str,
    output_path: str,
    scale: float = 1.0,
    resample=Image.NEAREST
) -> bool:
    """
    Convert a .pgm to .png and optionally upscale it.
    """
    try:
        img = Image.open(pgm_path)
        if scale != 1.0:
            orig_w, orig_h = img.size
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            img = img.resize((new_w, new_h), resample=resample)
            print(f"Resized from {orig_w}×{orig_h} to {new_w}×{new_h}")
        img.save(output_path)
        print(f"Converted and saved PNG at {output_path}")
        return True
    except Exception as e:
        print(f"Error converting/resizing image: {e}")
        return False


def load_map_metadata(yaml_path: str):
    """
    Load origin and resolution from a map.yaml file.
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    origin = tuple(data.get('origin', [0.0, 0.0, 0.0]))
    resolution = float(data.get('resolution', 0.05))
    return origin, resolution


def load_zone_centroids(yaml_path: str):
    """
    Extract zone centroids from a waypoint-style YAML file.
    Returns dict: zone -> [x, y]
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    centroids = {}
    for wp in data.get('waypoints', []):
        z = wp.get('zone'); x = wp.get('x'); y = wp.get('y')
        if z is not None and x is not None and y is not None:
            centroids[z] = [x, y]
    return centroids


def load_measurements(yaml_path: str):
    """
    Load measurement per zone from YAML.
    Returns dict: zone -> power
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    measurements_list = data.get('measurements', [])
    #return {e['zone']: e['power'] for e in measurements_list}
    return {e['zone']: (e['power'] if e['power'] is not None else np.nan) for e in measurements_list}


def generate_zones(
    img: np.ndarray,
    zone_size: int,
    free_threshold: int,
    resolution: float,
    origin: tuple,
    obstacle_buffer: float = 0.0,
    avoid_near_obstacles: bool = False
):
    """
    Partition the occupancy image into free-space zones.
    
    If `avoid_near_obstacles` is True, zones within 50 cm of any obstacle
    (black or gray pixel) are excluded. Otherwise, only free space check is applied.

    Returns:
        zones: List of (x0, y0, x1, y1) pixel boxes
        cents: List of (cx, cy) zone centroids in world coordinates
    """
    h, w = img.shape
    margin_px = int(obstacle_buffer / resolution) if avoid_near_obstacles else 0
    zones, cents = [], []

    for y0 in range(0, h, zone_size):
        for x0 in range(0, w, zone_size):
            y1 = min(y0 + zone_size, h)
            x1 = min(x0 + zone_size, w)
            patch = img[y0:y1, x0:x1]

            # Only proceed if the patch contains free space
            if np.any(patch >= free_threshold):
                ys, xs = np.where(patch >= free_threshold)
                x0n, x1n = x0 + xs.min(), x0 + xs.max() + 1
                y0n, y1n = y0 + ys.min(), y0 + ys.max() + 1

                if avoid_near_obstacles:
                    check_x0 = max(0, x0n - margin_px)
                    check_y0 = max(0, y0n - margin_px)
                    check_x1 = min(w, x1n + margin_px)
                    check_y1 = min(h, y1n + margin_px)
                    danger_patch = img[check_y0:check_y1, check_x0:check_x1]

                    # Skip zone if any pixel in margin is not free
                    if np.any(danger_patch < free_threshold):
                        continue

                # Compute world-space centroid
                cx_px = int(xs.mean()) + x0
                cy_px = int(ys.mean()) + y0
                cx_w = cx_px * resolution + origin[0]
                cy_w = (h - cy_px) * resolution + origin[1]
                zones.append((x0n, y0n, x1n, y1n))
                cents.append((cx_w, cy_w))

    return zones, cents



def generate_heatmap(
    orig_img, measurements_dict, centroid_dict, origin, resolution, scale,
    zone_size_factor=15, free_threshold=250, decay=1.0, num_iters=200
):
    """
    Generate a heatmap by clustering the map into zones and diffusing known measurement values.
    """
    # Convert image to grayscale
    gray_map = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY) if orig_img.ndim == 3 else orig_img.copy()
    omega=1.5  # Relaxation factor, must be >1.0 and <2.0
    # Cluster into zones and get centroids
    zone_size = int(zone_size_factor)
    zones, zone_centroids = generate_zones(gray_map, zone_size, free_threshold, resolution, origin)

    # Initialize measurement per zone
    zone_measurements = []
    for (zwx, zwy) in zone_centroids:
        val = np.nan
        zx = int((zwx - origin[0]) / resolution)
        zy = int(orig_img.shape[0] - ((zwy - origin[1]) / resolution))
        for idx, (cx, cy) in centroid_dict.items():
            if idx in measurements_dict and measurements_dict[idx] is not None:
                cx_px = int((cx - origin[0]) * scale / resolution)
                cy_px = int(orig_img.shape[0] - ((cy - origin[1]) * scale / resolution))
                dist = np.linalg.norm([zx - cx_px, zy - cy_px])
                if dist < zone_size * 1.5:
                    val = measurements_dict[idx]
                    break
        zone_measurements.append(val)

    zone_measurements = np.array(zone_measurements, dtype=np.float32)
    known = ~np.isnan(zone_measurements)
    
    # Initialize unknown zones to zero to help diffusion propagate values
    zone_measurements[np.isnan(zone_measurements)] = 0.0

    # Precompute neighbors for each zone
    zone_neighbors = []
    for i, (x0, y0, x1, y1) in enumerate(zones):
        neigh = []
        for j, (xx0, yy0, xx1, yy1) in enumerate(zones):
            if i == j:
                continue
            if not (x1 < xx0 or x0 > xx1 or y1 < yy0 or y0 > yy1):
                neigh.append(j)
        zone_neighbors.append(neigh)
        
    for _ in range(num_iters):
        for i, neighbors in enumerate(zone_neighbors):
            if known[i] or not neighbors:
                continue
            valid_vals = [zone_measurements[j] for j in neighbors if not np.isnan(zone_measurements[j])]
            if valid_vals:
                avg_neighbor = np.mean(valid_vals)
                residual = avg_neighbor - zone_measurements[i]
                zone_measurements[i] += omega * residual


    valid_vals = zone_measurements[~np.isnan(zone_measurements)]
    if valid_vals.size > 0 and valid_vals.max() > 0:
        zone_measurements /= valid_vals.max()
    # Normalize values
    if zone_measurements.max() > 0:
        zone_measurements /= zone_measurements.max()

    # Create zone-based heatmap
    colormap = cm.get_cmap('gist_heat')
    heatmap_colored = np.zeros_like(gray_map, dtype=np.float32)
    for (x0, y0, x1, y1), v in zip(zones, zone_measurements):
        heatmap_colored[y0:y1, x0:x1] = v

    if heatmap_colored.max() > 0:
        norm_map = heatmap_colored / heatmap_colored.max()
    else:
        norm_map = heatmap_colored

    heat_img = (colormap(norm_map)[:, :, :3] * 255).astype(np.uint8)
    heat_img_bgr = cv2.cvtColor(heat_img, cv2.COLOR_RGB2BGR)

    # Blend zones with original image
    base = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR) if orig_img.ndim == 2 else orig_img.copy()
    blended = cv2.addWeighted(base, 0.4, heat_img_bgr, 0.6, 0)
    mask = heatmap_colored > 0
    # Broadcast mask over color channels
    overlay = np.where(mask[:, :, None], blended, base)

    # Draw measurement points
    dark_purple = (64, 0, 128)
    for idx, (cx, cy) in centroid_dict.items():
        if idx not in measurements_dict or measurements_dict[idx] is None:
            continue
        px = int((cx - origin[0]) * scale / resolution)
        py = int(orig_img.shape[0] - ((cy - origin[1]) * scale / resolution))
        cv2.circle(overlay, (px, py), 5, dark_purple, -1)
        cv2.putText(overlay, f"{measurements_dict[idx]:.1f}", (px + 5, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, dark_purple, 1)

    return overlay

def generate_elevation_heatmap(
    orig_img, measurements_dict, centroid_dict, origin, resolution, scale,
    zone_size_factor=15, free_threshold=250, decay=1.0, num_iters=200
):
    """
    Generate a heatmap by clustering the map into zones and diffusing known measurement values.
    """
    # Convert image to grayscale
    gray_map = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY) if orig_img.ndim == 3 else orig_img.copy()
    omega = 1.5  # Relaxation factor, must be >1.0 and <2.0
    
    # Cluster into zones and get centroids
    zone_size = int(zone_size_factor)
    zones, zone_centroids = generate_zones(gray_map, zone_size, free_threshold, resolution, origin)

    # Initialize measurement per zone
    zone_measurements = []
    for (zwx, zwy) in zone_centroids:
        val = np.nan
        zx = int((zwx - origin[0]) / resolution)
        zy = int(orig_img.shape[0] - ((zwy - origin[1]) / resolution))
        for idx, (cx, cy) in centroid_dict.items():
            if idx in measurements_dict and measurements_dict[idx] is not None:
                cx_px = int((cx - origin[0]) * scale / resolution)
                cy_px = int(orig_img.shape[0] - ((cy - origin[1]) * scale / resolution))
                dist = np.linalg.norm([zx - cx_px, zy - cy_px])
                if dist < zone_size * 1.5:
                    val = measurements_dict[idx]
                    break
        zone_measurements.append(val)

    zone_measurements = np.array(zone_measurements, dtype=np.float32)
    known = ~np.isnan(zone_measurements)

    # Initialize unknown zones to zero to help diffusion propagate values
    zone_measurements[np.isnan(zone_measurements)] = 0.0

    # Precompute neighbors for each zone
    zone_neighbors = []
    for i, (x0, y0, x1, y1) in enumerate(zones):
        neigh = []
        for j, (xx0, yy0, xx1, yy1) in enumerate(zones):
            if i == j:
                continue
            # Check if zones touch/intersect
            if not (x1 < xx0 or x0 > xx1 or y1 < yy0 or y0 > yy1):
                neigh.append(j)
        zone_neighbors.append(neigh)

    # Diffuse measurements into unknown zones using relaxation
    for _ in range(num_iters):
        for i, neighbors in enumerate(zone_neighbors):
            if known[i] or not neighbors:
                continue
            valid_vals = [zone_measurements[j] for j in neighbors if not np.isnan(zone_measurements[j])]
            if valid_vals:
                avg_neighbor = np.mean(valid_vals)
                residual = avg_neighbor - zone_measurements[i]
                zone_measurements[i] += omega * residual

    # Normalize zone measurements to [0,1]
    valid_vals = zone_measurements[~np.isnan(zone_measurements)]
    if valid_vals.size > 0 and valid_vals.max() > 0:
        zone_measurements /= valid_vals.max()

    # Create zone-based heatmap (float32)
    heatmap_colored = np.zeros_like(gray_map, dtype=np.float32)
    for (x0, y0, x1, y1), v in zip(zones, zone_measurements):
        heatmap_colored[y0:y1, x0:x1] = v

    # Normalize heatmap_colored for colormap
    if heatmap_colored.max() > 0:
        norm_map = heatmap_colored / heatmap_colored.max()
    else:
        norm_map = heatmap_colored

    colormap = cm.get_cmap('gist_heat')
    heat_img = (colormap(norm_map)[:, :, :3] * 255).astype(np.uint8)
    heat_img_bgr = cv2.cvtColor(heat_img, cv2.COLOR_RGB2BGR)

    # Blend zones with original image
    base = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR) if orig_img.ndim == 2 else orig_img.copy()
    blended = cv2.addWeighted(base, 0.4, heat_img_bgr, 0.6, 0)

    # Define mask before use
    mask = heatmap_colored > 0

    # Apply mask to overlay blended zones on base
    overlay = np.where(mask[:, :, None], blended, base)

    # Force uint8 dtype for proper saving
    overlay = overlay.astype(np.uint8)

    # Draw measurement points
    dark_purple = (64, 0, 128)
    for idx, (cx, cy) in centroid_dict.items():
        if idx not in measurements_dict or measurements_dict[idx] is None:
            continue
        px = int((cx - origin[0]) * scale / resolution)
        py = int(orig_img.shape[0] - ((cy - origin[1]) * scale / resolution))
        cv2.circle(overlay, (px, py), 5, dark_purple, -1)
        cv2.putText(overlay, f"{measurements_dict[idx]:.1f}", (px + 5, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, dark_purple, 1)
    # After computing zone_measurements and zones in your function
    plot_3d_elevation(zones, zone_measurements)
    save_elevation_map(elevation_map, "elevation_map.png")
    return overlay

def save_elevation_map(zones, zone_measurements, shape):
    """
    Create a grayscale elevation image from zones and measurements.
    """
    elevation_img = np.zeros(shape, dtype=np.float32)
    for (x0, y0, x1, y1), val in zip(zones, zone_measurements):
        elevation_img[y0:y1, x0:x1] = val
    # Normalize to [0, 255]
    elevation_img = (elevation_img / np.max(elevation_img) * 255).astype(np.uint8)
    cv2.imwrite("elevation_map.png", elevation_img)

def save_waypoints_yaml(centroids, output_path: Path, resolution: float, origin):
    wps=[]; ox,oy=origin[0],origin[1]
    for i,(cx,cy) in enumerate(centroids):
        dist=np.hypot(cx-ox,cy-oy)
        wps.append({'zone':i,'x':cx,'y':cy,'theta':0.0,'distance':dist})
    wps.sort(key=lambda w:w['distance'],reverse=True)
    for w in wps: w.pop('distance')
    with open(output_path,'w') as f: yaml.dump({'waypoints':wps},f)
    print(f"Saved {len(wps)} waypoints to {output_path}")


def overlay_zones_on_map(
    orig_img: np.ndarray,
    output_path: Path,
    zone_size: int,
    origin,
    resolution,
    scale: float,
    output_type: str,
    zones=None,
    centroids=None,
    alpha: float=0.3
):
    if output_type=='heatmap':
        measurements_dict=load_measurements(str(Path(output_path).parent.parent/'yaml_files'/'measurements.yaml'))
        cen_dict=load_zone_centroids(str(Path(output_path).parent.parent/'yaml_files'/'zones_waypoints.yaml'))
        heat_img=generate_heatmap(orig_img,measurements_dict,cen_dict,origin,resolution,scale)
        cv2.imwrite(str(output_path), heat_img)
        print(f"Saved heatmap overlay to {output_path}")
        return
    vis=cv2.cvtColor(orig_img,cv2.COLOR_GRAY2BGR)
    over=vis.copy()
    c1=(255,0,0); c2=(0,255,0)
    for i,((x0,y0,x1,y1),(cx,cy)) in enumerate(zip(zones,centroids)):
        color=c1 if ((y0//zone_size)+(x0//zone_size))%2==0 else c2
        cv2.rectangle(over,(x0,y0),(x1,y1),color,-1)
        px=int((cx-origin[0])/resolution)
        py=int(orig_img.shape[0]-((cy-origin[1])/resolution))
        cv2.circle(over,(px,py),int(scale),(0,0,255),-1)
        cv2.putText(over,str(i),(px,py+int(2*scale)),cv2.FONT_HERSHEY_SIMPLEX,0.4*scale/10,(0,0,0),1)
    cv2.addWeighted(over,alpha,vis,1-alpha,0,vis)
    cv2.imwrite(str(output_path),vis)
    print(f"Saved zones overlay to {output_path}")
