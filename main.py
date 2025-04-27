#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
import yaml
from utils import (
    convert_pgm_to_png,
    load_map_metadata,
    generate_zones,
    save_waypoints_yaml,
    overlay_zones_on_map,
    Measurements
)


def main():
    parser = argparse.ArgumentParser(description="Choose the output type")
    parser.add_argument('--output', type=str, default='zones', help='Type of output: zones or heatmap')
    parser.add_argument('--values', type=str, default='existing', help='Type of output: existing or absent')
    args = parser.parse_args()

    # Load configuration
    config_path = Path(__file__).parent /'yaml_files'/ 'config.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    map_yaml = cfg['map_yaml']
    pgm_file = cfg['pgm_file']
    scale = cfg['scale']
    native_scale = 1
    native_zone_size = int(cfg['zone_size'] * native_scale)
    scaled_zone_size = int(cfg['zone_size'] * scale)
    free_threshold = cfg['free_threshold']

    # Load map metadata
    origin, resolution = load_map_metadata(map_yaml)

    # Convert PGM to PNG
    native_raw_png = Path(__file__).parent /'native_maps'/'raw_map.png'
    convert_pgm_to_png(pgm_file, str(native_raw_png), scale=native_scale)
    scaled_raw_png = Path(__file__).parent /'scaled_maps'/'raw_map.png'
    convert_pgm_to_png(pgm_file, str(scaled_raw_png), scale=scale)
    # Read grayscale map
    img = cv2.imread(str(native_raw_png), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot open map image: {native_raw_png}")

    script_folder = Path(__file__).parent
    output_yaml = script_folder /'yaml_files'/ 'zones_waypoints.yaml'

    if args.output == 'heatmap':
        if args.values == 'existing':
            output_img = script_folder /'scaled_maps'/ 'zones_overlay.png'
            # Heatmap overlay
            overlay_path = script_folder / 'scaled_maps' /'heatmap.png'
            # Use previously scaled raw_map.png for heatmap
            img_heatmap = cv2.imread(str(scaled_raw_png), cv2.IMREAD_GRAYSCALE)
            overlay_zones_on_map(
                img_heatmap, overlay_path,
                scaled_zone_size, origin, resolution,
                scale, 'heatmap'
            )
        else:
            # USE THIS CLASS TO PLUG IN YOUR MEASUREMENT FUNCTION
            measurements = Measurements(
            heatmap_yaml_path= script_folder /'yaml_files'/ 'measurements.yaml',
            zones_yaml_path= script_folder /'yaml_files'/ 'zones_waypoints.yaml'
            )
            measurements.measure_all()
            measurements.save_measurements_yaml()

            output_img = script_folder /'scaled_maps'/ 'zones_overlay.png'
            # Heatmap overlay
            overlay_path = script_folder / 'scaled_maps' /'heatmap.png'
            # Use previously scaled raw_map.png for heatmap
            img_heatmap = cv2.imread(str(scaled_raw_png), cv2.IMREAD_GRAYSCALE)
            overlay_zones_on_map(
                img_heatmap, overlay_path,
                scaled_zone_size, origin, resolution,
                scale, 'heatmap'
            )
    else:
        # Native map processing with original scale
        output_img = script_folder /'native_maps'/ 'zones_overlay.png'
        zones, cents = generate_zones(
            img, native_zone_size, free_threshold,
            resolution, origin
        )
        save_waypoints_yaml(cents, output_yaml, resolution, origin)
        overlay_zones_on_map(
            img, output_img,
            native_zone_size, origin,
            resolution, native_scale,
            'zones', zones=zones, centroids=cents
        )

        # Scaled map processing for human visualisation
        output_img = script_folder /'scaled_maps'/ 'zones_overlay.png'
        img = cv2.imread(str(scaled_raw_png), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot open map image: {native_raw_png}")
        zones, cents = generate_zones(
            img, scaled_zone_size, free_threshold,
            resolution, origin
        )
        overlay_zones_on_map(
            img, output_img,
            scaled_zone_size, origin,
            resolution, scale,
            'zones', zones=zones, centroids=cents
        )

if __name__ == '__main__':
    main()
