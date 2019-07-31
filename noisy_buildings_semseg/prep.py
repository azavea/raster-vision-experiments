import json
import random

import rasterio
from rastervision.data import RasterioCRSTransformer
from rastervision.utils.files import file_to_str, str_to_file

from noisy_buildings_semseg.data import VegasBuildings, NoiseMode


def make_noisy_data(scene_ids, vb, noise_mode):
    for scene_id in scene_ids:
        raster_uri = vb.get_raster_source_uri(scene_id)
        with rasterio.open(raster_uri) as dataset:
            crs_trans = RasterioCRSTransformer.from_dataset(dataset)
            labels_uri = vb.get_geojson_uri(scene_id)
            geojson = json.loads(file_to_str(labels_uri))
            new_features = []
            for f in geojson['features']:
                if noise_mode.type == NoiseMode.DROP:
                    if random.uniform(0.0, 1.0) < noise_mode.level:
                        pass
                    else:
                        new_features.append(f)
                elif noise_mode.type == NoiseMode.SHIFT:
                    if f['geometry']['type'] == 'Polygon':
                        map_coords_list = [f['geometry']['coordinates'][0]]
                    elif f['geometry']['type'] == 'MultiPolygon':
                        map_coords_list = [mc[0] for mc in f['geometry']['coordinates']]
                    else:
                        print('Skipping ' + f['geometry']['type'])
                        continue

                    for map_coords in map_coords_list:
                        pixel_coords = [crs_trans.map_to_pixel(p) for p in map_coords]
                        x_shift = round(random.uniform(-noise_mode.level, noise_mode.level))
                        y_shift = round(random.uniform(-noise_mode.level, noise_mode.level))
                        shift_coords = [(p[0] + x_shift, p[1] + y_shift) for p in pixel_coords]
                        shift_map_coords = [crs_trans.pixel_to_map(p) for p in shift_coords]

                        new_f = {
                            'geometry': {
                                'type': 'Polygon',
                                'coordinates': [shift_map_coords]
                            },
                            'properties': f['properties']
                        }
                        new_features.append(new_f)

            new_geojson = {
                'type': 'FeatureCollection',
                'features': new_features
            }
            noisy_uri = vb.get_noisy_geojson_uri(noise_mode, scene_id)
            print(noisy_uri)
            str_to_file(json.dumps(new_geojson), noisy_uri)


def main():
    random.seed(5678)
    use_remote_data = False
    vb = VegasBuildings(use_remote_data)
    scene_ids = vb.get_scene_ids()

    shifts = [0, 10, 20, 30, 40]
    probs = [0.0, 0.1, 0.2, 0.3, 0.4]

    for shift in shifts:
        nm = NoiseMode(NoiseMode.SHIFT, shift)
        make_noisy_data(scene_ids, vb, nm)

    for prob in probs:
        nm = NoiseMode(NoiseMode.DROP, prob)
        make_noisy_data(scene_ids, vb, nm)


if __name__ == '__main__':
    main()
