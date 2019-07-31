import random
import tempfile
import os

import numpy as np
from sklearn.metrics import confusion_matrix

from rastervision.utils.files import json_to_file
from noisy_buildings_semseg.data import (
    VegasBuildings, NoiseMode, get_root_uri, stats_uri)
from noisy_buildings_semseg.exp import (
    build_task, build_scene)


def compute_noise_metrics(scene_ids, spacenet_config, noise_mode, building_class_id):
    print('Computing metrics for {}...'.format(str(noise_mode)))
    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name
    conf_mat = np.zeros((3, 3))

    for scene_id in scene_ids:
        task_config = build_task(spacenet_config.get_class_map())

        orig_scene = build_scene(
            task_config, spacenet_config, noise_mode, scene_id, True)
        orig_scene = orig_scene.create_scene(task_config, tmp_dir)
        noisy_scene = build_scene(
            task_config, spacenet_config, noise_mode, scene_id, False)
        noisy_scene = noisy_scene.create_scene(task_config, tmp_dir)
        with orig_scene.ground_truth_label_source.source.activate():
            orig_arr = orig_scene.ground_truth_label_source.source.get_image_array()
        with noisy_scene.ground_truth_label_source.source.activate():
            noisy_arr = noisy_scene.ground_truth_label_source.source.get_image_array()
        conf_mat += confusion_matrix(orig_arr.ravel(), noisy_arr.ravel(), labels=[0, 1, 2])

    return conf_mat.tolist()

def main():
    random.seed(5678)
    use_remote_data = False
    vb = VegasBuildings(use_remote_data)
    scene_ids = vb.get_scene_ids()

    shifts = [0, 10, 20, 30, 40, 50]
    probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    sample_sz = 50
    random.shuffle(scene_ids)
    scene_ids = scene_ids[0:sample_sz]
    building_class_id = vb.get_class_map()['Building'][0]

    stats = {}
    for shift in shifts:
        nm = NoiseMode(NoiseMode.SHIFT, shift)
        stats[str(nm)] = compute_noise_metrics(scene_ids, vb, nm, building_class_id)

    for prob in probs:
        nm = NoiseMode(NoiseMode.DROP, prob)
        stats[str(nm)] = compute_noise_metrics(scene_ids, vb, nm, building_class_id)

    json_to_file(stats, stats_uri)

if __name__ == '__main__':
    main()
