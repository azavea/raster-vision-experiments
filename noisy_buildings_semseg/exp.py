import random
import os

import rastervision as rv
from noisy_buildings_semseg.data import (
    VegasBuildings, get_root_uri, get_exp_id, NoiseMode)


def build_scene(task, spacenet_config, noise_mode, id, is_validation):
    raster_source = rv.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE) \
                      .with_uri(spacenet_config.get_raster_source_uri(id)) \
                      .with_channel_order([0, 1, 2]) \
                      .with_stats_transformer() \
                      .build()

    vector_source = (spacenet_config.get_geojson_uri(id) if is_validation else
                     spacenet_config.get_noisy_geojson_uri(noise_mode, id))
    background_class_id = 2
    label_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIZED_SOURCE) \
        .with_vector_source(vector_source) \
        .with_rasterizer_options(background_class_id) \
        .build()
    label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION) \
        .with_raster_source(label_raster_source) \
        .build()

    scene = rv.SceneConfig.builder() \
                          .with_task(task) \
                          .with_id(id) \
                          .with_raster_source(raster_source) \
                          .with_label_source(label_source) \
                          .build()

    return scene


def build_dataset(task, spacenet_config, test, noise_mode):
    scene_ids = spacenet_config.get_scene_ids()
    if len(scene_ids) == 0:
        raise ValueError('No scenes found. Something is configured incorrectly.')
    scene_ids.sort()
    random.seed(5678)
    random.shuffle(scene_ids)
    train_prop = 0.8

    num_ids = len(scene_ids)
    # Use subset of scenes.
    num_ids = 1000
    if test:
        num_ids = 20
    num_train_ids = round(num_ids * train_prop)
    num_val_ids = num_ids - num_train_ids
    train_ids = scene_ids[0:num_train_ids]
    val_ids = scene_ids[num_train_ids:num_train_ids+num_val_ids]

    is_validation = False
    train_scenes = [build_scene(task, spacenet_config, noise_mode, id, is_validation)
                    for id in train_ids]
    is_validation = True
    val_scenes = [build_scene(task, spacenet_config, noise_mode, id, is_validation)
                  for id in val_ids]
    dataset = rv.DatasetConfig.builder() \
                              .with_train_scenes(train_scenes) \
                              .with_validation_scenes(val_scenes) \
                              .build()

    return dataset


def build_task(class_map):
    task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                        .with_chip_size(300) \
                        .with_classes(class_map) \
                        .with_chip_options(
                            chips_per_scene=9,
                            debug_chip_probability=1.0,
                            negative_survival_probability=0.25,
                            target_classes=[1],
                            target_count_threshold=1000) \
                        .build()
    return task


def build_backend(task, test):
    debug = False
    batch_size = 8
    num_steps = 30000
    if test:
        debug = True
        num_steps = 1
        batch_size = 1

    backend = rv.BackendConfig.builder(rv.TF_DEEPLAB) \
                              .with_task(task) \
                              .with_model_defaults(rv.MOBILENET_V2) \
                              .with_num_steps(num_steps) \
                              .with_batch_size(batch_size) \
                              .with_debug(debug) \
                              .build()

    return backend


def str_to_bool(x):
    if type(x) == str:
        if x.lower() == 'true':
            return True
        elif x.lower() == 'false':
            return False
        else:
            raise ValueError('{} is expected to be true or false'.format(x))
    return x


class NoisyBuildingsSemseg(rv.ExperimentSet):
    def exp_main(self, use_remote_data=True, test=False):
        """Run experiments on the Spacenet Vegas building semantic segmentation dataset.

        Each experiment using a different set of labels which were created from the
        original by dropping each feature with some probability. Only the training set
        uses the noisy labels -- the validation set uses the original labels.

        Args:
            root_uri: (str): root of where to put output
            use_remote_data: (bool or str) if True or 'True', then use data from S3,
                else local
            test: (bool or str) if True or 'True', run a very small experiment as a
                test and generate debug output
        """
        test = str_to_bool(test)
        use_remote_data = str_to_bool(use_remote_data)
        root_uri = get_root_uri(use_remote_data)
        root_uri = os.path.join(root_uri, 'rv')
        spacenet_config = VegasBuildings(use_remote_data)
        experiments = []
        runs = [0]

        noise_modes = [
            NoiseMode(NoiseMode.SHIFT, 0),
            NoiseMode(NoiseMode.SHIFT, 10),
            NoiseMode(NoiseMode.SHIFT, 20),
            NoiseMode(NoiseMode.SHIFT, 40)
        ]

        noise_modes = [
            NoiseMode(NoiseMode.DROP, 0.0),
            NoiseMode(NoiseMode.DROP, 0.1),
            NoiseMode(NoiseMode.DROP, 0.2),
            NoiseMode(NoiseMode.DROP, 0.4)
        ]

        for nm in noise_modes:
            for run in runs:
                exp_id = get_exp_id(nm, run)
                task = build_task(spacenet_config.get_class_map())
                backend = build_backend(task, test)
                analyzer = rv.AnalyzerConfig.builder(rv.STATS_ANALYZER) \
                                            .build()
                dataset = build_dataset(task, spacenet_config, test, nm)

                experiment = rv.ExperimentConfig.builder() \
                                                .with_id(exp_id) \
                                                .with_task(task) \
                                                .with_backend(backend) \
                                                .with_analyzer(analyzer) \
                                                .with_dataset(dataset) \
                                                .with_root_uri(root_uri) \
                                                .build()
                experiments.append(experiment)

        return experiments


if __name__ == '__main__':
    rv.main()
