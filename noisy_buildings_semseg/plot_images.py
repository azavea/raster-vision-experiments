import os
import collections
import json

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import rasterio

import rastervision as rv
from rastervision.rv_config import RVConfig
from rastervision.data import ActivateMixin, StatsTransformer, RasterStats, GeoTiffSource
from rastervision.data.utils import geojson_to_shapes
from rastervision.core import ClassMap
from rastervision.utils.files import make_dir, file_to_str
from noisy_buildings_semseg.data import (
    get_root_uri, get_exp_id, NoiseMode, VegasBuildings)


NOISY_LABELS = 'noisy-labels'
PREDS = 'preds'

ExpData = collections.namedtuple(
    'ExpData', ['raster_arr', 'label_geoms', 'noisy_label_geoms', 'pred_arr'])


def get_exp_data(vb, nm, id):
    tmp_dir = RVConfig.get_tmp_dir().name

    # Get raster source.
    stats = RasterStats()
    stats.means = np.array([462.4939189390183, 633.5548961566001, 464.99947912120706])
    stats.stds = np.array([248.46624190502172, 271.07249107975275, 162.06929299061807])
    raster_uri = vb.get_raster_source_uri(id)
    raster_source = GeoTiffSource([raster_uri], [StatsTransformer(stats)], tmp_dir)

    # Get label raster source.
    background_class_id = 2
    label_uri = vb.get_geojson_uri(id)
    '''
    label_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIZED_SOURCE) \
        .with_vector_source(label_uri) \
        .with_rasterizer_options(background_class_id) \
        .build()
    class_map = ClassMap.construct_from(vb.get_class_map())
    label_raster_source = label_raster_source.create_source(
        tmp_dir, raster_source.get_crs_transformer(), raster_source.get_extent(),
        class_map=class_map)
    '''
    label_geoms = geojson_to_shapes(json.loads(
        file_to_str(label_uri)), raster_source.get_crs_transformer())

    noisy_label_uri = vb.get_noisy_geojson_uri(nm, id)
    noisy_label_geoms = geojson_to_shapes(json.loads(
        file_to_str(noisy_label_uri)), raster_source.get_crs_transformer())

    # Get prediction raster source.
    run = 0
    exp_id = get_exp_id(nm, run)
    prediction_uri = os.path.join(
        vb.root_uri, 'rv', 'predict', exp_id, '{}.tif'.format(id))
    pred_raster_source = GeoTiffSource([prediction_uri], [], tmp_dir)

    with ActivateMixin.compose(raster_source, pred_raster_source):
        # Plot labels.
        return ExpData(
            raster_source.get_image_array(), label_geoms, noisy_label_geoms,
            pred_raster_source.get_image_array())


def plot_labels(all_exp_data, noise_type, levels, ids, plot_mode, plot_dir):
    building_class_id = 1
    subplot_ind = 1
    if plot_mode == NOISY_LABELS:
        levels = levels[1:]
        ids = ids[0:1]

    for id in ids:
        for level in levels:
            plt.subplot(len(ids), len(levels), subplot_ind)
            subplot_ind += 1

            exp_data = all_exp_data[(noise_type, level, id)]
            plt.imshow(exp_data.raster_arr)

            # Plot ground truth labels
            if not (plot_mode == NOISY_LABELS and noise_type == NoiseMode.DROP):
                for label_geom, _ in exp_data.label_geoms:
                    x, y = label_geom.exterior.xy
                    plt.plot(x, y, color='lightblue', alpha=0.8, linewidth=0.5)

            if plot_mode == NOISY_LABELS:
                for label_geom, _ in exp_data.noisy_label_geoms:
                    x, y = label_geom.exterior.xy
                    plt.plot(x, y, color='orange', alpha=0.8, linewidth=0.5)

            elif plot_mode == PREDS:
                label_arr = np.squeeze(exp_data.pred_arr == building_class_id).astype(int) * 140
                plt.imshow(label_arr, cmap=mpl.cm.hot, vmin=0, vmax=255, alpha=0.7)

            plt.axis('off')

            if id == ids[0]:
                plt.title(str(level))

    fig = plt.gcf()
    if plot_mode == PREDS:
        if noise_type == NoiseMode.SHIFT:
            title = 'Predictions after training on random shifts'
        elif noise_type == NoiseMode.DROP:
            title = 'Predictions after training on random deletions'
    elif plot_mode == NOISY_LABELS:
        if noise_type == NoiseMode.SHIFT:
            title = 'Noisy labels with random shifts'
        elif noise_type == NoiseMode.DROP:
            title = 'Noisy labels with random deletions'

    fig.suptitle(title, fontsize=14)
    plot_uri = os.path.join(plot_dir, '{}-{}.png'.format(plot_mode, noise_type))
    plt.savefig(plot_uri, dpi=300)
    print('Saved plot to {}'.format(plot_uri))


def main():
    use_remote_data = True
    vb = VegasBuildings(use_remote_data)
    plot_dir = os.path.join(get_root_uri(False), 'plots', 'images')
    make_dir(plot_dir)

    ids = [3590, 581, 1246]
    noise_types = [NoiseMode.SHIFT, NoiseMode.DROP]
    plot_modes = [NOISY_LABELS, PREDS]
    for noise_type in noise_types:
        all_exp_data = {}
        if noise_type == NoiseMode.SHIFT:
            levels = [0, 10, 20, 40]
        elif noise_type == NoiseMode.DROP:
            levels = [0.0, 0.1, 0.2, 0.4]

        for level in levels:
            for id in ids:
                nm = NoiseMode(noise_type, level)
                exp_data = get_exp_data(vb, nm, id)
                all_exp_data[(noise_type, level, id)] = exp_data

        for plot_mode in plot_modes:
            plot_labels(all_exp_data, noise_type, levels, ids, plot_mode, plot_dir)
            

if __name__ == '__main__':
    main()
