import json
import os

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from rastervision.utils.files import file_to_str, make_dir
from noisy_buildings_semseg.data import get_root_uri, get_exp_id, NoiseMode


class Stats():
    def __init__(self, levels, precisions, recalls, f1s):
        self.levels = np.array(levels)
        self.precisions = np.array(precisions)
        self.recalls = np.array(recalls)
        self.f1s = np.array(f1s)


def get_stats(root_uri, noise_type, levels, runs):
    precisions = []
    recalls = []
    f1s = []
    num_runs = len(runs)
    for level in levels:
        precision = 0.
        recall = 0.
        f1 = 0.
        for run in runs:
            noise_mode = NoiseMode(noise_type, level)
            exp_id = get_exp_id(noise_mode, run)
            eval_uri = os.path.join(root_uri, 'rv', 'eval', exp_id, 'eval.json')
            eval_json = json.loads(file_to_str(eval_uri))

            class_id = 1
            building_eval = next(filter(
                lambda e: e['class_id'] == class_id, eval_json['overall']))
            precision += building_eval['precision']
            recall += building_eval['recall']
            f1 += building_eval['f1']
        precision /= num_runs
        recall /= num_runs
        f1 /= num_runs

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return Stats(levels, precisions, recalls, f1s)


def save_plot(plot_uri, noise_type, stats):
    plt.cla()
    plt.plot(stats.levels, stats.precisions, label='precision')
    plt.plot(stats.levels, stats.recalls, label='recall')
    plt.plot(stats.levels, stats.f1s, label='f1')
    plt.xticks(stats.levels)
    plt.ylim([0.0, 1.0])
    plt.xlabel('Noise level')
    plt.ylabel('Prediction accuracy')
    plt.legend()

    if noise_type == NoiseMode.DROP:
        title = 'Trained on randomly deleted labels'
    elif noise_type == NoiseMode.SHIFT:
        title = 'Trained on randomly shifted labels'
    plt.title(title)
    plt.savefig(plot_uri)


def main():
    use_remote_data = True
    root_uri = get_root_uri(use_remote_data)

    def process_noise_type(noise_type, levels, runs):
        stats = get_stats(root_uri, noise_type, levels, runs)
        curves_uri = os.path.join(get_root_uri(False), 'plots', 'curves')
        make_dir(curves_uri)
        plot_uri = os.path.join(curves_uri, 'plot-{}.png'.format(noise_type))
        print('Saving plot to {}...'.format(plot_uri))
        save_plot(plot_uri, noise_type, stats)

    process_noise_type(NoiseMode.SHIFT, [0, 10, 20, 40], [0])
    process_noise_type(NoiseMode.DROP, [0.0, 0.1, 0.2, 0.4], [0])


if __name__ == '__main__':
    main()
