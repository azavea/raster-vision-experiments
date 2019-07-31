import json
import os

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from rastervision.utils.files import file_to_str, make_dir, file_to_json
from noisy_buildings_semseg.data import (
    get_root_uri, get_exp_id, NoiseMode, rv_output_dir, stats_uri)


class Stats():
    def __init__(self, levels, gt_conf_mats, pred_conf_mats):
        self.levels = np.array(levels)
        self.gt_conf_mats = np.array(gt_conf_mats)
        self.pred_conf_mats = np.array(pred_conf_mats)


def get_stats(root_uri, noise_type, levels, runs, level_metrics_dict):
    num_runs = len(runs)
    gt_conf_mats = []
    pred_conf_mats = []

    for level in levels:
        noise_mode = NoiseMode(noise_type, level)
        pred_conf_mat = np.zeros((3, 3))
        for run in runs:
            exp_id = get_exp_id(noise_mode, run)
            eval_uri = os.path.join(
                root_uri, rv_output_dir, 'eval', exp_id, 'eval.json')
            eval_json = json.loads(file_to_str(eval_uri))

            avg_eval = next(filter(
                lambda e: e['class_name'] == 'average', eval_json['overall']))
            # Hack to deal with fact that some experiments were run with
            # conf_mat with 3 rows (one for the zero class), and some were
            # run with 2 rows.
            cm = np.array(avg_eval['conf_mat'])
            if cm.shape[0] == 3:
                pred_conf_mat += cm
            elif cm.shape[0] == 2:
                pred_conf_mat[1:, :] += cm
        pred_conf_mat /= num_runs

        pred_conf_mats.append(pred_conf_mat)
        gt_conf_mats.append(level_metrics_dict[str(noise_mode)])

    return Stats(levels, gt_conf_mats, pred_conf_mats)


def get_probs(conf_mats, etype):
    probs = []
    for conf_mat in conf_mats:
        if etype == '1->2':
            probs.append(conf_mat[1, 2] / conf_mat[1, :].sum())
        else:
            probs.append(conf_mat[2, 1] / conf_mat[2, :].sum())
    return probs


def get_accs(conf_mats):
    return np.array([np.diag(cm).sum() / cm.sum() for cm in conf_mats])


def get_building_f1s(conf_mats):
    recalls = np.array([cm[1, 1] / cm[1, :].sum() for cm in conf_mats])
    precs = np.array([cm[1, 1] / cm[:, 1].sum() for cm in conf_mats])
    f1s = 2 * (precs * recalls) / (precs + recalls)
    return f1s


def save_prob_plot(plot_uri, noise_type, stats):
    if noise_type == NoiseMode.DROP:
        title = 'Trained on randomly dropped labels'
    elif noise_type == NoiseMode.SHIFT:
        title = 'Trained on randomly shifted labels'
    plt.suptitle(title)

    for plot_ind, xtype in enumerate(['1->2', '2->1']):
        if noise_type == NoiseMode.SHIFT:
            plt.subplot(2, 2, plot_ind+1)

        elif noise_type == NoiseMode.DROP and plot_ind == 1:
            continue

        x = get_probs(stats.gt_conf_mats, xtype)
        x = [round(_x, 2) for _x in x]
        y12 = get_probs(stats.pred_conf_mats, '1->2')
        y21 = get_probs(stats.pred_conf_mats, '2->1')
        plt.plot(x, y12, label='p(1->2)')
        plt.plot(x, y21, label='p(2->1)')
        plt.xticks(x)
        plt.xlabel('Label error: p({})'.format(xtype))
        if plot_ind == 0:
            plt.ylabel('Prediction error')
        plt.legend()

    plt.savefig(plot_uri, dpi=300)
    plt.close()


def save_metric_plot(plot_uri, drop_stats, shift_stats, metric='acc'):
    def _plot(stats, label):
        if metric == 'acc':
            x = get_accs(stats.gt_conf_mats)
            y = get_accs(stats.pred_conf_mats)
            xlabel = 'Label Accuracy'
            ylabel = 'Prediction Accuracy'
        elif metric == 'building_f1':
            x = get_building_f1s(stats.gt_conf_mats)
            y = get_building_f1s(stats.pred_conf_mats)
            xlabel = 'Label Building F1'
            ylabel = 'Prediction Building F1'

        plt.plot(x, y, label=label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        return x

    dropx = _plot(drop_stats, 'dropped')
    shiftx = _plot(shift_stats, 'shifted')
    x = np.concatenate([dropx, shiftx])

    max_level = np.max(x)
    min_level = np.min(x)
    span = max_level - min_level
    plt.xlim(max_level + 0.05 * span, min_level - 0.05 * span)
    plt.legend()
    # plt.xticks(x.round(3))

    plt.savefig(plot_uri, dpi=300)
    plt.close()


def main():
    use_remote_data = True
    root_uri = get_root_uri(use_remote_data)
    level_metrics_dict = file_to_json(stats_uri)

    shifts = [0, 10, 20, 30, 40, 50]
    probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    runs = [0]

    drop_stats = get_stats(
        root_uri, NoiseMode.DROP, probs, runs, level_metrics_dict)
    shift_stats = get_stats(
        root_uri, NoiseMode.SHIFT, shifts, runs, level_metrics_dict)

    curves_dir = os.path.join(get_root_uri(False), 'plots', 'curves')
    make_dir(curves_dir)
    plot_uri = os.path.join(curves_dir, 'plot-combined.png')
    print('Saving plot to {}...'.format(plot_uri))
    save_metric_plot(plot_uri, drop_stats, shift_stats, metric='building_f1')


if __name__ == '__main__':
    main()
