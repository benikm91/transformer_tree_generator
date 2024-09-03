"""

Idea: Make previous token bias a disadvantage by interleaving encodings from two trees.

Two points:
* Auto-regressive models have a previous token bias
* Embeddings have two roles, keep information for others, while predicting there own value

"""

import pickle
from collections import defaultdict
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tree_structure import FixSequenceSampleGen, SampleLeavesWithReplacementGen, SampleLeavesWithoutReplacementGen, \
    SampleNodesWithReplacementGen, SampleNodesWithoutReplacementGen


def plot_value(run_fs, key: str, ylabel: str, log_scale=False, ax=None, deltas=None):
    ax = ax or plt.gca()
    colors = {}
    linestyles = {}
    max_epochs = {}
    data = defaultdict(lambda: defaultdict(list))
    for run_name, filenames_f in run_fs.items():
        for depth in range(1, 4):
            filenames = filenames_f(depth)
            # if key == 'loss_history':
            #     delta = -expected_final_loss[run_name]
            for filename in filenames:
                if isinstance(filename, tuple):
                    params, filename = filename
                    if 'max_epoch' in params:
                        max_epochs[run_name] = params['max_epoch']
                    if 'color' in params:
                        colors[run_name] = params['color']
                    if 'linestyle' in params:
                        linestyles[run_name] = params['linestyle']
                with open(filename, 'rb') as f:
                    run_data = pickle.load(f)
                    delta = 0
                    if deltas is not None:
                        delta = deltas[run_name](depth)
                    data[run_name][depth].append([x + delta for x in run_data[key]])

    # Loop through each run's data to plot the mean line and scatter points
    print(data)

    max_lens = {}
    for depth in range(1, 4):
        max_lens[depth] = max(max(len(scores) + 1 for scores in depth_scores[depth]) for depth_scores in data.values())

    for run_name, depth_scores in data.items():
        dx = 0
        max_scores, mean_scores = np.zeros(sum(max_lens.values())), np.zeros(sum(max_lens.values()))
        for depth in range(1, 4):
            scores = depth_scores[depth]
            length = max(len(scores[k]) for k in range(len(scores)))
            for i in range(max_lens[depth]):
                if i >= length:
                    mean_scores[dx+i] = np.nan
                for score in scores:
                    if i >= len(score):
                        mean_scores[dx+i] += score[-1] / len(scores)
                        continue
                    mean_scores[dx+i] += score[i] / len(scores)
            dx += max_lens[depth]
            ax.axvline(x=dx, color='r', linestyle='--')
            ax.text(dx - 2, 0, f'depth={depth}', rotation=90, color='r')

        _ = sns.pointplot(
            ax=ax,
            data=mean_scores,
            errorbar=None,
            markers='o',
            label=f'{run_name}',
        )
        ax.set_xticklabels([])
        # plot red vertical line at start_x
        # color = line.get_lines()[-1].get_color()
        # # make ed similar color and ed-split (similar color)
        # for score in scores:
        #     if run_name in max_epochs:
        #         score = score[:max_epochs[run_name]]
        #     sns.scatterplot(x=xs, y=score, color=color, alpha=0.2, size=1, legend=False, ax=ax)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale('log')
    ax.set_title(f'{key}')
    ax.legend()


plots = {}


def get_filename(task: str, depth: int, run_id: int) -> str:
    return f'out/task/curriculum__decoder_l3_h8_e512_wi1_peglobal_learn_te0_ap0.00_ep0.00__{task}_b5_d3__64__actual__{task}_b5_d{depth}__{run_id}.pkl'

plots['Different task (bf=5)'] = {
    'fix': lambda depth: [get_filename('fix', depth, i) for i in range(3)],
    'lwr': lambda depth: [get_filename('slwr', depth, i) for i in range(3)],
    'nwr': lambda depth: [get_filename('snwr', depth, i) for i in range(3)],
    'lwor': lambda depth: [get_filename('slwor', depth, i) for i in range(3)],
    'nwor': lambda depth: [get_filename('snwor', depth, i) for i in range(3)],
}
expected_final_loss = {
    'fix': lambda depth: FixSequenceSampleGen(branch_factor=5, depth=depth).best_possible_loss,
    'lwr': lambda depth: SampleLeavesWithReplacementGen(branch_factor=5, depth=depth).best_possible_loss,
    'nwr': lambda depth: SampleNodesWithReplacementGen(branch_factor=5, depth=depth).best_possible_loss,
    'lwor': lambda depth: SampleLeavesWithoutReplacementGen(branch_factor=5, depth=depth).best_possible_loss,
    'nwor': lambda depth: SampleNodesWithoutReplacementGen(branch_factor=5, depth=depth).best_possible_loss,
}

for plot_name, run_fs in plots.items():
    fig, ax = plt.subplots(1, 2, figsize=(40, 5))
    plot_value(run_fs, 'tree_acc_history', 'Tree Generation Accuracy', ax=ax[0])
    plot_value(run_fs, 'loss_history', 'Loss', ax=ax[1], log_scale=False)
    plt.suptitle(plot_name)
    plt.savefig(f'imgs/{plot_name.replace(" ", "-")}.png')
    plt.show()
