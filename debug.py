"""

Idea: Make previous token bias a disadvantage by interleaving encodings from two trees.

Two points:
* Auto-regressive models have a previous token bias
* Embeddings have two roles, keep information for others, while predicting there own value

"""

import pickle
from collections import defaultdict
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np

def plot_value(runs, key: str, ylabel: str, log_scale=False, ax=None):
    ax = ax or plt.gca()
    colors = {}
    linestyles = {}
    max_epochs = {}
    data = defaultdict(list)
    for run_name, filenames in runs.items():
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
                data[run_name].append(run_data[key])

    # Loop through each run's data to plot the mean line and scatter points
    for run_name, scores in data.items():
        max_scores, mean_scores = np.zeros(len(scores[0])), np.zeros(len(scores[0]))
        for i in range(len(scores[0])):
            for score in scores:
                if i >= len(score):
                    mean_scores[i] += score[-1]
                    continue
                mean_scores[i] += score[i]
                max_scores[i] = max(max_scores[i], score[i])
            mean_scores[i] /= len(scores)
        color, linestyle = None, None
        if run_name in max_epochs:
            mean_scores = mean_scores[:max_epochs[run_name]]
        if run_name in colors:
            color = colors[run_name]
        if run_name in linestyles:
            linestyle = linestyles[run_name]
        line = sns.lineplot(data=mean_scores, label=f'{run_name}', color=color, linestyle=linestyle, ax=ax)
        color = line.get_lines()[-1].get_color()
        # make ed similar color and ed-split (similar color)
        for score in scores:
            if run_name in max_epochs:
                score = score[:max_epochs[run_name]]
            sns.scatterplot(x=range(len(score)), y=score, color=color, alpha=0.2, size=1, legend=False, ax=ax)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale('log')
    ax.set_title(f'{key}')
    ax.legend()


plots = {}

plots['Different tasks hard (b=5, d=3)'] = {
    'fix': [f'out/decoder_l3_h8_e512_wi1_peglobal_learn_te0_ap0.00_ep0.00__fix_b5_d3__64_64__{i}.pkl' for i in range(1)],
    'slwr': [f'out/decoder_l3_h8_e512_wi1_peglobal_learn_te0_ap0.00_ep0.00__slwr_b5_d3__64_64__{i}.pkl' for i in range(1)],
    'snwr': [f'out/decoder_l3_h8_e512_wi1_peglobal_learn_te0_ap0.00_ep0.00__snwr_b5_d3__64_64__{i}.pkl' for i in range(1)],
    'slwor': [f'out/decoder_l3_h8_e512_wi1_peglobal_learn_te0_ap0.00_ep0.00__slwor_b5_d3__64_64__{i}.pkl' for i in range(1)],
    'snwor': [f'out/decoder_l3_h8_e512_wi1_peglobal_learn_te0_ap0.00_ep0.00__snwor_b5_d3__64_64__{i}.pkl' for i in range(1)],
}
plots['Different tasks easy (b=3, d=3)'] = {
    'fix': [f'out/decoder_l3_h8_e512_wi1_peglobal_learn_te0_ap0.00_ep0.00__fix_b3_d3__64_64__{i}.pkl' for i in range(1)],
    'slwr': [f'out/decoder_l3_h8_e512_wi1_peglobal_learn_te0_ap0.00_ep0.00__slwr_b3_d3__64_64__{i}.pkl' for i in range(1)],
    'snwr': [f'out/decoder_l3_h8_e512_wi1_peglobal_learn_te0_ap0.00_ep0.00__snwr_b3_d3__64_64__{i}.pkl' for i in range(1)],
    'slwor': [f'out/decoder_l3_h8_e512_wi1_peglobal_learn_te0_ap0.00_ep0.00__slwor_b3_d3__64_64__{i}.pkl' for i in range(1)],
    'snwor': [f'out/decoder_l3_h8_e512_wi1_peglobal_learn_te0_ap0.00_ep0.00__snwor_b3_d3__64_64__{i}.pkl' for i in range(1)],
}
plots['Interleaving two x4_b2_d7 trees'] = {
    'std inter': [({'color': 'red', 'linestyle': '-'}, f'out/decoder_l3_h8_e512__interleave_x4_b2_d7__64_64__{i}.pkl') for i in range(3)],
    'split inter': [({'color': 'blue', 'linestyle': '-'}, f'out/decoder-split_l3_h8_e512__interleave_x4_b2_d7__64_64__{i}.pkl') for i in range(3)],
}
plots['Weight Initialization x4_b2_d7'] = {
    'none': [({'color': 'red', 'linestyle': '-'},
              f'out/decoder_l3_h8_e512_wi0_peglobal_learn_te0_ap0.00_ep0.00__x4_b2_d7__64_64__{i}.pkl') for i in range(2)],
    'init': [({'color': 'blue', 'linestyle': '-'},
              f'out/decoder_l3_h8_e512_wi1_peglobal_learn_te0_ap0.00_ep0.00__x4_b2_d7__64_64__{i}.pkl') for i in range(2)],
}
plots['Weight Initialization x4_b2_d8'] = {
    'none': [({'color': 'red', 'linestyle': '-'},
              f'out/decoder_l3_h8_e512_wi0_peglobal_learn_te0_ap0.00_ep0.00__x4_b2_d8__64_64__{i}.pkl') for i in range(1)],
    'init': [({'color': 'blue', 'linestyle': '-'},
              f'out/decoder_l3_h8_e512_wi1_peglobal_learn_te0_ap0.00_ep0.00__x4_b2_d8__64_64__{i}.pkl') for i in range(1)],
}
plots['Positional Encoding x4_b2_d8'] = {
    'learned': [({'color': 'red', 'linestyle': '-'},
                 f'out/decoder_l3_h8_e512_wi1_peglobal_learn_te0_ap0.00_ep0.00__x4_b2_d8__64_64__{i}.pkl') for i in range(1)],
    'sinusoidal': [({'color': 'blue', 'linestyle': '-'},
                    f'out/decoder_l3_h8_e512_wi1_peglobal_sinusoidal_te0_ap0.00_ep0.00__x4_b2_d8__64_64__{i}.pkl') for i in range(1)],
}
plots['Positional Encoding x4_b2_d7'] = {
    'learned': [({'color': 'red', 'linestyle': '-'},
                 f'out/decoder_l3_h8_e512_wi1_peglobal_learn_te0_ap0.00_ep0.00__x4_b2_d7__64_64__{i}.pkl') for i in range(1)],
    'sinusoidal': [({'color': 'blue', 'linestyle': '-'},
                    f'out/decoder_l3_h8_e512_wi1_peglobal_sinusoidal_te0_ap0.00_ep0.00__x4_b2_d7__64_64__{i}.pkl') for i in range(1)],
}
plots['Tie embeddings'] = {
    'no tie': [f'out/decoder_l3_h8_e512_wi1_peglobal_learn_te0_ap0.00_ep0.00__x4_b2_d8__64_64__{i}.pkl' for i in range(1)],
    'tie': [f'out/decoder_l3_h8_e512_wi1_peglobal_learn_te1_ap0.00_ep0.00__x4_b2_d8__64_64__{i}.pkl' for i in range(1)],
}

for plot_name, runs in plots.items():
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    plot_value(runs, 'tree_acc_history', 'Tree Generation Accuracy', ax=ax[0])
    plot_value(runs, 'loss_history', 'Loss', log_scale=True, ax=ax[1])
    plt.suptitle(plot_name)
    plt.savefig(f'imgs/{plot_name.replace(" ", "-")}.png')
    # plt.show()
