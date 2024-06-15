import argparse
import re
import os

import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


def load_data(data_dir):
    import load_data
    return load_data.load_imagenet_torch_preds(data_dir)


def predicted_risk(P, w, beta):
    import torch

    P, w = torch.Tensor(P), torch.Tensor(w)
    w_P = P * w.reshape(1, -1)
    w_P_sq = torch.square(P) * w.reshape(1, -1)
    sort_arr, idx_arr = torch.sort(w_P, descending=True, dim=-1, stable=True)

    sums_arr = torch.cumsum(sort_arr, dim=-1)
    top_idx = torch.argmax((sums_arr >= beta).int(), dim=-1)
    # if all sums are below beta, set to last possible index (i.e., include all labels)
    sum_small_idxs = torch.all(sums_arr < beta, dim=-1)
    top_idx[sum_small_idxs] = w.shape[0] - 1

    exc_sums = []
    for i in range(P.shape[0]):
        if top_idx[i] == w.shape[0] - 1:
            exc_sums.append(0)
        else:
            exc_idxs = idx_arr[i, (top_idx[i] + 1):]
            exc_sums.append(((w[exc_idxs]**2) * P[i, :][exc_idxs]).sum())
    return torch.Tensor(exc_sums).numpy()


def plot_imagenet_true_rho(scores,
                           labels,
                           betas=[
                               0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.
                           ]):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from tqdm.notebook import tqdm

    import scipy
    import seaborn as sns

    from Risk import TopBetaCoverage
    rng = np.random.default_rng(seed=322)

    weights = rng.uniform(size=scores.shape[1])

    rho = TopBetaCoverage()

    print(betas)

    risk_dists = [
        rho(((scores, None), labels), beta)
        for beta in tqdm(betas, 'Compute average risk')
    ]
    fig, ax = plt.figure(figsize=(4, 4)), plt.gca()

    print(risk_dists[0].shape)
    df = pd.DataFrame({
        beta: risk_dist
        for beta, risk_dist in zip(betas, risk_dists)
    })
    new_df = df.melt(id_vars=[],
                     value_vars=df.columns,
                     var_name='$\\widehat{\\beta}$',
                     value_name='risk')
    sns.histplot(data=new_df, hue='$\\widehat{\\beta}$', x='risk')

    risks = [np.mean(risk_dist) for risk_dist in risk_dists]
    print(risks)

    theta = 0.1

    def func(beta):
        return np.mean(rho(((scores, None), labels), beta)) - theta

    res = scipy.optimize.root_scalar(func,
                                     method='brentq',
                                     rtol=1e-4,
                                     bracket=(0, 1))
    assert res.converged, "Didn't converge"

    fig, ax = plt.figure(), plt.gca()
    ax.set_title(f"Imagenet ($\\theta={theta}, \\beta^* = {res.root:.4f}$)")

    ax.plot(betas, risks,
            label='True risk'), ax.set_xlabel('$\\beta$'), ax.set_ylabel(
                'Miscoverage: $\\mathbb{E}[\\rho(X, Y, \\beta)]$')
    # ax.plot(betas, predicted_risks, label='Predicted risk')
    ax.legend()


# In[3]:

# data_dir = 'data/imagenet_no_T'
# scores, labels = load_data(data_dir)


def plot_fn():
    import numpy as np
    plot_imagenet_true_rho(scores, labels, betas=np.linspace(0.1, 1, 10))


# plot_fn()

# OK, so we can see from the histogram the risks are almost all 0! If nearly all risks are 0, that kinda defeats the point of doing any active learning at all really. So let's see what it is if we have a calibrated T.

# In[4]:


def make_probe_data(test_beta, weights, margin, featurize):
    import torch
    Xs = []
    ps = []
    temp_weights = torch.ones(1000)
    for i in range(len(margin)):
        temp_p = torch.zeros(1000)
        temp_p[0] = test_beta + (1 - margin[i])
        temp_p[1] = margin[i]
        ps.append(temp_p.reshape(1, -1))
        Xs.append(
            featurize(temp_p.reshape(1, -1), weights,
                      test_beta).reshape(1, -1))

    X = torch.concatenate(Xs)
    p = torch.concatenate(ps)
    return p, X


def probe(minimax, featurize, scores, labels, budget):
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from Risk import TopBetaCoverage

    betas = np.arange(0.9, 1, 0.01)
    margins = np.arange(0.00, 0.11, 0.005)
    probe_datas = [
        make_probe_data(beta, torch.ones(1000), margins, featurize)
        for beta in betas
    ]
    results = []

    for probe_data in probe_datas:
        (Q, _), cons = minimax._primal_player.predict((probe_data, None))
        results.append(Q.detach().numpy())

    q_vals = np.array(results)
    tbc = TopBetaCoverage()
    cov_set = tbc.beta_coverage_set(torch.Tensor(scores), 0.9)
    miscovers = ~(cov_set[np.arange(cov_set.shape[0]), labels])
    est_miscovers = (torch.Tensor(scores) *
                     (~(cov_set)).int()).sum(dim=-1).detach().numpy()

    fig, ax = plt.figure(), plt.gca()
    #for i in range(0, len(betas), 1):
    i = 0
    beta, q_val = betas[i], q_vals[i]
    max_margin = 1 - beta
    mask = margins <= max_margin
    ax.plot(margins[mask], q_val[mask], label=f"$\\beta={beta:.2f}$")
    twin_ax = ax.twinx()
    _, bins, _ = twin_ax.hist(est_miscovers, alpha=0.5, density=True)

    bin_indices = np.digitize(est_miscovers, bins)

    bin_miscovers = []
    for i in range(1, len(bins)):
        bin_miscovers.append(
            torch.mean(miscovers[bin_indices == i].float()).detach().item())
    print(bin_miscovers)
    ax.stairs(bin_miscovers, bins, color='red')

    twin_ax.set_ylabel('Freq. of est. margins across dataset')
    ax.axhline(budget, linestyle='dashed', color='gray')
    ax.set_xlabel('Est. miscoverage'), ax.set_ylabel('$q(x)')
    ax.set_title(
        f'# of labels: {minimax._primal_player._suml} ({minimax._primal_player._suml / len(scores)} of {len(scores)})'
    )
    ax.legend()


# In[5]:


# This experiment is correct out of the pre-computed results
# other experiment in the results/ dir had bugs and had incorrect performance
def load_debug_results(
        data_dir='results/imagenet_d/imagenet_w_cons_debug_01_'):
    import dill
    import pandas as pd
    import IwUpperMartingale
    from IwUpperMartingale import DummyMartingale
    import Player
    import os

    Player.DummyMinimax = DummyMinimax
    Player.DummyPlayer = DummyPlayer
    with open(f'{data_dir}/run_args.pkl', 'rb') as in_f:
        run_args = dill.load(in_f)

    #data_df = pd.read_csv(f'{data_dir}/last_beta_dist.csv')
    trial_result_dir = f'{data_dir}/trial_results'
    data_df = pd.concat([
        pd.read_csv(f'{trial_result_dir}/{path}')
        for path in os.listdir(trial_result_dir)
    ])
    # return datas, data_df
    return run_args, data_df


# some_debug_df = load_debug_results()
# Results with overlapping std. error in bars in the real coverage of number of labels
# means more trials need to be run

# ## visualization functions

# In[6]:

import numpy as np


def mean_coverage_size(scores, beta, weights):
    from Risk import TopBetaCoverage
    tbc = TopBetaCoverage(w=weights)
    return tbc.beta_coverage_set(scores,
                                 beta).float().sum(dim=-1).mean().item()


def mean_threshold_coverage_size(scores, beta, weights):
    from Risk import WeightedCoverage
    tbc = WeightedCoverage(w=weights)
    return tbc.beta_coverage_set(scores,
                                 beta).float().sum(dim=-1).mean().item()


def in_fill_beta(sumls, betas, max_sumls):
    ext_pairs = []
    for i in range(len(sumls)):
        rg = range(int(sumls[i]),
                   max_sumls + 1 if i == len(sumls) - 1 else int(sumls[i + 1]))
        ext_pairs.extend([(j, betas[i]) for j in rg])
    return ext_pairs


def visualize_beta_hat(  # run_args,
    beta_hat_df,
    fig_kwargs={
        'figsize': (4, 3),
        'dpi': 100
    },
    pointplot_kwargs={
        'errorbar': ('se', 2),
        'capsize': .4,
        'err_kws': {
            'color': ".5",
            'linewidth': 2.5
        },
        'linestyle': 'none'
    },
    alpha=None,
    beta_star=None
    # in_fill_max=2000
):

    sns.set_theme()
    max_label_ct = beta_hat_df['label_ct'].max()
    method_map = {
        'label all': 'all',
        'optimal': 'pretrain',
        'cv only': 'oblivious w/ pretrain c.v.',
        'cv + learn_var': 'learned  w/ pretrain c.v.'
    }
    beta_hat_df = beta_hat_df.replace({'Method': method_map})

    last_beta_df = beta_hat_df[beta_hat_df['label_ct'] == max_label_ct]
    fig, ax = plt.figure(**fig_kwargs), plt.gca()
    ax = sns.pointplot(data=last_beta_df,
                       x='Method',
                       y='$\\widehat{\\beta}$',
                       ax=ax,
                       **pointplot_kwargs)

    if beta_star is not None:
        ax.axhline(beta_star, linestyle='dashed', color='gray')
    plt.xticks(rotation=90)
    fig.tight_layout()

    points = 21
    x_bins = np.linspace(0, max_label_ct, points)
    beta_hat_df['grid_x'] = x_bins[np.digitize(
        beta_hat_df['label_ct'].to_numpy(), x_bins, right=True)]
    # .apply(lambda x: np.searchsorted(x_bins, x, side='left'))
    binned_df = beta_hat_df \
                .sort_values('$\\widehat{\\beta}$') \
                .groupby(['Method', 'grid_x', 'seed']).first() \
                .reset_index()
    binned_df = binned_df.rename(columns={"grid_x": 'Labels queried'})

    fig_2, ax_2 = plt.figure(**fig_kwargs), plt.gca()

    sns.lineplot(
        data=binned_df,
        x='Labels queried',
        y='$\\widehat{\\beta}$',
        hue='Method',
        # err_style='bars',
        # errorbar=("se", 2),
        # err_kws={"color": ".5", "linewidth": 1, 'solid_capstyle': 'round'},
        ax=ax_2)
    sns.move_legend(ax_2, "upper left", bbox_to_anchor=(1, 1))
    if beta_star is not None:
        ax_2.axhline(beta_star, linestyle='dashed', color='gray')

    plt.xticks(rotation=90)
    fig_2.tight_layout()

    res = [(fig, ax), (fig_2, ax_2)]
    if beta_star is not None:
        last_beta_df['Error rate'] = (last_beta_df['$\\widehat{\\beta}$']
                                      < beta_star).astype(int)
        fig_3, ax_3 = plt.figure(**fig_kwargs), plt.gca()
        ax_3 = sns.pointplot(data=last_beta_df,
                             x='Method',
                             y='Error rate',
                             ax=ax_3,
                             **pointplot_kwargs)
        if alpha is not None:
            ax_3.axhline(alpha, linestyle='dashed', color='gray')

        plt.xticks(rotation=90)
        fig_3.tight_layout()
        res.append((fig_3, ax_3))

    return res
    # print(f'Arguments of experiment: {run_args}\n')

    # torch_scores = torch.Tensor(scores)
    # beta_hat_df = beta_hat_df.copy()
    # # beta_hat_df['ACS'] = pd.Series([mean_coverage_size(torch_scores, beta, weights=None) \
    # #                                 for beta in tqdm(beta_hat_df['$\\widehat{\\beta}$'])])

    # if risk_fn is None:
    #     res = beta_hat_df[beta_hat_df['label_ct'] == max_label_ct] \
    #         .groupby(['Method'], as_index=False) \
    #         .agg({'$\\widehat{\\beta}$':['mean','std']})
    #     res.columns = res.columns.get_level_values(1)

    #     res = res.rename(
    #         columns=dict(zip(res.columns, ['Method', 'Mean', 'Std'])))
    #     res['Lower'] = res['Mean'] - (2 * res['Std'])
    #     res['Upper'] = res['Mean'] + (2 * res['Std'])

    #     res['ACS'] = [
    #         mean_coverage_size(torch_scores, beta, weights=None)
    #         for beta in tqdm(res['Mean'], leave=False)
    #     ]
    #     res['ACS_lower'] = [
    #         mean_coverage_size(torch_scores, beta, weights=None)
    #         for beta in res['Lower']
    #     ]
    #     res['ACS_upper'] = [
    #         mean_coverage_size(torch_scores, beta, weights=None)
    #         for beta in res['Upper']
    #     ]
    #     res['ACS_center'] = (res['ACS_lower'] + res['ACS_upper']) / 2
    #     res['ACS_error'] = (res['ACS_upper'] - res['ACS_lower']) / 2

    # else:
    #     avg_cov_fn = risk_fn.data_covsize_curve(torch_scores)
    #     res = beta_hat_df[beta_hat_df['label_ct'] == max_label_ct].copy()
    #     res['ACS'] = [avg_cov_fn(beta) for beta in res['$\\widehat{\\beta}$']]
    #     res = res.groupby(['Method'], as_index=False) \
    #         .agg({'$\\widehat{\\beta}$':['mean','std'],
    #               'ACS': ['mean', 'std']})
    #     # res.columns = res.columns.get_level_values(1)
    #     res.columns = [' '.join(col).strip() for col in res.columns.values]

    #     res = res.rename(columns=dict(
    #         zip(res.columns,
    #             ['Method', 'Mean', 'Std', 'ACS_mean', 'ACS_std'])))
    #     print(res)
    #     res['ACS_center'] = res['ACS_mean']
    #     res['ACS_error'] = 1 * res['ACS_std']

    # Method vs. actual prediction interval size
    # plt.figure(figsize=(4, 4), dpi=dpi_val)
    # ax = plt.gca()
    # res.plot(ax=ax, x='Method', y='ACS_center', kind="scatter")
    # ax.errorbar(x=res['Method'],
    #             y=res['ACS_center'],
    #             yerr=res['ACS_error'],
    #             capsize=4,
    #             color=".5",
    #             linewidth=2.5,
    #             linestyle='none',
    #             elinewidth=2)
    # plt.xticks(rotation=90)

    # Method vs. beta hat

    # Histogram (of trials) of beta hat for each method
    # fig, ax = plt.figure(figsize=(4, 4), dpi=dpi_val), plt.gca()
    # sns.histplot(data=last_beta_df,
    #              hue='Method',
    #              x='$\\widehat{\\beta}$',
    #              bins=20,
    #              ax=ax)
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    # if beta_star is not None:
    #     ax.axvline(beta_star, linestyle='dashed', color='gray')

    # plt.xticks(rotation=90)


def make_figures(out_dir, exp_args, df, beta_star, alpha):
    results = visualize_beta_hat(df, beta_star=beta_star, alpha=alpha)
    fig, ax = results[0]

    fig.savefig(f'{out_dir}/method_v_beta.png', dpi=300)
    fig, ax = results[1]
    fig.savefig(f'{out_dir}/labels_v_beta.png', dpi=300)
    fig, ax = results[2]
    fig.savefig(f'{out_dir}/method_v_error_rate.png', dpi=300)


# coco_data_dir = 'data/coco'
# coco_scores, coco_labels = load_data(coco_data_dir)
# coco_scores.shape, coco_labels.shape

# In[11]:


def plot_coco_true_rho(coco_scores, coco_labels):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from Risk import NegRecall

    rho = NegRecall()
    step = 0.1
    betas = np.arange(0.1, 1. + step, step)

    fig, axes = plt.subplots(2)
    axes[0].hist(coco_scores[coco_labels.astype(bool)])
    axes[1].hist(coco_scores.reshape(-1))

    axes[0].set_xlabel('$p(x)$ of true classes'), axes[0].set_ylabel('Freq.')
    axes[1].set_xlabel('$p(x)$'), axes[1].set_ylabel('Freq.')
    fig.suptitle('COCO-MS Stats')
    fig.tight_layout()

    risk_dists = [
        rho(((coco_scores, None), coco_labels), beta) for beta in betas
    ]
    fig, ax = plt.figure(figsize=(4, 4)), plt.gca()

    print(risk_dists[0].shape)
    df = pd.DataFrame({
        beta: risk_dist
        for beta, risk_dist in zip(betas, risk_dists)
    })
    new_df = df.melt(id_vars=[],
                     value_vars=df.columns,
                     var_name='$\\widehat{\\beta}$',
                     value_name='risk')
    sns.histplot(data=new_df, hue='$\\widehat{\\beta}$', x='risk')
    ax.legend(bbox_to_anchor=(1.1, 1.05))

    risks = [np.mean(risk_dist) for risk_dist in risk_dists]

    fig, ax = plt.figure(), plt.gca()
    ax.plot(betas, risks), ax.set_xlabel('$\\beta$'), ax.set_ylabel(
        'True neg. recall: $\\mathbb{E}[\\rho(X, Y, \\beta)]$')

    print('average scores greater than 1 per example',
          (coco_scores >= 1).sum(axis=-1).mean())
    print('average labels per example', coco_labels.sum(axis=-1).mean())
    print('recall (threshold at 0.5)',
          (((coco_scores >= 0.5) * coco_labels).sum(axis=-1) /
           coco_labels.sum(axis=-1)).mean())
    print('precision (threshold at 0.5)',
          (((coco_scores >= 0.5) * coco_labels).sum(axis=-1) /
           (coco_scores >= 0.5).sum(axis=-1)).mean())
    print(pd.DataFrame({'beta': betas, 'avg. neg. recall': risks}))


# plot_coco_true_rho(coco_scores, coco_labels)
#
# # In[12]:
#
# visualize_beta_hat(
#     *(load_debug_results('results/coco_d/fbml_target_rate=0.6')))


def load_coco_cv_results():
    import dill
    import os
    import re
    import pandas as pd
    import IwUpperMartingale
    from IwUpperMartingale import DummyMartingale
    import Player
    data_dir = 'results/coco_d/big_lam'
    trial_dir = f'{data_dir}/trial_results'
    print(f"Trial dir: {trial_dir}")
    data_df = pd.concat([
        pd.read_csv(f'{trial_dir}/{fname}')
        for fname in os.listdir(trial_dir) if re.fullmatch(r'.*\.csv', fname)
    ],
                        axis=0)
    print(data_df.columns)
    # with open(f'{data_dir}/result_dump.pkl', 'rb') as in_f:
    #    datas = dill.load(in_f)
    f_df = data_df.groupby('seed')['label_ct'].agg(['min', 'max'])
    print(f'Data length: {len(data_df)}')
    bad_seeds = f_df[f_df['min'] < 800].reset_index()['seed']
    print(f'Bad/incomplete seeds/trials: {len(bad_seeds)}')
    data_df = data_df[~(data_df['seed'].isin(bad_seeds))]
    print(f'Data length after rem. inc.: {len(data_df)}')
    # data_df = data_df[~(data_df['Method'].isin(['active (partial)', 'active (partial+q_cv)']))]
    return data_df


def load_parent_dir(path):
    import os
    import pandas as pd
    df = pd.concat([pd.read_csv(f'{path}/{subdir}/trial_results/{filename}') \
        for subdir in os.listdir(path) \
        for filename in os.listdir(f'{path}/{subdir}/trial_results')])
    return os.listdir(path), df


def visualize_q_ablation(df, points=21, start_prop=0, end_prop=1.):
    import seaborn as sns
    import matplotlib.pyplot as plt
    max_label_ct = df['label_ct'].max()
    final_df = df.sort_values('label_ct') \
                .groupby(['Method', 'target_rate', 'seed']) \
                .last() # get largest label_ct
    print(final_df.groupby(['target_rate', 'Method']).mean())

    sns.set_theme()

    sns.pointplot(data=final_df,
                  hue='Method',
                  x='target_rate',
                  y='$\\widehat{\\beta}$')

    # label count vs. beta hat
    if len(set(df['label_ct'])) > 1:
        points = 21
        start, end = start_prop * max_label_ct, end_prop * max_label_ct
        x_bins = np.linspace(start, end, points)
        df['grid_x'] = x_bins[np.digitize(df['label_ct'].to_numpy(),
                                          x_bins,
                                          right=True)]
        binned_df = df \
                    .sort_values('$\\widehat{\\beta}$') \
                    .groupby(['Method', 'grid_x', 'target_rate', 'seed']).first() \
                    .reset_index()

        # fig, ax = plt.figure(figsize=(4,4), dpi=200), plt.gca()

        sns.relplot(
            data=binned_df,
            x='grid_x',
            y='$\\widehat{\\beta}$',
            hue='Method',
            col='target_rate',
            col_wrap=3,
            kind='line'
            # err_style='bars',
            # errorbar=("se", 2),
            # err_kws={"color": ".5", "linewidth": 1, 'solid_capstyle': 'round'},
        )
        plt.xticks(rotation=90)


def load_data_from_results_dir(in_dirs):
    all_dfs = []
    for data_dir in in_dirs:
        trial_dir = f'{data_dir}/trial_results'
        data_dfs = []
        for fname in os.listdir(trial_dir):
            if re.fullmatch(r'.*\.csv', fname):
                small_df = pd.read_csv(f'{trial_dir}/{fname}')
                print(f'fname: {fname}')
                data_dfs.append(small_df)

        all_dfs += data_dfs
    data_df = pd.concat(all_dfs, axis=0)
    print(data_df.columns)
    # f_df = data_df.groupby(['seed', 'method'])['label_ct'].agg(['min', 'max'])
    # print(f'Data length: {len(data_df)}')
    # print(f_df)
    # bad_seeds = f_df[f_df['min'] < 800].reset_index()['seed']
    # print(f'Bad/incomplete seeds/trials: {len(bad_seeds)}')
    # data_df = data_df[~(data_df['seed'].isin(bad_seeds))]
    # print(f'Data length after rem. inc.: {len(data_df)}')
    return data_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir',
                        type=str,
                        action='append',
                        required=True,
                        help="Directory of trial results")
    parser.add_argument('--out_dir',
                        type=str,
                        default='figures/',
                        help="Directoy to output figures")
    parser.add_argument('--beta_star',
                        type=float,
                        default=None,
                        help="beta star value")
    parser.add_argument('--alpha',
                        type=float,
                        default=None,
                        help="alpha value")
    args = parser.parse_args()
    df = load_data_from_results_dir(args.results_dir)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    make_figures(args.out_dir, {},
                 df,
                 beta_star=args.beta_star,
                 alpha=args.alpha)
