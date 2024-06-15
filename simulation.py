import argparse
from copy import copy
import dill
import multiprocess as mp
import os

from cocob import COCOB
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
from tqdm import tqdm

from load_data import load_imagenet_torch_preds
from Features import sim_featurize
from IwUpperMartingale import FullIwUpperMartingale, PartialIwUpperMartingale, FullyObservedUpperMartingale, ShiftedIwUpperMartingale, BaseMartingale
from LogisticRegression import DummyRegressor, MultilabelRegressor, ClassRiskRegressor, CVWrapperRegressor, NegRecallRegressor
from OnlineMinimax import OnlineMinimax, WindowedBeTheLeader
from Player import Player
from Predictor import ConstantPredictor, FixedConstantPredictor, FalsePositivePredictor, LinearRiskPredictor
from Risk import FalsePositiveRho


def test_once(
    seed,
    label_ct,
    theta,
    q_min,
    target_rate,
    beta_dim,
    est_loss_dim,
):

    beta_knots = torch.linspace(0, 1, beta_dim)
    est_loss_knots = torch.linspace(0, 1, est_loss_dim)
    torch.manual_seed(seed)
    np.set_printoptions(2, floatmode='fixed')

    rho = FalsePositiveRho()
    # rho = lambda x, beta: wc_rho(x, beta, is_torch=True)

    feature_ct = (len(beta_knots) + 1) * len(est_loss_knots)

    def makeMinimax(policy, iwmart, cv_predictor=None, optimizer='cocob'):
        from Player import LabellingPolicyPrimalPlayer
        available_params = list(policy.parameters()) + (
            [] if cv_predictor is None else list(cv_predictor.parameters()))
        if len(available_params) > 0:
            opt = COCOB(available_params)
        else:
            opt = None
        sched = None
        primal_player = LabellingPolicyPrimalPlayer(policy=policy,
                                                    q_min=q_min,
                                                    target_rate=target_rate,
                                                    theta=theta,
                                                    rho=rho,
                                                    opt=opt,
                                                    sched=sched,
                                                    iwmart=iwmart,
                                                    cv_predictor=cv_predictor)
        # anything bigger than (1 - Log[2]) should be good enough for max_dual
        dual_player = WindowedBeTheLeader(max_dual=1 - np.log(2),
                                          window_size=256)
        return OnlineMinimax(primal_player=primal_player,
                             dual_player=dual_player)

    def makePluginMinimax(var_predictor, iwmart, cv_predictor=None):
        from Player import RiskPredictorPolicyPrimalPlayer
        primal_player = RiskPredictorPolicyPrimalPlayer(
            var_predictor=var_predictor,
            q_min=q_min,
            target_rate=target_rate,
            iwmart=iwmart,
            cv_predictor=cv_predictor)
        # anything bigger than (1 - Log[2]) should be good enough for max_dual
        dual_player = WindowedBeTheLeader(max_dual=1 - np.log(2),
                                          window_size=256)
        return OnlineMinimax(primal_player=primal_player,
                             dual_player=dual_player)

    base_rate = (target_rate - q_min) / (
        1 - q_min
    )  # to initilize the label policy to be random sampling at the the target rate.

    def make_policy():
        return MultilabelRegressor(in_features=feature_ct,
                                   out_classes=1,
                                   bias=True,
                                   base_rate=base_rate,
                                   in_dim=(len(beta_knots) + 1,
                                           len(est_loss_knots)))

    iwum_kwargs = {
        'rho': rho,
        'theta': theta,
        'q_min': q_min,
        'n_betas': 100,
        'n_beta_min': 0.,
        'alpha': 0.05,
    }

    no_q_kwargs = copy(iwum_kwargs)
    del no_q_kwargs['q_min']

    mm_kwargs = {'q_min': q_min, 'target_rate': target_rate}

    # Budget level oblivious sampling
    obv_minimax = makeMinimax(policy=DummyRegressor(in_features=feature_ct,
                                                    out_classes=1,
                                                    base_rate=base_rate),
                              iwmart=PartialIwUpperMartingale(**iwum_kwargs))

    # Fully oblivious, no budget
    fobv_minimax = OnlineMinimax(primal_player=Player(
        iwmart=FullyObservedUpperMartingale(**no_q_kwargs)),
                                 dual_player=WindowedBeTheLeader(
                                     max_dual=1 - np.log(2), window_size=256))

    optimal_cv_only_minimax = makePluginMinimax(
        var_predictor=None,
        cv_predictor=FalsePositivePredictor(mode='mean'),
        iwmart=PartialIwUpperMartingale(**iwum_kwargs))

    iwmart = PartialIwUpperMartingale(**iwum_kwargs)
    optimal_cv_only_learn_var_minimax = makePluginMinimax(
        var_predictor=LinearRiskPredictor(
            beta_knots=beta_knots,
            risk_knots=est_loss_knots,
            featurize_fn=lambda p, beta: sim_featurize(p, beta, beta_knots,
                                                       est_loss_knots),
            risk_fn=rho,
            pretr_fn=FalsePositivePredictor(mode='mean'),
            n_beta_min=iwum_kwargs['n_beta_min'],
            n_betas=iwum_kwargs['n_betas'],
            iwmart=iwmart),
        cv_predictor=FalsePositivePredictor(mode='mean'),
        iwmart=iwmart)

    # Optimal variance + cv
    optimal_minimax = makePluginMinimax(
        var_predictor=FalsePositivePredictor(mode='var'),
        cv_predictor=FalsePositivePredictor(mode='mean'),
        iwmart=PartialIwUpperMartingale(**iwum_kwargs))

    # Binning based on optimal var + cv, but still has to learn correct values
    #
    iwmart = PartialIwUpperMartingale(**iwum_kwargs)
    learned_minimax = makePluginMinimax(
        var_predictor=LinearRiskPredictor(
            beta_knots=beta_knots,
            risk_knots=est_loss_knots,
            featurize_fn=lambda p, beta: sim_featurize(p, beta, beta_knots,
                                                       est_loss_knots),
            risk_fn=rho,
            pretr_fn=FalsePositivePredictor(mode='var'),
            n_beta_min=iwum_kwargs['n_beta_min'],
            n_betas=iwum_kwargs['n_betas'],
            iwmart=iwmart),
        cv_predictor=LinearRiskPredictor(
            beta_knots=beta_knots,
            risk_knots=est_loss_knots,
            featurize_fn=lambda p, beta: sim_featurize(p, beta, beta_knots,
                                                       est_loss_knots),
            risk_fn=rho,
            pretr_fn=FalsePositivePredictor(mode='mean'),
            n_beta_min=iwum_kwargs['n_beta_min'],
            n_betas=iwum_kwargs['n_betas'],
            iwmart=iwmart),
        iwmart=iwmart)

    # backprop_minimax = makeMinimax(
    #     policy=CVWrapperRegressor(FalsePositivePredictor(mode='var')),
    #     iwmart=PartialIwUpperMartingale(**iwum_kwargs),
    #     cv_predictor=LinearRiskPredictor(
    #         beta_knots=beta_knots,
    #         risk_knots=est_loss_knots,
    #         featurize_fn=lambda p, beta: sim_featurize(p, beta, beta_knots,
    #                                                    est_loss_knots),
    #         risk_fn=rho,
    #         pretr_fn=FalsePostivePredictor(mode='mean'),
    #         n_beta_min=iwum_kwargs['n_beta_min'],
    #         n_betas=iwum_kwargs['n_betas']),
    #     **mm_kwargs)

    # minimaxes, names = [
    #     obv_minimax, fobv_minimax, optimal_minimax, learned_minimax
    # ],
    # ['oblivious', 'label all', 'optimal', 'learned']
    minimaxes, names = [
        obv_minimax, fobv_minimax, optimal_minimax, learned_minimax,
        optimal_cv_only_minimax, optimal_cv_only_learn_var_minimax
    ], [
        'oblivious', 'label all', 'optimal', 'learned', 'cv only',
        'cv + learn_var'
    ]
    # minimaxes, names = [learned_minimax], ['learned']

    sumdses, sumlses, betases = [[0] for _ in range(len(minimaxes))], [
        [0] for _ in range(len(minimaxes))
    ], [[minimax._primal_player._iwmart.curbeta[0]] for minimax in minimaxes]
    assert len(names) == len(minimaxes)

    n = 0
    pbar = tqdm(total=label_ct, leave=False, disable=True)
    min_label_ct = 0

    while True:

        p, y = rho.sample()
        cur_label_cts = []
        for name, minimax, sumds, sumls, betas in list(
                zip(names, minimaxes, sumdses, sumlses, betases)):
            # print(name)
            if minimax._primal_player._suml < label_ct:
                curbeta = minimax._primal_player._iwmart.curbeta[1]
                x = sim_featurize(p.reshape(1, -1), torch.tensor([curbeta]),
                                  beta_knots, est_loss_knots).reshape(1, -1)
                in_data = ((p.reshape(1, -1), x), y.reshape(1, -1))
                minimax.addobs(in_data)
                cur_label_cts.append(minimax._primal_player._suml)
                if minimax._primal_player._iwmart.curbeta[0] < betas[
                        -1] or minimax._primal_player._suml >= label_ct:
                    sumls.append(minimax._primal_player._suml)
                    betas.append(curbeta)
                    sumds.append(n)
        label_ct_delta = min(cur_label_cts) - min_label_ct
        if label_ct_delta > 0:
            min_label_ct += label_ct_delta
            pbar.update(label_ct_delta)
        n += 1
        if np.all(
                np.array(
                    [minimax._primal_player._suml
                     for minimax in minimaxes]) >= label_ct):
            break
    assert np.all(
        np.array([minimax._primal_player._suml
                  for minimax in minimaxes]) >= label_ct), [
                      minimax._primal_player._suml for minimax in minimaxes
                  ]

    return names, minimaxes, sumlses, betases, sumdses


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='Beta CI')

    parser.add_argument('--processes', type=int, default=1)
    parser.add_argument('--out_dir',
                        type=str,
                        default='results/simulation/default')
    parser.add_argument('--trial_start', type=int, default=1)
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--label_ct', type=int, default=2500)
    parser.add_argument('--q_min', type=float, default=0.1)
    parser.add_argument('--theta', type=float, default=0.1)
    parser.add_argument('--target_rate', type=float, default=0.3)
    parser.add_argument('--est_loss_dim', type=int, default=10)
    parser.add_argument('--beta_dim', type=int, default=10)
    args = parser.parse_args()

    out_dir, trial_start, trials, label_ct = args.out_dir, args.trial_start, args.trials, args.label_ct
    processes = args.processes

    run_args = {'out_dir': out_dir, 'trials': trials, 'label_ct': label_ct}
    q_min, theta, target_rate = args.q_min, args.theta, args.target_rate
    problem_params = {
        'q_min': q_min,
        'theta': theta,
        'target_rate': target_rate,
        'beta_dim': args.beta_dim,
        'est_loss_dim': args.est_loss_dim
    }

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    trial_dir = f'{out_dir}/trial_results'
    if not os.path.exists(trial_dir):
        os.makedirs(trial_dir)

    with open(f'{out_dir}/run_args.pkl', 'wb') as out_f:
        dill.dump(run_args, out_f)

    res_list = []

    def run_trial(seed):
        res = test_once(seed=seed, label_ct=label_ct, **problem_params)
        records = []
        for name, minimax, sumls, betas, sumds in zip(*(res)):
            for lc, beta, n in zip(sumls, betas, sumds):
                records.append({
                    'Method':
                    name,
                    '$\\widehat{\\beta}$':
                    beta,
                    'seed':
                    seed,
                    'label_ct':
                    lc,
                    'data_ct':
                    n,
                    'q_hist': [entry['pred'][0][0] for entry in minimax._hist],
                    **problem_params
                })
        trial_df = pd.DataFrame.from_records(records)
        trial_df.to_csv(
            f'{trial_dir}/simulation_lc={label_ct}_seed={seed}.csv')
        return res

    with mp.Pool(processes) as p:
        res_list = list(
            tqdm(p.imap(run_trial, range(trial_start, trials + trial_start)),
                 desc="Trials",
                 total=trials))
    with open(f'{out_dir}/result_dump.pkl', 'wb') as out_f:
        dill.dump(res_list, out_f)
    records = []
    for item in res_list:
        for name, minimax, sumls, betas, _ in zip(*(item)):
            records.append({'Method': name, '$\\widehat{\\beta}$': betas[-1]})
    df = pd.DataFrame.from_records(records)
    df.to_csv(f'{out_dir}/last_beta_dist.csv')
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Method", y="$\\widehat{\\beta}$", data=df, errorbar="sd")
    plt.tight_layout()
    plt.yscale('log')
    plt.savefig(f'{out_dir}/last_beta_dist.png', dpi=300)
