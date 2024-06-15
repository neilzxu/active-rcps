from itertools import product
from copy import copy

from Features import imagenet_featurize
from IwUpperMartingale import FullIwUpperMartingale, PartialIwUpperMartingale, FullyObservedUpperMartingale, ShiftedIwUpperMartingale, BaseMartingale
from LogisticRegression import DummyRegressor, MultilabelRegressor, ClassRiskRegressor, CVWrapperRegressor
from OnlineMinimax import OnlineMinimax, WindowedBeTheLeader
from Player import Player
from Predictor import ConstantPredictor, FixedConstantPredictor, ClassRiskPredictor
from Risk import TopBetaCoverage

from cocob import COCOB
import numpy as np
import torch

torch.set_num_threads(1)
from tqdm import tqdm


def test_once(scores,
              labels,
              seed: float,
              label_ct: int,
              target_rate: float,
              q_min: float,
              theta: float,
              incrs=[],
              weights=None):

    if True:
        torch.manual_seed(seed)
        np.set_printoptions(2, floatmode='fixed')

        wc_rho = TopBetaCoverage() if weights is None else TopBetaCoverage(
            w=weights)
        rho = lambda x, beta: wc_rho(x, beta, is_torch=True)

        feature_ct = 60

        def makeMinimax(policy, q_min, target_rate, iwmart, cv_predictor=None):
            from Player import LabellingPolicyPrimalPlayer
            if len(list(policy.parameters())) > 0:
                opt = COCOB(policy.parameters())
            else:
                opt = None
            sched = None
            primal_player = LabellingPolicyPrimalPlayer(
                policy=policy,
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

        base_rate = (target_rate - q_min) / (
            1 - q_min
        )  # to initilize the label policy to be random sampling at the the target rate.

        def make_policy():
            return MultilabelRegressor(in_features=feature_ct,
                                       out_classes=1,
                                       bias=True,
                                       base_rate=base_rate,
                                       in_dim=(6, 10))

        iwum_kwargs = {
            'rho': rho,
            'theta': theta,
            'q_min': q_min,
            'n_betas': 1000,
            'n_beta_min': 0.7,
            'alpha': 0.05,
        }

        no_q_kwargs = copy(iwum_kwargs)
        del no_q_kwargs['q_min']

        mm_kwargs = {'q_min': q_min, 'target_rate': target_rate}
        minimaxes, names = [], []
        partial_minimax = makeMinimax(
            policy=make_policy(),
            iwmart=PartialIwUpperMartingale(**iwum_kwargs),
            **mm_kwargs)
        minimaxes.append(partial_minimax)
        names.append('active (partial)')

        half_cv_policy = FixedConstantPredictor(constant=0.5)
        half_minimax = makeMinimax(
            policy=make_policy(),
            iwmart=PartialIwUpperMartingale(**iwum_kwargs),
            cv_predictor=half_cv_policy,
            **mm_kwargs)
        minimaxes.append(half_minimax)
        names.append('active (half)')

        shifted_minimax = makeMinimax(
            policy=make_policy(),
            iwmart=ShiftedIwUpperMartingale(**no_q_kwargs),
            **mm_kwargs)
        minimaxes.append(shifted_minimax)
        names.append('active (shifted)')

        full_minimax = makeMinimax(policy=make_policy(),
                                   iwmart=FullIwUpperMartingale(**iwum_kwargs),
                                   **mm_kwargs)
        minimaxes.append(full_minimax)
        names.append('full (partial)')

        const_minimax = makeMinimax(
            policy=DummyRegressor(in_features=feature_ct,
                                  out_classes=1,
                                  base_rate=base_rate),
            iwmart=PartialIwUpperMartingale(**iwum_kwargs),
            **mm_kwargs)
        minimaxes.append(const_minimax)
        names.append('oblivious (partial)')

        cons_cv_policy = ConstantPredictor(
            risk_fn=rho,
            n_betas=iwum_kwargs['n_betas'],
            n_beta_min=iwum_kwargs['n_beta_min'])
        partial_conscv_minimax = makeMinimax(policy=make_policy(),
                                             iwmart=PartialIwUpperMartingale(
                                                 cv_predictor=cons_cv_policy,
                                                 **iwum_kwargs),
                                             cv_predictor=cons_cv_policy,
                                             **mm_kwargs)
        minimaxes.append(partial_conscv_minimax)
        names.append('oblivious (learned)')

        pretrain_label_minimax = makeMinimax(
            policy=ClassRiskRegressor(),
            iwmart=PartialIwUpperMartingale(**iwum_kwargs),
            **mm_kwargs)
        minimaxes.append(pretrain_label_minimax)
        names.append('pretrain (partial)')

        pretrain_cv_predictor = ClassRiskPredictor(w=weights)
        pretrain_cv_minimax = makeMinimax(
            policy=DummyRegressor(in_features=feature_ct,
                                  out_classes=1,
                                  base_rate=base_rate),
            iwmart=PartialIwUpperMartingale(cv_predictor=pretrain_cv_predictor,
                                            **iwum_kwargs),
            cv_predictor=pretrain_cv_predictor,
            **mm_kwargs)
        minimaxes.append(pretrain_cv_minimax)
        names.append('oblivious (pretrain)')

        pretrain_cv_predictor = ClassRiskPredictor(w=weights)
        pretrain_both_minimax = makeMinimax(
            policy=CVWrapperRegressor(pretrain_cv_predictor),
            iwmart=PartialIwUpperMartingale(cv_predictor=pretrain_cv_predictor,
                                            **iwum_kwargs),
            cv_predictor=pretrain_cv_predictor,
            **mm_kwargs)
        minimaxes.append(pretrain_both_minimax)
        names.append('pretrain_var (pretrain)')

        fobv_minimax = OnlineMinimax(primal_player=Player(
            iwmart=FullyObservedUpperMartingale(**no_q_kwargs)),
                                     dual_player=WindowedBeTheLeader(
                                         max_dual=1 - np.log(2),
                                         window_size=256))
        minimaxes.append(fobv_minimax)
        names.append('sample everything')

    randperm = torch.randperm(len(scores))
    rand_scores, rand_labels = torch.Tensor(scores[randperm]), torch.Tensor(
        labels[randperm]).int()

    # minimaxes = [
    #     partial_minimax, partial_conscv_minimax, half_minimax, shifted_minimax,
    #     const_minimax, fobv_minimax
    # ]
    # minimaxes = [const_minimax]
    sumlses, betases, sumdses = [[0] for _ in range(len(minimaxes))], [
        [minimax._primal_player._iwmart.curbeta[0]] for minimax in minimaxes
    ], [[0] for _ in range(len(minimaxes))]

    for n in tqdm(range(len(scores)),
                  desc="Running samples",
                  leave=False,
                  disable=True):

        p, y = rand_scores[n], rand_labels[n]

        for minimax, sumls, betas, sumds in list(
                zip(minimaxes, sumlses, betases, sumdses)):
            # Only sample if label count has not been met
            if minimax._primal_player._suml < label_ct:
                curbeta = minimax._primal_player._iwmart.curbeta[0]
                x = imagenet_featurize(p.reshape(1, -1), torch.ones(1000),
                                       curbeta).reshape(1, -1)
                in_data = ((p.reshape(1, -1), x), y.reshape(1, -1))
                minimax.addobs(in_data)

                if curbeta < betas[
                        -1] or minimax._primal_player._suml >= label_ct:
                    sumls.append(minimax._primal_player._suml)
                    betas.append(curbeta)
                    sumds.append(n)
        # Early stopping if label count is met.
        if np.all(
                np.array(
                    [minimax._primal_player._suml
                     for minimax in minimaxes]) >= label_ct):
            break
    # Check if data was large enough such that all methods got enough labels to meet label threshold.
    assert np.all(
        np.array([minimax._primal_player._suml
                  for minimax in minimaxes]) >= label_ct), [
                      minimax._primal_player._suml for minimax in minimaxes
                  ]
    return names, minimaxes, sumlses, betases, sumdses


if __name__ == '__main__':
    import argparse
    import dill
    import os
    import matplotlib.pyplot as plt
    import multiprocess as mp
    import pandas as pd
    import seaborn as sns

    from load_data import load_imagenet_torch_preds

    parser = argparse.ArgumentParser(prog='Beta CI')

    parser.add_argument('--data_dir', type=str, default='data/imagenet_no_T')
    parser.add_argument('--out_dir', type=str, default='results/imagenet/new')
    parser.add_argument('--trial_start', type=int, default=1)
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--label_ct', type=int, default=2000)
    parser.add_argument('--weight_path', type=str, default=None)
    parser.add_argument('--processes', type=int, default=1)
    parser.add_argument('--q_min', type=float, default=0.1)
    parser.add_argument('--theta', type=float, default=0.1)
    parser.add_argument('--target_rate', type=float, default=0.3)
    args = parser.parse_args()

    data_dir, out_dir, trials, label_ct, weight_path = args.data_dir, args.out_dir, args.trials, args.label_ct, args.weight_path
    trial_start = args.trial_start

    q_min, theta, target_rate = args.q_min, args.theta, args.target_rate

    processes = args.processes
    print(f'Processes {processes}')
    print(f'Torch thread_ct = {torch.get_num_threads()}')
    # set torch to 1 so I can parallel through processes
    print(f'Torch thread_ct (should be 1) = {torch.get_num_threads()}')

    scores, labels = load_imagenet_torch_preds(data_dir)
    print("Imagenet data size", scores.shape, labels.shape)

    weights = None if weight_path is None else np.load(args.weight_path)

    # run_args = {
    #     'data_dir': data_dir,
    #     'out_dir': out_dir,
    #     'trials': trials,
    #     'label_ct': label_ct,
    #     'weight_path': weight_path,
    #     'weights': weights
    # }
    run_args = vars(args)
    print(run_args)
    trial_dir = f'{out_dir}/trial_results'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(trial_dir):
        os.makedirs(trial_dir)

    with open(f'{out_dir}/run_args.pkl', 'wb') as out_f:
        dill.dump(run_args, out_f)

    rng = np.random.default_rng(seed=322)
    weights = rng.uniform(size=scores.shape[1])
    with open(f'{out_dir}/weights.npy', 'wb') as out_f:
        np.save(out_f, weights)

    def run_trial(seed):
        res = test_once(scores,
                        labels,
                        seed=seed,
                        label_ct=label_ct,
                        theta=theta,
                        q_min=q_min,
                        target_rate=target_rate,
                        weights=None)
        records = []
        for name, minimax, sumls, betas, sumds in zip(*(res)):
            for lc, beta, n in zip(sumls, betas, sumds):
                records.append({
                    'Method': name,
                    '$\\widehat{\\beta}$': beta,
                    'seed': seed,
                    'label_ct': lc,
                    'data_ct': n,
                    'theta': theta,
                    'q_min': q_min,
                    'target_rate': target_rate
                })
        trial_df = pd.DataFrame.from_records(records)
        trial_df.to_csv(f'{trial_dir}/imagenet_lc={label_ct}_seed={seed}.csv')
        return res

    with mp.Pool(processes) as p:
        res_list = list(
            tqdm(p.imap(run_trial, range(trial_start, trials + trial_start)),
                 desc="Trials",
                 total=trials))

    records = []
    for item in res_list:
        for name, minimax, sumls, betas, sumds in zip(*(item)):
            records.append({'Method': name, '$\\widehat{\\beta}$': betas[-1]})
    df = pd.DataFrame.from_records(records)
    df.to_csv(f'{out_dir}/last_beta_dist.csv')
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Method", y="$\\widehat{\\beta}$", data=df, errorbar="sd")
    plt.tight_layout()
    plt.yscale('log')
    plt.savefig(f'{out_dir}/last_beta_dist.png', dpi=300)
