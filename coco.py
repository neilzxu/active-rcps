import argparse
from copy import copy
import dill
import multiprocess as mp
import os

from load_data import load_imagenet_torch_preds

from Features import negrec_featurize
from IwUpperMartingale import FullIwUpperMartingale, PartialIwUpperMartingale, FullyObservedUpperMartingale, ShiftedIwUpperMartingale, BaseMartingale
from LogisticRegression import DummyRegressor, MultilabelRegressor, ClassRiskRegressor, CVWrapperRegressor, NegRecallRegressor
from OnlineMinimax import OnlineMinimax, WindowedBeTheLeader
from Player import Player
from Predictor import ConstantPredictor, FixedConstantPredictor, ClassRiskPredictor, NegRecallPredictor
from Risk import NegRecall

from cocob import COCOB
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

torch.set_num_threads(1)
from tqdm import tqdm


def test_once(
    scores,
    labels,
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

    def featurize(probs, beta):
        return negrec_featurize(probs, beta, beta_knots,
                                est_loss_knots).reshape(1, -1)
        # return probs.reshape(1, -1)

    class LabelPolicyWrapper:
        from Features import negrec_featurize
        """Takes an instance of a policy used for labeling and transforms it
        into a control variate predictor."""

        def __init__(self,
                     label_policy,
                     risk_fn,
                     q_min: float,
                     has_update=False,
                     beta_knots=beta_knots.clone().detach(),
                     est_loss_knots=est_loss_knots.clone().detach()):
            self._label_policy = label_policy
            self._q_min = q_min
            self._risk_fn = risk_fn

            self._beta_knots = beta_knots
            self._est_loss_knots = est_loss_knots
            self._has_update = has_update

            self._opt = COCOB(
                label_policy.parameters()) if self._has_update else None

        def _make_features(self, P, betas):
            if not isinstance(betas, np.ndarray) or isinstance(betas, list):
                betas = [betas]
            X = torch.cat([
                negrec_featurize(P, float(beta), self._beta_knots,
                                 self._est_loss_knots) for beta in betas
            ])
            assert torch.all(~torch.isnan(X))
            return X

        def __call__(self, PX, betas):

            # atm, our featurizers are put beta into x, so we won't use betas
            P, X = PX
            X = self._make_features(P, betas)
            res = self._q_min + ((1 - self._q_min) * self._label_policy(X))
            assert not torch.isnan(res)
            return res.squeeze(1)

        def min_val(self):
            return 0

        def max_val(self):
            return 1

        def update(self, PXY, betas):
            """Compute risk for each value of beta considered in the CS and
            update regressor based on the average."""
            if self._has_update:
                # print('Updating predictor')
                risks = self._risk_fn(PXY, betas)
                loss = (risks - self(PXY, betas)).square().mean()
                self._opt.zero_grad()
                loss.backward()
                self._opt.step()

        def parameters(self):
            return []

    if True:  # set namespace (?)
        torch.manual_seed(seed)
        np.set_printoptions(2, floatmode='fixed')

        wc_rho = NegRecall()
        rho = lambda x, beta: wc_rho(x, beta, is_torch=True)

        feature_ct = (len(beta_knots) + 1) * len(est_loss_knots)

        def makeMinimax(policy,
                        q_min,
                        target_rate,
                        iwmart,
                        cv_predictor=None,
                        optimizer='cocob'):
            from Player import LabellingPolicyPrimalPlayer
            available_params = list(
                policy.parameters()) + ([] if cv_predictor is None else list(
                    cv_predictor.parameters()))
            if len(available_params) > 0:
                opt = COCOB(available_params)
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
            policy=NegRecallRegressor(),
            iwmart=PartialIwUpperMartingale(**iwum_kwargs),
            **mm_kwargs)
        minimaxes.append(pretrain_label_minimax)
        names.append('pretrain (partial)')

        pretrain_cv_predictor = NegRecallPredictor()
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

        pretrain_cv_predictor = NegRecallPredictor()
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
    # randperm works as expected

    sumdses, sumlses, betases = [[0] for _ in range(len(minimaxes))], [
        [0] for _ in range(len(minimaxes))
    ], [[minimax._primal_player._iwmart.curbeta[0]] for minimax in minimaxes]
    assert len(names) == len(minimaxes)

    for n in tqdm(range(len(scores)),
                  desc="Running samples",
                  leave=False,
                  disable=True):

        p, y = rand_scores[n], rand_labels[n]

        for name, minimax, sumds, sumls, betas in list(
                zip(names, minimaxes, sumdses, sumlses, betases)):
            # print(name)
            if minimax._primal_player._suml < label_ct:
                curbeta = minimax._primal_player._iwmart.curbeta[1]
                x = featurize(p.reshape(1, -1), curbeta).reshape(1, -1)
                in_data = ((p.reshape(1, -1), x), y.reshape(1, -1))
                minimax.addobs(in_data)

                if minimax._primal_player._iwmart.curbeta[0] < betas[
                        -1] or minimax._primal_player._suml >= label_ct:
                    sumls.append(minimax._primal_player._suml)
                    betas.append(curbeta)
                    sumds.append(n)

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
    parser.add_argument('--data_dir', type=str, default='data/coco')
    parser.add_argument('--out_dir', type=str, default='results/coco')
    parser.add_argument('--trial_start', type=int, default=1)
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--label_ct', type=int, default=800)
    parser.add_argument('--q_min', type=float, default=0.1)
    parser.add_argument('--theta', type=float, default=0.1)
    parser.add_argument('--target_rate', type=float, default=0.3)
    parser.add_argument('--est_loss_dim', type=int, default=20)
    parser.add_argument('--beta_dim', type=int, default=20)
    args = parser.parse_args()

    data_dir, out_dir, trial_start, trials, label_ct = args.data_dir, args.out_dir, args.trial_start, args.trials, args.label_ct
    processes = args.processes

    scores, labels = load_imagenet_torch_preds(data_dir)
    print("COCO MS data size", scores.shape, labels.shape)

    run_args = {
        'data_dir': data_dir,
        'out_dir': out_dir,
        'trials': trials,
        'label_ct': label_ct
    }
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
        res = test_once(scores,
                        labels,
                        seed=seed,
                        label_ct=label_ct,
                        **problem_params)
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
        trial_df.to_csv(f'{trial_dir}/coco_lc={label_ct}_seed={seed}.csv')
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
