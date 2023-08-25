def test_once(scores, labels, seed, label_ct):
    from itertools import product
    from cocob import COCOB
    from IwUpperMartingale import FullIwUpperMartingale, PartialIwUpperMartingale, FullyObservedUpperMartingale, ShiftedIwUpperMartingale, BaseMartingale
    from OnlineMinimax import OnlineMinimax, WindowedBeTheLeader
    from LogisticRegression import MultilabelRegressor
    from Risk import NegRecall
    from Features import negrec_featurize
    import numpy as np
    import torch
    from tqdm import tqdm
    from Player import Player
    
    
    
    class LabelPolicyWrapper:
        """Takes an instance of a policy used for labeling
        and transforms it into a control variate predictor"""
        def __init__(self, label_policy, q_min: float):
            self._label_policy = label_policy
            self._q_min = q_min
        def __call__(self, x, betas):
            # atm, our featurizers are put beta into x, so we won't use betas
            return self._q_min + ((1 - self._q_min) * self._label_policy(x)).squeeze(1)
        def update(self, PXY):
            (P, X), Y = PXY

    if True: # set namespace (?)
        torch.manual_seed(seed)
        np.set_printoptions(2, floatmode='fixed')

        wc_rho = NegRecall()
        rho = lambda x, beta: wc_rho(x, beta, is_torch=True)
        theta = 0.1
        q_min = 0.1
        target_rate = 0.3
        
        beta_knots = torch.linspace(0, 1, 20)
        est_loss_knots = torch.linspace(0, 1, 20)

        feature_ct = (len(beta_knots) + 1) * len(est_loss_knots)

        def makeMinimax(policy, q_min, target_rate, iwmart, cv_predictor=None, optimizer='cocob'):
            from Player import LabellingPolicyPrimalPlayer
            opt = COCOB(policy.parameters())
            if cv_predictor is None:
                cv_opt = None 
            else:
                cv_opt = COCOB(cv_predictor.parameters())
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
                cv_predictor=cv_predictor,
                cv_opt=cv_opt)
            # anything bigger than (1 - Log[2]) should be good enough for max_dual
            dual_player = WindowedBeTheLeader(max_dual=1 - np.log(2),
                                              window_size=256)
            return OnlineMinimax(primal_player=primal_player,
                                 dual_player=dual_player)

        base_rate = (target_rate - q_min) / (1 - q_min)

        partial_minimax = makeMinimax(
            policy=MultilabelRegressor(in_features=feature_ct,
                                       out_classes=1,
                                       bias=True,
                                       base_rate=base_rate),
            q_min=q_min,
            target_rate=target_rate,
            iwmart=PartialIwUpperMartingale(rho=rho,
                                            theta=theta,
                                            q_min=q_min,
                                            n_betas=100,
                                            alpha=0.05),
        )
        
        
        mlr = MultilabelRegressor(in_features=feature_ct,
                                       out_classes=1,
                                       bias=True,
                                       base_rate=base_rate)
        partial_qcv_minimax = makeMinimax(
            policy=mlr,
            q_min=q_min,
            target_rate=target_rate,
            iwmart=PartialIwUpperMartingale(rho=rho,
                                            theta=theta,
                                            q_min=q_min,
                                            cv_predictor=LabelPolicyWrapper(label_policy=mlr, q_min=q_min),
                                            cv_min=q_min,
                                            n_betas=100,
                                            alpha=0.05),
        )
        
        cv_mlr = MultilabelRegressor(in_features=feature_ct,
                                       out_classes=1,
                                       bias=True,
                                       base_rate=0)
        partial_sepcv_minimax = makeMinimax(
            policy=MultilabelRegressor(in_features=feature_ct,
                                       out_classes=1,
                                       bias=True,
                                       base_rate=base_rate),
            q_min=q_min,
            target_rate=target_rate,
            iwmart=PartialIwUpperMartingale(rho=rho,
                                            theta=theta,
                                            q_min=q_min,
                                            cv_predictor=LabelPolicyWrapper(cv_mlr, q_min=0),
                                            cv_min=0,
                                            n_betas=100,
                                            alpha=0.05),
        )

        shifted_minimax = makeMinimax(
            policy=MultilabelRegressor(in_features=feature_ct,
                                       out_classes=1,
                                       bias=True,
                                       base_rate=base_rate),
            q_min=q_min,
            target_rate=target_rate,
            iwmart=ShiftedIwUpperMartingale(rho=rho,
                                            theta=theta,
                                            n_betas=100,
                                            alpha=0.05),
        )

        full_minimax = makeMinimax(
            policy=MultilabelRegressor(in_features=feature_ct,
                                       out_classes=1,
                                       bias=True,
                                       base_rate=base_rate),
            q_min=q_min,
            target_rate=target_rate,
            iwmart=FullIwUpperMartingale(rho=rho,
                                         theta=theta,
                                         q_min=q_min,
                                         n_betas=100,
                                         alpha=0.05),
        )
        const_minimax = makeMinimax(
            policy=MultilabelRegressor(in_features=feature_ct,
                                       out_classes=1,
                                       bias=False),
            q_min=q_min,
            target_rate=target_rate,
            iwmart=PartialIwUpperMartingale(rho=rho,
                                            theta=theta,
                                            q_min=q_min,
                                            n_betas=100,
                                            alpha=0.05),
        )
        fobv_minimax = OnlineMinimax(
            primal_player=Player(iwmart=FullyObservedUpperMartingale(
                rho=rho, theta=theta, n_betas=100, alpha=0.05)),
            dual_player=WindowedBeTheLeader(max_dual=1 - np.log(2),
                                            window_size=256))

    randperm = torch.randperm(len(scores))
    rand_scores, rand_labels = torch.Tensor(scores[randperm]), torch.Tensor(
        labels[randperm]).int()

    minimaxes = [
        partial_minimax, shifted_minimax, full_minimax, const_minimax,
        fobv_minimax
    ]
    sumlses, betases = [[0] for _ in range(len(minimaxes))
                        ], [[minimax._primal_player._iwmart.curbeta[0]]
                            for minimax in minimaxes]

    names = [
        'active (partial)', 'active (shifted)', 'active (full)',
        'oblivious (partial)', 'sample everything'
    ]
    
    
    for n in tqdm(range(len(scores)),
                  desc="Running samples",
                  leave=False,
                  disable=True):

        p, y = rand_scores[n], rand_labels[n]

        for minimax, sumls, betas in list(zip(minimaxes, sumlses, betases)):
            if minimax._primal_player._suml < label_ct:
                curbeta = minimax._primal_player._iwmart.curbeta[0]
                x = negrec_featurize(p.reshape(1, -1), curbeta, beta_knots, est_loss_knots).reshape(1, -1)
                in_data = ((p.reshape(1, -1), x), y.reshape(1, -1))
                minimax.addobs(in_data)

                if curbeta < betas[
                        -1] or minimax._primal_player._suml >= label_ct:
                    sumls.append(minimax._primal_player._suml)
                    betas.append(curbeta)
        if np.all(
                np.array(
                    [minimax._primal_player._suml
                     for minimax in minimaxes]) >= label_ct):
            break
    # assert np.all(np.array([minimax._primal_player._suml for minimax in minimaxes]) >= label_ct), [minimax._primal_player._suml for minimax in minimaxes]
    return names, minimaxes, sumlses, betases


if __name__ == '__main__':
    import argparse
    import dill
    import os
    import matplotlib.pyplot as plt
    import multiprocess as mp
    import pandas as pd
    import seaborn as sns
    from tqdm import tqdm

    from load_data import load_imagenet_torch_preds

    parser = argparse.ArgumentParser(prog='Beta CI')

    parser.add_argument('--processes', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='data/coco')
    parser.add_argument('--out_dir', type=str, default='results/coco')
    parser.add_argument('--trial_start', type=int, default=1)
    parser.add_argument('--trials', type=int, default=600)
    parser.add_argument('--label_ct', type=int, default=800)
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

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    trial_dir = f'{out_dir}/trial_results'
    if not os.path.exists(trial_dir):
        os.makedirs(trial_dir)

    with open(f'{out_dir}/run_args.pkl', 'wb') as out_f:
        dill.dump(run_args, out_f)

    res_list = []

    def run_trial(seed):
        res = test_once(scores, labels, seed=seed, label_ct=label_ct)
        records = []
        for name, minimax, sumls, betas in zip(*(res)):
            records.append({
                'Method': name,
                '$\\widehat{\\beta}$': betas[-1],
                'seed': seed,
                'label_ct': minimax._primal_player._suml
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
        for name, minimax, sumls, betas in zip(*(item)):
            records.append({'Method': name, '$\\widehat{\\beta}$': betas[-1]})
    df = pd.DataFrame.from_records(records)
    df.to_csv(f'{out_dir}/last_beta_dist.csv')
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Method", y="$\\widehat{\\beta}$", data=df, errorbar="sd")
    plt.tight_layout()
    plt.yscale('log')
    plt.savefig(f'{out_dir}/last_beta_dist.png', dpi=300)
