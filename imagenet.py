def test_once(scores, labels, seed, label_ct, weights=None):
    from itertools import product
    from cocob import COCOB
    from IwUpperMartingale import FullIwUpperMartingale, PartialIwUpperMartingale, FullyObservedUpperMartingale, ShiftedIwUpperMartingale, BaseMartingale
    from OnlineMinimax import OnlineMinimax, WindowedBeTheLeader
    from LogisticRegression import MultilabelRegressor
    from Risk import TopBetaCoverage
    from Features import featurize
    import numpy as np
    import torch
    from tqdm import tqdm
    import multiprocess as mp
    from Player import Player

    if True:
        torch.manual_seed(seed)
        np.set_printoptions(2, floatmode='fixed')

        wc_rho = TopBetaCoverage() if weights is None else wc_rho(w=weights)
        rho = lambda x, beta: wc_rho(x, beta, is_torch=True)
        theta = 0.1
        q_min = 0.1
        target_rate = 0.3

        feature_ct = 60

        def makeMinimax(policy,
                        lr,
                        q_min,
                        target_rate,
                        iwmart,
                        optimizer='adam'):
            from Player import LabellingPolicyPrimalPlayer
            if optimizer == 'adam':
                opt = torch.optim.Adam(policy.parameters(), lr=lr)
                sched = torch.optim.lr_scheduler.LambdaLR(
                    opt, lr_lambda=lambda t: 1)  # (1+t/1000)**(-0.5))
            else:
                opt = COCOB(policy.parameters())
                sched = None
            primal_player = LabellingPolicyPrimalPlayer(
                policy=policy,
                q_min=q_min,
                target_rate=target_rate,
                theta=theta,
                rho=rho,
                opt=opt,
                sched=sched,
                iwmart=iwmart)
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
            lr=5e-3,
            q_min=q_min,
            target_rate=target_rate,
            iwmart=PartialIwUpperMartingale(rho=rho,
                                            theta=theta,
                                            q_min=q_min,
                                            n_betas=100,
                                            alpha=0.05),
            optimizer='cocob'
        )

        shifted_minimax = makeMinimax(
            policy=MultilabelRegressor(in_features=feature_ct,
                                       out_classes=1,
                                       bias=True,
                                       base_rate=base_rate),
            lr=5e-3,
            q_min=q_min,
            target_rate=target_rate,
            iwmart=ShiftedIwUpperMartingale(rho=rho,
                                            theta=theta,
                                            n_betas=100,
                                            alpha=0.05),
            optimizer='cocob'
        )


        full_minimax = makeMinimax(
            policy=MultilabelRegressor(in_features=feature_ct,
                                       out_classes=1,
                                       bias=True,
                                       base_rate=base_rate),
            lr=5e-3,
            q_min=q_min,
            target_rate=target_rate,
            iwmart=FullIwUpperMartingale(rho=rho,
                                         theta=theta,
                                         q_min=q_min,
                                         n_betas=100,
                                         alpha=0.05),
            optimizer='cocob'
        )
        const_minimax = makeMinimax(
            policy=MultilabelRegressor(in_features=feature_ct,
                                       out_classes=1,
                                       bias=False),
            lr=0,
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
        partial_minimax, shifted_minimax, full_minimax,
        const_minimax, fobv_minimax
    ]
    sumlses, betases = [[0] for _ in range(len(minimaxes))
                        ], [[minimax._primal_player._iwmart.curbeta[0]]
                            for minimax in minimaxes]
    names = [
        'active (partial)', 'active (shifted)', 'active (full)',
        'oblivious (partial)', 'sample everything'
    ]
    for n in tqdm(range(len(scores)), desc="Running samples", leave=False, disable=True):

        p, y = rand_scores[n], rand_labels[n]

        for minimax, sumls, betas in list(zip(minimaxes, sumlses, betases)):
            # Only sample if label count has not been met
            if minimax._primal_player._suml < label_ct:
                curbeta = minimax._primal_player._iwmart.curbeta[0]
                x = featurize(p.reshape(1, -1), torch.ones(1000),
                              curbeta).reshape(1, -1)
                in_data = ((p.reshape(1, -1), x), y.reshape(1, -1))
                minimax.addobs(in_data)

                if curbeta < betas[
                        -1] or minimax._primal_player._suml >= label_ct:
                    sumls.append(minimax._primal_player._suml)
                    betas.append(curbeta)
        # Early stopping if label count is met.
        if np.all(
                np.array(
                    [minimax._primal_player._suml
                     for minimax in minimaxes]) >= label_ct):
            break
    # Check if data was large enough such that all methods got enough labels to meet label threshold.
    assert np.all(np.array([minimax._primal_player._suml for minimax in minimaxes]) >= label_ct), [minimax._primal_player._suml for minimax in minimaxes]
    return names, minimaxes, sumlses, betases


if __name__ == '__main__':
    import argparse
    import dill
    import os
    import matplotlib.pyplot as plt
    import multiprocess as mp
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from tqdm import tqdm

    from load_data import load_imagenet_torch_preds

    parser = argparse.ArgumentParser(
                    prog='Beta CI')

    parser.add_argument('--data_dir', type=str, default='data/imagenet_no_T')
    parser.add_argument('--out_dir', type=str, default='results/imagenet/new')
    parser.add_argument('--trial_start', type=int, default=1)
    parser.add_argument('--trials', type=int, default=1000)
    parser.add_argument('--label_ct', type=int, default=2000)
    parser.add_argument('--weight_path', type=str, default=None)
    parser.add_argument('--processes', type=int, default=1)

    args = parser.parse_args()

    data_dir, out_dir, trials, label_ct, weight_path = args.data_dir, args.out_dir, args.trials, args.label_ct, args.weight_path
    trial_start = args.trial_start
    processes = args.processes

    scores, labels = load_imagenet_torch_preds(data_dir)
    print("Imagenet data size", scores.shape, labels.shape)

    weights = None if weight_path is None else np.load(args.weight_path)

    run_args = {
        'data_dir': data_dir,
        'out_dir': out_dir,
        'trials': trials,
        'label_ct': label_ct,
        'weight_path': weight_path,
        'weights': weights
    }
    trial_dir = f'{out_dir}/trial_results'
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(trial_dir):
        os.makedirs(trial_dir)

    with open(f'{out_dir}/run_args.pkl', 'wb') as out_f:
        dill.dump(run_args, out_f)

    
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
        trial_df.to_csv(f'{trial_dir}/imagenet_lc={label_ct}_seed={seed}.csv')
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
