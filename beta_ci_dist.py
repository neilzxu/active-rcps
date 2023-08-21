def test_once(scores, labels, seed, label_ct):
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

        wc_rho = TopBetaCoverage()
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
        )

        full_cocob_minimax = makeMinimax(
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
            optimizer='cocob')

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
        partial_minimax, shifted_minimax, full_cocob_minimax, full_minimax,
        const_minimax, fobv_minimax
    ]
    sumlses, betases = [[0] for _ in range(len(minimaxes))
                        ], [[minimax._primal_player._iwmart.curbeta[0]]
                            for minimax in minimaxes]
    names = [
        'active (partial)', 'active (shifted)', 'active (full)',
        'active (full+cocob)', 'oblivious (partial)', 'sample everything'
    ]
    for n in tqdm(range(len(scores)), desc="Running samples", leave=False):

        p, y = rand_scores[n], rand_labels[n]

        for minimax, sumls, betas in list(zip(minimaxes, sumlses, betases)):
            if minimax._primal_player._suml < label_ct:
                curbeta = minimax._primal_player._iwmart.curbeta[0]
                x = featurize(p.reshape(1, -1), torch.ones(1000),
                              curbeta).reshape(1, -1)
                in_data = ((p.reshape(1, -1), x), y.reshape(1, -1))
                minimax.addobs(in_data)

                if curbeta < betas[
                        -1] or minimax._primal_player._suml == label_ct:
                    sumls.append(minimax._primal_player._suml)
                    betas.append(curbeta)
        if np.all(
                np.array(
                    [minimax._primal_player._suml
                     for minimax in minimaxes]) >= label_ct):
            break

    return names, minimaxes, sumlses, betases


if __name__ == '__main__':
    import dill
    import os
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from tqdm import tqdm

    from load_data import load_imagenet_torch_preds

    data_dir = 'data/imagenet_no_T'
    scores, labels = load_imagenet_torch_preds(data_dir)
    print("Imagenet data size", scores.shape, labels.shape)

    out_dir = 'results/beta_est_600_trials'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    trials = 600
    label_ct = 2000
    res_list = []
    for seed in tqdm(range(1, trials + 1), desc="Trials"):
        res = test_once(scores, labels, seed=seed, label_ct=label_ct)
        res_list.append(res)
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
