
def test_once(scores, labels, seed, step=10, run_length=1000, alpha=0.05):
    from itertools import product
    from IwUpperMartingale import DummyMartingale     
    from Risk import TopBetaCoverage
    from OnlineMinimax import OnlineMinimax, WindowedBeTheLeader
    from LogisticRegression import MultilabelRegressor, DummyRegressor
    import numpy as np
    import torch
    from tqdm import tqdm
    from Features import featurize
            
        
    if True:
        torch.manual_seed(seed)
        np.set_printoptions(2, floatmode='fixed')

        wc_rho = TopBetaCoverage()
        def rho(x, beta):
            return wc_rho(x, beta, is_torch=True)
        # rho = lambda x, beta: 
        theta = 0.1
        q_min = 0.1
        target_rate = 0.5
        init_beta = 0.90
        
        feature_ct = 60
        
        
        class DummyPlayer(object):
            def __init__(self, *, iwmart):
                self._iwmart = iwmart
                self._suml = 0

            def addobs(self, x):
                (P, X), Y = x
                with torch.no_grad():
                    for pj, xj, yj, qj, lj in zip(P, X, Y, torch.ones(Y.shape[0]), torch.ones(Y.shape[0])):
                        self._iwmart.addobs(x=((pj, xj), yj), q=qj.item(), l=lj)
                        self._suml += Y.shape[0]
            
        class DummyMinimax(object):
            def __init__(self, *, player):
                self._primal_player = player
            def addobs(self, x):
                return self._primal_player.addobs(x)
            

                
        def makeMinimax(policy, lr, q_min, target_rate, iwmart):
            from Player import LabellingPolicyPrimalPlayer
            opt = torch.optim.Adam(policy.parameters(), lr=lr)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda t:1) # (1+t/1000)**(-0.5))

            primal_player = LabellingPolicyPrimalPlayer(policy=policy, 
                                                        q_min=q_min,
                                                        target_rate=target_rate,
                                                        theta=theta,
                                                        rho=rho,
                                                        opt=opt,
                                                        sched=sched,
                                                        iwmart=iwmart)
            # anything bigger than (1 - Log[2]) should be good enough for max_dual
            dual_player = WindowedBeTheLeader(max_dual=1 - np.log(2), window_size=256)
            return OnlineMinimax(primal_player=primal_player, dual_player=dual_player)        
        
        """
        const_minimax = makeMinimax(policy=MultilabelRegressor(in_features=feature_ct, out_classes=1, bias=False, \ 
                                                               init_weight=torch.zeros(feature_ct), \
                                                               init_bias=((target_rate - q_min) / (1 - q_min))),
                                       lr=0,
                                       q_min=q_min, 
                                       target_rate=target_rate,
                                       iwmart=DummyMartingale(rho=rho, theta=theta, q_min=q_min, n_betas=1000, alpha=0.05),
                                      )
        """
        full_obv_minimax = DummyMinimax(player=DummyPlayer(iwmart=DummyMartingale(rho=rho, theta=theta, q_min=q_min, alpha=alpha,
                                                              init_beta=0.9, xi_fn='fully_observed')))
        rand_init_minimax = makeMinimax(policy=MultilabelRegressor(in_features=feature_ct, out_classes=1, bias=True),
                                       lr=5e-3,
                                       q_min=q_min, 
                                       target_rate=target_rate,
                                       iwmart=DummyMartingale(rho=rho, theta=theta, q_min=q_min, alpha=alpha,
                                                              init_beta=init_beta, xi_fn='full')
                                      )
        uni_init_minimax = makeMinimax(policy=MultilabelRegressor(in_features=feature_ct, out_classes=1, bias=True, 
                                                               init_weight=torch.zeros(feature_ct),
                                                               init_bias=((target_rate - q_min) / (1 - q_min))),
                                       lr=5e-3,
                                       q_min=q_min, 
                                       target_rate=target_rate,
                                       iwmart=DummyMartingale(rho=rho, theta=theta, q_min=q_min, alpha=alpha,
                                                              init_beta=init_beta, xi_fn='full')
                                      )
        
        
        
        
        

    # print(f'{"n":5s}\t', f'{"L":5s}\t', f'{"bet":10s}\t', f'{"beta":10s}\t', f'{"p":60s}\t', f'{"Q(p)":60s}\t', f'{"cons":10s}\t', f'{"dual":10s}\t')
    
    randperm = torch.randperm(len(scores))
    rand_scores, rand_labels = torch.Tensor(scores[randperm]), torch.Tensor(labels[randperm]).int()
    
    minimaxes = [full_obv_minimax, rand_init_minimax, uni_init_minimax]
    sumlses, logwes = [[0] for _ in range(len(minimaxes))], [[1] for _ in range(len(minimaxes))]
    names = ['sample_every', 'rand_init', 'uni_init']
    
    threshold = np.log(1 / alpha)
    torch_init_beta = torch.Tensor([init_beta])
    
      
    
    for n in tqdm(range(run_length), desc="Running samples", leave=False, disable=True):
    
        p, y = rand_scores[n], rand_labels[n]
        
        x = featurize(p.reshape(1, -1), torch.ones(1000), torch_init_beta).reshape(1, -1)
        
        in_data = ((p.reshape(1, -1), x), y.reshape(1, -1))
        
        for minimax, sumls, logw in list(zip(minimaxes, sumlses, logwes)):
            minimax.addobs(in_data)
            cur_logw = minimax._primal_player._iwmart._stats[0, 0]
            if n % step == (step - 1) or (logw[-1] < threshold and cur_logw >= threshold):
                sumls.append(minimax._primal_player._suml)
                logw.append(cur_logw)
        if np.all([logw[-1] >= threshold for logw in logwes]):
            break
            

    return names, minimaxes, sumlses, logwes

    
if __name__ == '__main__':
    import os
    from load_data import load_imagenet_torch_preds
    import dill
    from tqdm import tqdm
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import multiprocess as mp
    
    run_length = 50000
    trials = 1000
    alpha = 0.1
    data_dir = 'data/imagenet_no_T'
    scores, labels = load_imagenet_torch_preds(data_dir)
    print("Imagenet data size", scores.shape, labels.shape)
    out_dir = 'results/single_beta=0.9_trials=1000'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    res_list = []
    with mp.Pool(12) as p:
        res_list = list(tqdm(p.imap(lambda seed: test_once(scores, labels, seed=seed, run_length=run_length, alpha=alpha), range(1, trials+1)), desc='Trials', total=trials))
    with open(f'{out_dir}/result_dump.pkl', 'wb') as out_f:
        dill.dump(res_list, out_f)
        
    stop_times = []
    
    threshold = np.log(1 / alpha)
    for names, _, sumlses, logwes in res_list:
        print(names)
        for name, sumls, logw in zip(names, sumlses, logwes):
            entry = {}

            entry['Method'] = name
            if np.all(logw < threshold):
                entry['Time'] = run_length
            else:
                entry['Time'] = sumls[np.argmax(logw >= threshold)] + 1
            entry['Max Label'] = sumls[-1]
            stop_times.append(entry)
    df = pd.DataFrame.from_records(stop_times)
    df.to_csv(f'{out_dir}/stopping_times.csv')
    sns.histplot(data=df, x='Time', hue='Method')
    plt.savefig(f'{out_dir}/stopping_times.png', dpi=300)