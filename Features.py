def digitize(p, knots):
    import torch
    rv = torch.zeros(len(knots))
    if p >= knots[-1]:
        rv[-1] = 1.
    else:
        z = torch.bucketize(p, knots)
        vlow, vhigh = knots[z - 1], knots[z]
        frac = (p - vlow) / (vhigh - vlow)
        rv[z - 1], rv[z] = 1 - frac, frac
    return rv


def sim_featurize(p, beta, beta_knots, est_loss_knots):
    import torch

    beta_vec = torch.cat([digitize(beta, beta_knots), torch.ones(1)])
    # scale_factor = 1 if torch.all(beta_knots > beta) else (
    #     1 - beta_knots[torch.argmin((beta_knots <= beta).int()) - 1])
    rvs = []
    for i in range(len(p)):
        #    exc_sum_vec = digitize(p[i], est_loss_knots * scale_factor)
        exc_sum_vec = digitize(p[i], est_loss_knots)
        rvs.append(torch.outer(beta_vec, exc_sum_vec).reshape(1, -1))
    res = torch.concatenate(rvs)
    return res


def imagenet_featurize(P, w, beta, beta_knots, est_loss_knots):
    import torch
    from Risk import TopBetaCoverage

    tbc = TopBetaCoverage(torch_w=w)
    w_P_sq = P * (w**2).reshape(1, -1)

    exc_sums = (w_P_sq *
                (~(tbc.beta_coverage_set(P, torch.Tensor([beta])))).int()).sum(
                    dim=-1)
    beta_vec = torch.cat([digitize(beta, beta_knots), torch.ones(1)])
    scale_factor = 1 if torch.all(beta_knots > beta) else (
        1 - beta_knots[torch.argmin((beta_knots <= beta).int()) - 1])

    rvs = []
    for i in range(len(exc_sums)):
        exc_sum_vec = digitize(exc_sums[i], est_loss_knots * scale_factor)
        rvs.append(torch.outer(beta_vec, exc_sum_vec).reshape(1, -1))
    res = torch.concatenate(rvs)
    return res


def negrec_featurize(P, beta, beta_knots, est_loss_knots):
    import torch
    from Risk import NegRecall
    # beta_knots = torch.linspace(0, 1, 5)
    # est_loss_knots = torch.linspace(0, 1, 10)
    negrec = NegRecall()
    exp_miscovers = (
        (~(negrec.beta_coverage_set(P, torch.tensor([beta]))) * P).float() *
        P).sum(dim=-1)
    exp_ys = P.sum(dim=-1)
    exp_neg_rec = (exp_miscovers / exp_ys).reshape(-1)

    beta_vec = torch.cat([digitize(beta, beta_knots), torch.ones(1)])
    rvs = []
    for i in range(len(exp_neg_rec)):
        exp_vec = digitize(exp_neg_rec[i], est_loss_knots)
        rvs.append(torch.outer(beta_vec, exp_vec).reshape(1, -1))
    res = torch.concatenate(rvs)
    return res
