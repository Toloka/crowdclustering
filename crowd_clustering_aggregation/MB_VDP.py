from math import ceil
from typing import Tuple, Optional, List

import numpy as np
from scipy.special import gammaln, psi

from crowd_clustering_aggregation.data_structures import VdpPrior, MB_VDP_posterior, MB_VDP_Data, MB_VDP_options, \
    MB_VDP_Q_Z, MB_VDP_partition, MB_VDP_results


def disp_status(free_energy: float, posterior: MB_VDP_posterior):
    N_k_str = np.array2string(posterior.true_N_k[0], formatter={'float_kind': lambda x: '%.4g' % x})
    print(f'F={free_energy:0.4f}; K= {posterior.true_N_k.shape[1]}; N_k=[{N_k_str}]')


def detln(X: np.ndarray) -> np.ndarray:
    d = np.linalg.cholesky(X)
    d = np.diag(d)
    return np.sum(np.log(d)) * 2


def gamma_multivariate_ln(X: np.ndarray, p: int) -> np.ndarray:
    K = X.shape[1]
    gammaln_val = gammaln(np.tile(X, (p, 1)) + 0.5 * (1 - np.tile(np.arange(1., p + 1)[:, np.newaxis], (1, K))))
    return p * (p - 1) * 0.25 * np.log(np.pi) + np.sum(gammaln_val, 0)


def eval_component_KL(posterior: MB_VDP_posterior, prior: VdpPrior) -> np.ndarray:
    D = posterior.m.shape[0]
    K = posterior.eta.shape[1]
    log_det_B = np.zeros((1, K))
    term_eta = np.zeros((2, K))

    for c in range(K):
        d = posterior.m[:, c][:, np.newaxis] - prior.m_0

        log_det_B[0, c] = detln(posterior.B[:, :, c])
        term_eta[0, c] = np.sum(posterior.inv_B[:, :, c] * (prior.xi_0 * d * d.conj().T))
        term_eta[1, c] = np.sum(posterior.inv_B[:, :, c] * prior.B_0) - D

    E_log_q_p_mean = 0.5 * D * (prior.xi_0 / posterior.xi - np.log(prior.xi_0 / posterior.xi) - 1) + 0.5 * (
        posterior.eta) * term_eta[0, :]

    psi_sum = np.sum(
        psi((np.tile(posterior.eta + 1, (D, 1)) - np.tile(np.arange(1., D + 1)[:, np.newaxis], (1, K))) * 0.5),
        axis=0)

    log_det_B0 = detln(prior.B_0)

    E_log_q_p_cov = 0.5 * prior.eta_0 * (log_det_B - log_det_B0) + 0.5 * posterior.N_k * psi_sum + \
                    0.5 * (posterior.eta) * term_eta[1, :] + \
                    gamma_multivariate_ln(np.array([prior.eta_0 * 0.5, ], ndmin=2), D) - \
                    gamma_multivariate_ln(posterior.eta * 0.5, D)

    return E_log_q_p_mean + E_log_q_p_cov


def helper_logsumexp(X, X_max):
    m = X.shape[0]
    n = X.shape[1]

    val = np.zeros(m)

    for j in range(n):
        for i in range(m):
            val[i] += np.exp(X[i, j] - X_max[i])
    for i in range(m):
        val[i] = np.log(val[i]) + X_max[i]
    return val


def eval_S_sk(data: MB_VDP_Data, posterior: MB_VDP_posterior, prior: VdpPrior, options: MB_VDP_options,
              relevant_components: Optional[np.ndarray] = None) -> np.ndarray:
    if relevant_components is None:
        K = posterior.m.shape[1]
        relevant_components = np.arange(0, K)
    else:
        K = relevant_components.shape[0]

    D = options.D
    n_items = data.singlets.shape[1] + data.sum_x.shape[1]

    S_sk = np.zeros((n_items, K))

    clump_ind = (data.Nc > 0)
    num_clumps = np.nonzero(data.Nc[0])[0].shape[0] if data.Nc.size > 0 else 0
    num_singlet = data.singlets.shape[1]

    if num_clumps > 0:
        sum_x = np.tile(np.reshape(data.sum_x[:, clump_ind], (D, 1, num_clumps)), (1, D, 1))
        Na = np.tile(np.reshape(data.Nc[clump_ind], (1, 1, num_clumps)), (D, D, 1))

    psi_sum = np.sum(psi((np.tile(posterior.eta + 1, (D, 1)) - np.tile(np.arange(1., D + 1)[:, np.newaxis],
                                                                       (1, len(posterior.eta)))) * 0.5), axis=0)

    for c in range(K):
        g_c = relevant_components[c]

        E_log_p_of_z_given_V_1 = psi(posterior.gamma[0, g_c]) - psi(np.sum(posterior.gamma[:, g_c], axis=0))
        E_log_p_of_z_given_V_2 = psi(posterior.gamma[1, np.arange(0, g_c)]) - \
                                 psi(np.sum(posterior.gamma[:, np.arange(0, g_c)], axis=0))

        E_log_p_of_z_given_V = E_log_p_of_z_given_V_1 + \
                               (np.sum(E_log_p_of_z_given_V_2) if E_log_p_of_z_given_V_2.size != 0 else 0)

        E_log_p_of_x_singlets = np.zeros((1, num_singlet))

        precision = 0.5 * posterior.inv_B[:, :, c] * posterior.eta[0, c]
        log_par = -0.5 * D * np.log(np.pi) - 0.5 * detln(posterior.B[:, :, c]) + 0.5 * psi_sum[c] - 0.5 * D / (
            posterior.xi[0, c])

        d = data.singlets - np.tile(posterior.m[:, c][:, np.newaxis], (1, num_singlet))
        E_log_p_of_x_singlets = -np.sum(d * (precision @ d), axis=0) + log_par

        if num_clumps > 0:
            t2 = sum_x * np.tile(posterior.m[:, c].T, (D, 1, num_clumps))
            term_dependent_on_n = (data.sum_xx[:, :, clump_ind] - t2 - np.permute(t2, [1, 0, 2])) / Na + np.tile(
                posterior.m[:, c] * posterior.m[:, c].T, (1, 1, num_clumps))
            S_sk[num_singlet + np.find(clump_ind), c] = -np.squeeze(
                np.sum(np.sum(np.tile(precision, (1, 1, num_clumps)) * term_dependent_on_n, axis=1),
                       axis=0)) + log_par + E_log_p_of_z_given_V

        S_sk[np.arange(0, num_singlet), c] = E_log_p_of_x_singlets + E_log_p_of_z_given_V
    S_sk[:, -1] -= np.log(1 - np.exp(psi(prior.alpha) - psi(1 + prior.alpha)))

    return S_sk


def eval_free_energy(data: MB_VDP_Data,
                     posterior: MB_VDP_posterior,
                     prior: VdpPrior,
                     options: MB_VDP_options) -> Tuple[float, np.ndarray]:
    component_KL = eval_component_KL(posterior, prior)
    S_sk = eval_S_sk(data, posterior, prior, options)

    posterior_gamma_sum = np.sum(posterior.gamma, axis=0)
    stick_break_KL = gammaln(posterior_gamma_sum) - \
                     gammaln(1 + prior.alpha) - \
                     np.sum(gammaln(posterior.gamma), axis=0) + \
                     gammaln(prior.alpha) + \
                     ((posterior.gamma[0, :] - 1) * (psi(posterior.gamma[0, :]) - psi(posterior_gamma_sum))) + \
                     ((posterior.gamma[1, :] - prior.alpha) * (psi(posterior.gamma[1, :]) - psi(posterior_gamma_sum)))

    num_singlets = data.singlets.shape[1]

    lse_S_sk = helper_logsumexp(S_sk, S_sk.max(1))

    free_energy = -np.sum(component_KL) - np.sum(stick_break_KL) + \
                  options.mag_factor * np.sum(lse_S_sk[0: num_singlets])

    if data.Nc.size != 0:
        extra_free_energy = options.mag_factor * data.Nc[0] * lse_S_sk[num_singlets:]
        free_energy += extra_free_energy if np.nonzero(extra_free_energy)[0].any() else 0

    return free_energy, S_sk


def free_energy_improved(free_energy: float, new_free_energy: float, options: MB_VDP_options) -> bool:
    diff = new_free_energy - free_energy
    return abs(diff / free_energy) > options.threshold and diff > 0


def helper_column_sub(X_in: np.ndarray, lse: np.ndarray) -> np.ndarray:
    m, n = X_in.shape

    for j in range(n):
        for i in range(m):
            X_in[i, j] -= lse[i]
    return X_in


def eval_q_z(data: MB_VDP_Data, posterior: MB_VDP_posterior, prior: VdpPrior, options: MB_VDP_options,
             S_sk: Optional[np.ndarray] = None) -> MB_VDP_Q_Z:
    num_singlet = data.singlets.shape[1]

    if S_sk is None:
        S_sk = eval_S_sk(data, posterior, prior, options)

    lse_S_sk = helper_logsumexp(S_sk, S_sk.max(1))
    S_sk = helper_column_sub(S_sk, lse_S_sk)

    return MB_VDP_Q_Z(singlets=np.exp(S_sk[0:num_singlet, :]), clumps=np.exp(S_sk[num_singlet:, :]))


def iterate_posterior(data: MB_VDP_Data, posterior: MB_VDP_posterior, prior: VdpPrior, options: MB_VDP_options,
                      iter: float = np.inf, do_sort: bool = True,
                      suppress_output: bool = False) -> Tuple[float, MB_VDP_posterior, MB_VDP_Q_Z]:
    if not suppress_output:
        print('Updating until convergence...')

    free_energy = -np.inf
    i = 0
    start_sort = 0

    while True:
        i = i + 1
        new_free_energy, S_sk = eval_free_energy(data, posterior, prior, options)
        if not suppress_output:
            disp_status(new_free_energy, posterior)
        if (np.isinf(iter) and not free_energy_improved(free_energy, new_free_energy, options)) or \
                (not np.isinf(iter) and i >= iter):
            free_energy = new_free_energy
            if not start_sort and do_sort:
                start_sort = 1
            else:
                break

        free_energy = new_free_energy
        q_z = eval_q_z(data, posterior, prior, options, S_sk)

        sum1 = np.sum(q_z.singlets[:, -1])
        sum2 = np.sum(q_z.clumps[data.Nc[0] > 0, -1]) if data.Nc.size > 0 else 0
        if sum1 + sum2 > 1.0e-20:
            q_z.singlets = np.append(q_z.singlets, np.zeros((q_z.singlets.shape[0], 1)), axis=1)
            q_z.clumps = np.append(q_z.clumps, np.zeros((q_z.clumps.shape[0], 1)), axis=1)

        if start_sort:
            N_k = np.sum(q_z.singlets, 0) + data.Nc[0] @ q_z.clumps
            I = np.argsort(-N_k)
            q_z.singlets = q_z.singlets[:, I]
            q_z.clumps = q_z.clumps[:, I]

        sum1 = np.sum(q_z.singlets[:, -2])
        sum2 = np.sum(q_z.clumps[data.Nc[0] > 0, -2]) if data.Nc.size > 0 else 0
        if (sum1 + sum2) < 1.0e-10:
            q_z.singlets = np.delete(q_z.singlets, -2, axis=1)
            q_z.clumps = np.delete(q_z.clumps, -2, axis=1)

        posterior = eval_posterior(data, q_z, prior, options)

    if suppress_output:
        disp_status(new_free_energy, posterior)

    return free_energy, posterior, q_z


def eval_delta_fe(subdata: MB_VDP_Data, post_posterior: MB_VDP_posterior, pre_posterior: MB_VDP_posterior,
                  pre_N_k: np.ndarray, post_N_k: np.ndarray, options: MB_VDP_options, prior: VdpPrior,
                  comps: int) -> Tuple[float, MB_VDP_posterior, MB_VDP_posterior]:
    post_component_KL = eval_component_KL(post_posterior, prior)
    pre_component_KL = eval_component_KL(pre_posterior, prior)

    pre_idx = np.argsort(-options.mag_factor * pre_N_k)
    pre_N_k = pre_N_k[pre_idx]
    post_idx = np.argsort(-options.mag_factor * post_N_k)
    post_N_k = post_N_k[post_idx]

    pre_N_k = np.append(pre_N_k, 0)
    post_N_k = np.append(post_N_k, 0)

    pre_gamma = np.zeros((2, len(pre_N_k)))
    pre_gamma[0, :] = 1 + pre_N_k
    pre_gamma[1, :] = prior.alpha + np.sum(pre_N_k) - np.cumsum(pre_N_k)
    post_gamma = np.zeros((2, len(post_N_k)))
    post_gamma[0, :] = 1 + post_N_k
    post_gamma[1, :] = prior.alpha + np.sum(post_N_k) - np.cumsum(post_N_k)

    pre_stick_KL = gammaln(np.sum(pre_gamma, 0)) - gammaln(1 + prior.alpha) - np.sum(gammaln(pre_gamma), 0) + \
                   gammaln(prior.alpha) + ((pre_gamma[0, :] - 1) * (psi(pre_gamma[0, :]) - psi(np.sum(pre_gamma, 0)))) + \
                   ((pre_gamma[1, :] - prior.alpha) * (psi(pre_gamma[1, :]) - psi(np.sum(pre_gamma, 0))))

    post_stick_KL = gammaln(np.sum(post_gamma, 0)) - gammaln(1 + prior.alpha) - np.sum(gammaln(post_gamma), 0) + \
                    gammaln(prior.alpha) + (
                        (post_gamma[0, :] - 1) * (psi(post_gamma[0, :]) - psi(np.sum(post_gamma, 0)))) + \
                    ((post_gamma[1, :] - prior.alpha) * (psi(post_gamma[1, :]) - psi(np.sum(post_gamma, 0))))

    pre_posterior.gamma = pre_gamma
    post_posterior.gamma = post_gamma

    pre_comps = np.append(np.nonzero(pre_idx == comps)[0], pre_N_k.shape[0] - 1)
    post_comps = np.append(np.nonzero(post_idx == comps)[0], np.nonzero(post_idx == (post_N_k.shape[0] - 2))[0])
    post_comps = np.append(post_comps, post_N_k.shape[0] - 1)

    pre_S_sk = eval_S_sk(subdata, pre_posterior, prior, options, pre_comps)
    post_S_sk = eval_S_sk(subdata, post_posterior, prior, options, post_comps)

    lse_pre_S_sk = helper_logsumexp(pre_S_sk, pre_S_sk.max(1))
    lse_post_S_sk = helper_logsumexp(post_S_sk, post_S_sk.max(1))

    num_singlet = subdata.singlets.shape[1]

    delta_fe = -np.sum(post_component_KL) + np.sum(pre_component_KL) - np.sum(post_stick_KL) + np.sum(pre_stick_KL) + \
               options.mag_factor * np.sum(lse_post_S_sk[0: num_singlet]) - \
               options.mag_factor * np.sum(lse_pre_S_sk[0: num_singlet])

    if subdata.Nc.size != 0:
        delta_fe = delta_fe + subdata.Nc * lse_post_S_sk[num_singlet:] - subdata.Nc * lse_pre_S_sk[num_singlet:]

    return delta_fe, post_posterior, pre_posterior


def divide_by_principal_component(data: np.ndarray, covariance: np.ndarray, mean: np.ndarray) -> np.ndarray:
    N = data.shape[1]
    D, V = np.linalg.eig(covariance)
    principal_component_i = np.argmax(D)
    principal_component = V[:, principal_component_i]
    return np.sum((data - np.tile(mean[:, np.newaxis], (1, N))) *
                  np.tile(principal_component[:, np.newaxis], (1, N)), 0)


def split(data: MB_VDP_Data, c: int, q_z: MB_VDP_Q_Z, posterior: MB_VDP_posterior) -> MB_VDP_Q_Z:
    if c >= posterior.B.shape[1] - 1:
        cluster_mean = posterior.m[:, 0]
        cluster_cov = posterior.B[:, :, 0] / posterior.eta[0, 0]
    else:
        cluster_mean = posterior.m[:, c]
        cluster_cov = posterior.B[:, :, c] / posterior.eta[0, c]

    valid_ind = data.Nc > 0

    dir_singlet = divide_by_principal_component(data.singlets, cluster_cov, cluster_mean)
    if valid_ind.size > 0:
        dir_clump = divide_by_principal_component(
            data.sum_x[:, valid_ind] / data.Nc[np.ones((data.sum_x.shape[0], 1)), valid_ind], cluster_cov,
            cluster_mean)
    else:
        dir_clump = []

    q_z_c1_s = np.zeros((q_z.singlets.shape[0], 1))
    q_z_c2_s = q_z.singlets[:, c].copy()
    q_z_c1_s[dir_singlet >= 0] = q_z.singlets[dir_singlet >= 0, c].copy()[:, np.newaxis]
    q_z_c2_s[dir_singlet >= 0] = 0

    new_q_z_singlets = np.zeros((q_z.singlets.shape[0], q_z.singlets.shape[1] + 1))
    new_q_z_singlets[:, :q_z.singlets.shape[1] - 1] = q_z.singlets[:, :q_z.singlets.shape[1] - 1]
    new_q_z_singlets[:, -1] = q_z.singlets[:, -1]
    new_q_z_singlets[:, c] = q_z_c1_s.squeeze()
    new_c = new_q_z_singlets.shape[1] - 2
    new_q_z_singlets[:, new_c] = q_z_c2_s

    new_q_z_clumps = None
    if q_z.clumps is not None:
        q_z_c1_cl = np.zeros((q_z.clumps.shape[0], 1))
        q_z_c2_cl = q_z.clumps[:, c]
        q_z_c1_cl[dir_clump >= 0] = q_z.clumps[dir_clump >= 0, c]
        q_z_c2_cl[dir_clump >= 0] = 0

        new_q_z_clumps = np.zeros((q_z.clumps.shape[0], q_z.clumps.shape[1] + 1))
        new_q_z_clumps[:, :q_z.clumps.shape[1] - 1] = q_z.clumps[:, :q_z.clumps.shape[1] - 1]
        new_q_z_clumps[:, -1] = q_z.clumps[:, -1]
        new_q_z_clumps[:, c] = q_z_c1_cl
        new_c = new_q_z_clumps.shape[1] - 1
        new_q_z_clumps[:, new_c] = q_z_c2_cl

    return MB_VDP_Q_Z(singlets=new_q_z_singlets, clumps=new_q_z_clumps)


def choose_split_merge(data: MB_VDP_Data, posterior: MB_VDP_posterior, prior: VdpPrior, options: MB_VDP_options,
                       old_q_z: MB_VDP_Q_Z) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], MB_VDP_partition]:
    K = posterior.m.shape[1] - 1
    c_max = K
    split_candidates = np.arange(0, K)
    sub_partition = [] * K
    member_singlets = [[]] * K
    member_clumps = [[]] * K

    N_k = np.zeros((1, K))

    top_clusters = old_q_z.singlets.argmax(1)
    if np.nonzero(data.Nc[0])[0].any():
        top_clusters_cl = old_q_z.clumps.argmax(1)

    for c in range(K):
        member_singlets[c] = np.nonzero(top_clusters == c)[0]
        N_k[0, c] = len(member_singlets[c])
        if np.nonzero(data.Nc[0])[0].any():
            member_clumps[c] = np.nonzero(top_clusters_cl == c)[0]
            N_k[0, c] += np.sum(data.Nc[member_clumps[c]])

    split_delta_free_energy = -np.inf * np.ones((1, c_max))
    for i in range(min(c_max, len(split_candidates))):
        c = split_candidates[i]

        if len(member_singlets[c]) == 0 and len(member_clumps[c]) == 0:
            continue

        subdata_sum_xx = data.sum_xx[:, :, member_clumps[c]]
        subdata = MB_VDP_Data(singlets=data.singlets[:, member_singlets[c]], sum_x=data.sum_x[:, member_clumps[c]],
                              sum_xx=subdata_sum_xx, Nc=data.Nc[member_clumps[c]])
        pre_q_z_singlets = np.zeros((member_singlets[c].shape[0], 2))
        pre_q_z_singlets[:, 0] = 1
        if len(member_clumps[c]) > 0:
            pre_q_z_clumps = np.zeros((member_clumps[c].shape[0], 2))
            pre_q_z_clumps[:, 0] = 1
        else:
            pre_q_z_clumps = None
        pre_q_z = MB_VDP_Q_Z(singlets=pre_q_z_singlets, clumps=pre_q_z_clumps)
        pre_posterior = eval_posterior(subdata, pre_q_z, prior, options)

        sub_q_z = split(subdata, 0, pre_q_z, pre_posterior)

        sub_q_z_singlets_sum = np.sum(sub_q_z.singlets[:, 0:-1], 0)
        sub_q_z_clumps_sum = np.sum(sub_q_z.clumps[subdata.Nc > 0, 0:-1], 0) if sub_q_z.clumps is not None else 0

        if np.nonzero((sub_q_z_singlets_sum + sub_q_z_clumps_sum) < 1.0e-10)[0].size != 0:
            continue

        sub_posterior = eval_posterior(subdata, sub_q_z, prior, options)
        sub_fe, sub_posterior, sub_q_z = iterate_posterior(subdata, sub_posterior, prior, options, 10, False, True)

        new_N_k = N_k.copy()

        top_sub_par = sub_q_z.singlets.argmax(1)
        new_sub_partition = MB_VDP_partition(member_s1=member_singlets[c][top_sub_par == 0],
                                             member_s2=member_singlets[c][top_sub_par == 1])
        new_N_k = np.resize(new_N_k, (1, K + 1))
        new_N_k[0, c] = new_sub_partition.member_s1.shape[0]
        new_N_k[0, K] = new_sub_partition.member_s2.shape[0]

        if subdata.Nc.size > 0:
            top_sub_par_cl = sub_q_z.clumps.argmax(1)
            new_sub_partition.member_c1 = member_clumps[c][top_sub_par_cl == 0]
            new_N_k[0, c] += np.sum(data.Nc[sub_partition[c].member_c1])
            new_sub_partition.member_c2 = member_clumps[c][top_sub_par_cl == 1]
            new_N_k[0, K] += np.sum(data.Nc[sub_partition[c].member_c2])

        sub_partition.append(new_sub_partition)

        if (new_sub_partition.member_s1.size == 0 and
            (new_sub_partition.member_c1 is None or new_sub_partition.member_c1.size == 0)) \
            or (new_sub_partition.member_s2.size == 0 and
                (new_sub_partition.member_c2 is None or new_sub_partition.member_c2.size == 0)):
            continue

        split_delta_free_energy[0, i], sub_posterior, pre_posterior = eval_delta_fe(subdata, sub_posterior,
                                                                                    pre_posterior,
                                                                                    N_k[0], new_N_k[0],
                                                                                    options, prior, c)

    return split_delta_free_energy, member_singlets, member_clumps, sub_partition


def split_merge(data: MB_VDP_Data,
                posterior: MB_VDP_posterior,
                prior: VdpPrior,
                options: MB_VDP_options) -> Tuple[float, MB_VDP_posterior]:
    c_max = 5
    free_energy, posterior, q_z = iterate_posterior(data, posterior, prior, options)
    fe_improved = True
    K = q_z.singlets.shape[1] - 1

    while fe_improved:
        split_delta_free_energy, member_singlets, member_clumps, sub_partition = choose_split_merge(data, posterior,
                                                                                                    prior, options, q_z)
        split_ind = np.argsort(-split_delta_free_energy[0])

        for i in range(min(c_max, len(split_ind))):
            split_c = split_ind[i]

            if sub_partition[split_c] is None:
                fe_improved = False
                continue

            new_q_z_singlets = np.zeros((q_z.singlets.shape[0], q_z.singlets.shape[1] + 1))
            new_q_z_singlets[:, :-1] = q_z.singlets.copy()
            new_q_z_singlets[member_singlets[split_c], :] = 0
            if sub_partition[split_c].member_s1 is not None:
                new_q_z_singlets[sub_partition[split_c].member_s1, split_c] = 1
            if sub_partition[split_c].member_s2 is not None:
                new_q_z_singlets[sub_partition[split_c].member_s2, K] = 1

            new_q_z_clumps = np.zeros((q_z.clumps.shape[0], q_z.clumps.shape[1] + 1))
            new_q_z_clumps[:, :-1] = q_z.clumps
            new_q_z_clumps[member_clumps[split_c], :] = 0
            if sub_partition[split_c].member_c1 is not None:
                new_q_z_clumps[sub_partition[split_c].member_c1, split_c] = 1
            if sub_partition[split_c].member_c2 is not None:
                new_q_z_clumps[sub_partition[split_c].member_c2, -1] = 1

            new_q_z = MB_VDP_Q_Z(singlets=new_q_z_singlets, clumps=new_q_z_clumps)

            new_posterior = eval_posterior(data, new_q_z, prior, options)

            new_free_energy, new_posterior, new_q_z = iterate_posterior(data, new_posterior, prior, options)

            if free_energy_improved(free_energy, new_free_energy, options):
                free_energy = new_free_energy
                posterior = new_posterior
                fe_improved = True
                K = new_q_z.singlets.shape[1] - 1
                q_z = new_q_z
                break
            elif i == min(c_max, len(split_ind)) - 1:
                fe_improved = False

    return free_energy, posterior


def helper_B_full(X: np.ndarray, q_of_z: np.ndarray) -> np.ndarray:
    m = X.shape[0]
    n = X.shape[1]
    B = np.zeros((m, m))
    for k in range(n):
        for j in range(m):
            for i in range(m):
                B[i, j] += q_of_z[k] * X[i, k] * X[j, k]
    return B


def eval_posterior(data: MB_VDP_Data, q_z: MB_VDP_Q_Z, prior: VdpPrior, options: MB_VDP_options) -> MB_VDP_posterior:
    threshold_for_N = 1.0e-200
    K = q_z.singlets.shape[1]
    D = options.D

    true_N_k = np.sum(q_z.singlets, axis=0)[np.newaxis, :]

    q_z.singlets[:, -1] = 0
    if q_z.clumps is not None:
        q_z.clumps[:, -1] = 0

    N_k = options.mag_factor * true_N_k

    if q_z.clumps is not None:
        sum_x = options.mag_factor * (data.singlets @ q_z.singlets + data.sum_x @ q_z.clumps)
    else:
        sum_x = options.mag_factor * (data.singlets @ q_z.singlets)

    inv_N_k = np.zeros((1, K))
    inv_N_k[N_k > threshold_for_N] = 1. / N_k[N_k > threshold_for_N]

    means = sum_x * np.tile(inv_N_k, (D, 1))

    posterior_inv_B = np.zeros((D, D, K))
    posterior_B = np.zeros((D, D, K))

    posterior = MB_VDP_posterior(B=posterior_B, inv_B=posterior_inv_B, eta=prior.eta_0 + N_k, xi=prior.xi_0 + N_k)

    for c in range(K):
        v0 = means[:, c][:, np.newaxis] - prior.m_0

        if q_z.clumps is not None:
            cq_of_z_c = np.reshape(q_z.clumps[:, c], (1, 1, q_z.clumps.shape[0]))
            S = options.mag_factor * np.sum(np.tile(cq_of_z_c, (D, D, 1)) * data.sum_xx, axis=2) + \
                options.mag_factor * helper_B_full(data.singlets, q_z.singlets[:, c]) - \
                N_k[0, c] * means[:, c][np.newaxis, :] * means[:, c][np.newaxis, :].conj().T
        else:
            S = options.mag_factor * helper_B_full(data.singlets, q_z.singlets[:, c]) - \
                N_k[0, c] * means[:, c][np.newaxis, :] * means[:, c][np.newaxis, :].conj().T
        posterior.B[:, :, c] = prior.B_0 + S + N_k[0, c] * prior.xi_0 * v0 * v0.conj().T / (posterior.xi[:, c])

    for c in range(K):
        posterior.inv_B[:, :, c] = np.linalg.inv(posterior.B[:, :, c])

    posterior.m = (sum_x + np.tile((prior.xi_0 * prior.m_0), (1, K))) / np.tile(N_k + prior.xi_0, (D, 1))
    posterior.gamma = np.zeros((2, K))
    posterior.gamma[0, :] = 1 + N_k
    posterior.gamma[1, :] = prior.alpha + np.sum(N_k) - np.cumsum(N_k)
    posterior.N_k = N_k
    posterior.true_N_k = true_N_k

    return posterior


def memory_cost(N_k, D: int, THRESHOLD: float):
    CLUMP_COST = (D + 3) / 2.0

    num_clumps = len(np.nonzero(N_k >= THRESHOLD))
    num_singlets = np.sum(N_k(N_k < THRESHOLD))

    return np.ceil(num_clumps * CLUMP_COST + num_singlets)


def compression_phase(data: MB_VDP_Data, prior: VdpPrior, posterior: MB_VDP_posterior, model_q_z: MB_VDP_Q_Z,
                      options: MB_VDP_options, T: int) -> Tuple[MB_VDP_Data, MB_VDP_options]:
    K = posterior.m.shape[1] - 1

    options.mag_factor = options.N / T

    sub_partition = np.zeros((1, K))
    member_singlets = np.zeros((1, K))
    member_clumps = np.zeros((1, K))
    evaluated = np.zeros((1, K))
    delta_free_energy = -np.inf * np.ones((1, K))
    N_k = np.zeros((1, K))

    top_clusters = model_q_z.singlets.max(1)

    if np.nonzero(data.Nc):
        top_clusters_cl = model_q_z.clumps.max(1)

    for c in range(K):
        member_singlets[c] = np.nonzero(top_clusters == c)
        N_k[c] = len(member_singlets[c])
        if np.nonzero(data.Nc):
            member_clumps[c] = np.nonzero(top_clusters_cl == c)
            N_k[c] = N_k[c] + np.sum(data.Nc[member_clumps[c]])

    THRESHOLD = (options.D + 3) / 2.0

    while memory_cost(N_k, options.D, THRESHOLD) < options.M:
        for c in range(K):
            if N_k[c] < THRESHOLD:
                evaluated[c] = True
            elif member_singlets[c].size == 0 and member_clumps[c] == 0:
                evaluated[c] = True
            else:
                subdata_sum_xx = data.sum_xx[:, :, member_clumps[c]]

                subdata = MB_VDP_Data(singlets=data.singlets[:, member_singlets[c]],
                                      sum_x=data.sum_x[:, member_clumps[c]],
                                      sum_xx=subdata_sum_xx,
                                      Nc=data.Nc[member_clumps[c]])
                pre_q_z_singlets = np.zeros((member_singlets[c].shape[0], 2))
                pre_q_z_singlets[:, 0] = 1
                pre_q_z_clumps = np.zeros((member_clumps[c].shape[0], 2))
                pre_q_z_clumps[:, 0] = 1
                pre_q_z = MB_VDP_Q_Z(singlets=pre_q_z_singlets, clumps=pre_q_z_clumps)
                pre_posterior = eval_posterior(subdata, pre_q_z, prior, options)

                sub_q_z = split(subdata, 0, pre_q_z, pre_posterior, options)
                if not np.nonzero((np.sum(sub_q_z.singlets[:, :-1], 0) +
                                   np.sum(sub_q_z.clumps[subdata.Nc > 0, :-1], 0)) < 1.0e-10, axis=1).any():
                    evaluated[c] = True
                else:
                    sub_posterior = eval_posterior(subdata, sub_q_z, prior, options)
                    sub_fe, sub_posterior, sub_q_z = iterate_posterior(subdata, sub_posterior, prior, options, 10,
                                                                       False, True)

                    new_N_k = N_k

                    top_sub_par = sub_q_z.singlets.max(1)
                    sub_partition[c].member_s1 = member_singlets[c, top_sub_par == 1]
                    new_N_k[c] = len(sub_partition[c].member_s1)
                    sub_partition[c].member_s2 = member_singlets[c, top_sub_par == 2]
                    new_N_k[K + 1] = len(sub_partition[c].member_s2)
                    if np.nonzero(subdata.Nc):
                        top_sub_par_cl = sub_q_z.clumps.max(1)
                        sub_partition[c].member_c1 = member_clumps[c, top_sub_par_cl == 1]
                        new_N_k[c] += np.sum(data.Nc[sub_partition[c].member_c1])
                        sub_partition[c].member_c2 = member_clumps[c, top_sub_par_cl == 2]
                        new_N_k[K + 1] += np.sum(data.Nc[sub_partition[c].member_c2])
                    else:
                        sub_partition[c].member_c1 = []
                        sub_partition[c].member_c2 = []

                    if ((sub_partition[c].member_s1.size == 0 and sub_partition[c].member_c1.size == 0) or
                            (sub_partition[c].member_s2.size == 0 and sub_partition[c].member_c2.size == 0)):
                        evaluated[c] = True
                    else:
                        delta_free_energy[c] = eval_delta_fe(subdata, sub_posterior, pre_posterior, N_k[0], new_N_k[0],
                                                             options, prior, c)
                        evaluated[c] = True

        free_energy, c = np.max(delta_free_energy)
        if np.isinf(delta_free_energy):
            break

        member_singlets[c] = sub_partition[c].member_s1
        member_clumps[c] = sub_partition[c].member_c1
        member_singlets[K + 1] = sub_partition[c].member_s2
        member_clumps[K + 1] = sub_partition[c].member_c2

        N_k[c] = len(member_singlets[c]) + np.sum(data.Nc[member_clumps[c]])
        N_k[K] = len(member_singlets[K]) + np.sum(data.Nc[member_clumps[K]])

        evaluated[c] = False
        evaluated[K] = False
        delta_free_energy[c] = -np.inf
        delta_free_energy[K] = -np.inf
        K += 1

    the_clumps = N_k >= THRESHOLD
    the_singlets = (N_k > 0) & (N_k < THRESHOLD)

    temp_x = data.sum_x
    temp_xx = data.sum_xx
    temp_Nc = data.Nc

    for c in the_clumps.tolist():
        data.sum_x[:, c] = (np.sum(data.singlets[:, member_singlets[c]], axis=1) +
                            np.sum(temp_x[:, member_clumps[c]], axis=1))

        data.Nc[c] = len(member_singlets[c]) + np.sum(temp_Nc[member_clumps[c]])

        if member_singlets[c].size != 0:
            data.sum_xx[:, :, c] = data.singlets[:, member_singlets[c]] * data.singlets[:, member_singlets[c]].conj().T
        else:
            data.sum_xx[:, :, c] = np.zeros((options.D, options.D, 1))
        if member_clumps[c].size != 0:
            data.sum_xx[:, :, c] += data.sum_xx[:, :, c] + np.sum(temp_xx[:, :, member_clumps[c]], axis=2)

    data.sum_x = data.sum_x[:, the_clumps]
    data.sum_xx = data.sum_xx[:, :, the_clumps]

    data.Nc = data.Nc[the_clumps]

    data.singlets = data.singlets[:, np.vstack(member_singlets[the_singlets])]

    return data, options


def MB_VDP(X: np.ndarray, M: int, E: int, prior: VdpPrior) -> MB_VDP_results:
    D = X.shape[0]
    options = MB_VDP_options(M=M, E=E, N=X.shape[1], D=D)

    max_clumps = ceil((2 * M) / (D + 3))
    data = MB_VDP_Data(singlets=X[:, :(M + E)], sum_x=np.zeros((D, max_clumps)), Nc=np.zeros((1, max_clumps)),
                       sum_xx=np.zeros((D, D, max_clumps)))

    n_pts_in_epoch = M + E
    i = 0
    T = 0

    while T < options.N:
        i += 1

        if options.restart or i == 1:
            q_z_singlets = np.zeros((data.singlets.shape[1], 2))
            q_z_singlets[:, 0] = 1
            q_z_clumps = np.zeros((data.sum_x.shape[1], 2))
            q_z_clumps[:, 0] = 1
            q_z = MB_VDP_Q_Z(singlets=q_z_singlets, clumps=q_z_clumps)
            posterior = eval_posterior(data, q_z, prior, options)

        free_energy, posterior = split_merge(data, posterior, prior, options)

        T = T + n_pts_in_epoch

        if T < options.N:
            q_z = eval_q_z(data, posterior, prior, options)
            data, options = compression_phase(data, prior, posterior, q_z, options, T)

            if (options.N - T) >= E:
                n_pts_in_epoch = E
            else:
                n_pts_in_epoch = options.N - T

            data.singlets = np.concatenate(data.singlets, X[:, T: T + n_pts_in_epoch + 1])

    results_q_z = eval_q_z(data, posterior, prior, options)

    return MB_VDP_results(data=data, free_energy=free_energy, prior=prior,
                          posterior=posterior, K=posterior.eta.shape[-1],
                          options=options, q_z=results_q_z)
