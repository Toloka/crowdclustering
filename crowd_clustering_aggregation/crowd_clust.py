from typing import List, Tuple

import numpy as np

from crowd_clustering_aggregation.data_structures import Label, Params, Prior, Post, RelevantLabels


def eye_3d(dim_1_2: int, dim_3: int) -> np.ndarray:
    a = np.zeros((dim_1_2, dim_1_2, dim_3))
    for j in range(dim_3):
        a[:, :, j] = np.eye(dim_1_2)
    return a


def create_post(params: Params, L: List[int], prior: Prior) -> Post:
    dir = prior.alpha * np.ones((params.D, 1))
    mu_x = np.random.randn(params.D, params.N)
    sig_x = prior.sig_x * np.ones((params.D, params.N))
    sig_W = prior.sig_w * np.ones((params.D, params.D, params.J))

    mu_W = eye_3d(params.D, params.J)

    mu_tau = np.zeros(params.J)
    sig_tau = prior.sig_tau * np.ones(params.J)
    delta = 0.01 * np.ones(len(L))
    lambda_delta = (0.5 - 1. / (1 + np.exp(-delta))) / (2 * delta)

    return Post(dir=dir, mu_x=mu_x, sig_x=sig_x, sig_W=sig_W, mu_W=mu_W, mu_tau=mu_tau, sig_tau=sig_tau, delta=delta,
                lambda_delta=lambda_delta)


def stimes(A: np.ndarray, B: np.ndarray, reshape_A: List[int], reshape_B: List[int]) -> np.ndarray:
    if len(reshape_A) > 0:
        A = A.reshape(reshape_A, order='F').copy()
    if len(reshape_B) > 0:
        B = B.reshape(reshape_B, order='F').copy()

    C = np.multiply(A, B)
    return C


def inner(A: np.ndarray, B: np.ndarray, reshape_A: List[int], reshape_B: List[int], sum_dim: int) -> np.ndarray:
    C = stimes(A, B, reshape_A, reshape_B)
    C = C.sum(sum_dim)
    return C.squeeze()


def diag_tensor(A: np.ndarray) -> np.ndarray:
    return inner(eye_3d(A.shape[0], A.shape[2]), A, [], [], 0)


def tensor_trace(A: np.ndarray) -> np.ndarray:
    inner1 = inner(eye_3d(A.shape[0], A.shape[2]), A, [], [], 1)
    return np.sum(inner1, 0)


def update_delta(post: Post, label_id: List[Label]) -> Post:
    D = post.mu_x.shape[0]
    T = len(label_id)

    aa, bb, jj = [x[0] for x in label_id], [x[1] for x in label_id], [x[2] for x in label_id]

    E_xaxa = stimes(post.mu_x[:, aa], post.mu_x[:, aa], [D, 1, T], [1, D, T]) + \
             stimes(eye_3d(D, T), post.sig_x[:, aa], [], [1, D, T])
    E_xbxb = stimes(post.mu_x[:, bb], post.mu_x[:, bb], [D, 1, T], [1, D, T]) + \
             stimes(eye_3d(D, T), post.sig_x[:, bb], [], [1, D, T])
    C = E_xaxa * E_xbxb * post.sig_W[:, :, jj]

    E_xaWxb_sqr = tensor_trace(
        inner(
            inner(E_xbxb, post.mu_W[:, :, jj], [D, D, 1, T], [1, D, D, T], 1),
            inner(E_xaxa, post.mu_W[:, :, jj], [D, D, 1, T], [1, D, D, T], 1),
            [D, D, 1, T], [1, D, D, T], 1)
    ) + \
                  np.sum(
                      inner(
                          stimes(diag_tensor(E_xaxa), diag_tensor(E_xbxb), [D, 1, T], [1, D, T]),
                          post.sig_W[:, :, jj], [], [], 1
                      ), 0) + \
                  np.sum(np.sum(C, 1), 0) - tensor_trace(C)

    delta_sqr = E_xaWxb_sqr + \
                2 * post.mu_tau[jj] * np.sum(
        np.squeeze(
            np.sum(
                stimes(
                    stimes(post.mu_x[:, aa], post.mu_x[:, bb], [D, 1, T], [1, D, T]),
                    post.mu_W[:, :, jj], [], [])
                , 1)
        ), 0) + \
                post.mu_tau[jj] ** 2 + \
                post.sig_tau[jj]
    post.delta = delta_sqr ** 0.5
    post.lambda_delta = (0.5 - 1. / (1 + np.exp(-post.delta))) / (2 * post.delta)
    return post


def compute_relevant(params: Params, label_id: List[Label]) -> RelevantLabels:
    relevant_labels: RelevantLabels = RelevantLabels([[], ] * params.N, [[], ] * params.J, [[], ] * params.N)

    for i in range(params.N):
        label_idx1 = [j for j, x in enumerate(label_id) if x[0] == i]
        label_idx2 = [j for j, x in enumerate(label_id) if x[1] == i]
        relevant_labels.img[i] = [*label_idx1, *label_idx2]
        labels_bb1 = [label_id[j][1] for j in label_idx1]
        labels_bb2 = [label_id[j][0] for j in label_idx2]
        relevant_labels.bb[i] = [*labels_bb1, *labels_bb2]

    for j in range(params.J):
        relevant_labels.ann[j] = [k for k, x in enumerate(label_id) if x[2] == j]
    return relevant_labels


def update_img_vectors(post: Post, L: List[int], label_id: List[Label], prior: Prior,
                       relevant_labels: RelevantLabels) -> Post:
    D = post.mu_x.shape[0]
    L_np = np.array(L)

    for i in range(post.mu_x.shape[1]):
        label_idx = relevant_labels.img[i]

        T = len(label_idx)

        bb = relevant_labels.bb[i]
        j = [label_id[i][2] for i in label_idx]

        c = L_np[label_idx] / 2 + 2 * post.lambda_delta[label_idx] * post.mu_tau[j]
        cx = np.multiply(post.mu_x[:, bb], c)

        stat_first = np.sum(np.sum(stimes(post.mu_W[:, :, j], cx, [], [1, D, T]), 2), 1)

        E_xbxb = stimes(post.mu_x[:, bb], post.mu_x[:, bb], [D, 1, T], [1, D, T]) + \
                 stimes(eye_3d(D, T), post.sig_x[:, bb], [], [1, D, T])
        B = np.multiply(E_xbxb, post.sig_W[:, :, j])
        B = B - np.multiply(eye_3d(D, B.shape[2]), B)
        E_WxxW = inner(
            inner(post.mu_W[:, :, j], E_xbxb, [D, D, 1, T], [1, D, D, T], 1),
            post.mu_W[:, :, j], [D, D, 1, T], [1, D, D, T], 1) + B + \
                 stimes(eye_3d(D, T),
                        inner(post.sig_W[:, :, j], diag_tensor(E_xbxb), [], [1, D, T], 1),
                        [], [1, D, T])

        stat_second = 2 * inner(E_WxxW, post.lambda_delta[label_idx], [], [1, 1, T], 2)

        post.sig_x[:, i] = 1 / (1 / prior.sig_x - np.diag(stat_second))
        V = np.multiply(post.sig_x[:, i], stat_first)
        U = np.multiply(stat_second, post.sig_x[:, i].reshape((D, 1)))
        np.fill_diagonal(U, 0)

        post.mu_x[:, i] = np.linalg.lstsq(np.eye(D) - U, V)[0]
    return post


def init_tensors(D: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp1 = np.zeros((D, D, D, D))
    tp2 = np.zeros((D, D, D, D))
    tp3 = np.zeros((D, D, D, D))

    for i in range(D):
        tp1[:, :, i, i] = np.ones((D, D))
        tp3[:, :, i, i] = np.eye(D)
        for j in range(D):
            tp2[:, :, i, j] = np.eye(D)

    return tp1, tp2, tp3


def update_ann_weights(post: Post, L: List[int], label_id: List[Label], prior: Prior,
                       relevant_labels: RelevantLabels) -> Post:
    D = post.mu_x.shape[0]

    tp1, tp2, tp3 = init_tensors(D)
    first_tensor = np.ones((D, D, D, D)) - tp1 - tp2 + tp3
    second_tensor = np.ones((D, D, D, D)) - tp2
    third_tensor = np.ones((D, D, D, D)) - tp1

    for jj in range(post.mu_tau.size):
        label_idx = relevant_labels.ann[jj]

        T = len(label_idx)

        aa = [label_id[i][0] for i in label_idx]
        bb = [label_id[i][1] for i in label_idx]
        L_idx = np.array([L[i] for i in label_idx])
        c = L_idx / 2 + 2 * post.lambda_delta[label_idx] * post.mu_tau[jj]

        ExaExb = np.dot(post.mu_x[:, aa] * c, post.mu_x[:, bb].conj().T)
        stat_first = ExaExb + ExaExb.conj().T - np.diag(np.diag(ExaExb))

        lmA = np.multiply(post.mu_x[:, aa], post.lambda_delta[label_idx])
        lvA = np.multiply(post.sig_x[:, aa], post.lambda_delta[label_idx])

        E_xaxa = stimes(lmA, post.mu_x[:, aa], [D, 1, T], [1, D, T]) + \
                 stimes(eye_3d(D, T), lvA, [], [1, D, T])

        E_xbxb = stimes(post.mu_x[:, bb], post.mu_x[:, bb], [D, 1, T], [1, D, T]) + \
                 stimes(eye_3d(D, T), post.sig_x[:, bb], [], [1, D, T])

        E_xaxa_xbxb = inner(E_xaxa, E_xbxb, [D, D, 1, 1, T], [1, 1, D, D, T], 4)

        stat_second = 2 * (np.transpose(E_xaxa_xbxb, [0, 2, 1, 3]) * first_tensor +
                           np.transpose(E_xaxa_xbxb, [0, 2, 3, 1]) * second_tensor +
                           np.transpose(E_xaxa_xbxb, [2, 0, 1, 3]) * third_tensor +
                           np.transpose(E_xaxa_xbxb, [2, 0, 3, 1]))

        U = np.reshape(stat_second, [D ** 2, D ** 2])

        sW = 1. / (1. / prior.sig_w - np.reshape(np.diag(U), [D, D]))
        post.sig_W[:, :, jj] = sW

        V = np.multiply(sW.flatten(), stat_first.flatten())

        U = np.multiply(U, sW.flatten().reshape((D ** 2, 1)))
        np.fill_diagonal(U, 0)

        mW = np.linalg.lstsq(np.eye(D ** 2) - U, V)[0]

        post.mu_W[:, :, jj] = np.reshape(mW, [D, D])
    return post


def update_ann_bias(post: Post, L: List[int], label_id: List[Label], prior: Prior,
                    relevant_labels: RelevantLabels) -> Post:
    for jj in range(len(post.mu_tau)):
        label_idx = relevant_labels.ann[jj]
        aa = [label_id[i][0] for i in label_idx]
        bb = [label_id[i][1] for i in label_idx]
        L_idx = np.array([L[i] for i in label_idx])

        stat_second = -2 * np.sum(post.lambda_delta[label_idx])

        stimes1 = np.multiply(post.mu_x[:, aa], post.lambda_delta[label_idx])
        stimes1_mul = np.dot(stimes1, post.mu_x[:, bb].conj().T)
        stat_first = sum(L_idx) / 2 + 2 * sum(sum(np.multiply(post.mu_W[:, :, jj], stimes1_mul)))

        post.sig_tau[jj] = 1 / (1 / prior.sig_tau + stat_second)
        post.mu_tau[jj] = post.sig_tau[jj] * stat_first

    return post


def update_post(post: Post, L: List[int], label_id: List[Label], prior: Prior, relevant_labels: RelevantLabels) -> Post:
    post = update_img_vectors(post, L, label_id, prior, relevant_labels)

    post = update_ann_weights(post, L, label_id, prior, relevant_labels)

    post = update_ann_bias(post, L, label_id, prior, relevant_labels)

    post = update_delta(post, label_id)

    return post


def compute_free_energy(post: Post, L: List[int], label_id: List[Label], prior: Prior) -> Post:
    D = post.mu_x.shape[0]
    N = post.mu_x.shape[1]
    T = len(L)
    J = post.mu_W.shape[2]

    x_term = -np.sum(np.sum(post.mu_x ** 2 + post.sig_x, axis=0) + 1) / (2 * prior.sig_x) - \
             (N * D / 2) * np.log(2 * np.pi * prior.sig_x) + \
             (N * D / 2) * np.log(2 * np.pi * np.e) + \
             0.5 * np.sum(np.sum(np.log(post.sig_x)))
    w_term = 0

    for j in range(J):
        pW = np.triu(post.sig_W[:, :, j])

        w_term = w_term - \
                 np.sum(np.sum(np.triu(post.mu_W[:, :, j] ** 2 + post.sig_W[:, :, j]), axis=0)) / (2 * prior.sig_w) + \
                 0.5 * np.sum(np.log(pW[pW > 0]))

    w_term = w_term - J * ((D ** 2 + D) / 4) * (np.log(2 * np.pi * prior.sig_w) - np.log(2 * np.pi * np.e))

    tau_term = -np.sum(post.mu_tau ** 2 + post.sig_tau) / (2 * prior.sig_tau) - (J / 2) * np.log(
        2 * np.pi * prior.sig_tau) + (J / 2) * np.log(2 * np.pi * np.e) + 0.5 * np.sum(np.log(post.sig_tau))

    label_term = np.sum(np.log(1. / (1 + np.exp(-post.delta))) - post.delta / 2)

    aa, bb, jj = [x[0] for x in label_id], [x[1] for x in label_id], [x[2] for x in label_id]
    E_xWxt = np.sum(
        inner(post.mu_W[:, :, jj], stimes(post.mu_x[:, aa], post.mu_x[:, bb], [D, 1, T], [1, D, T]), [], [], 1), 0) + \
             post.mu_tau[jj]

    label_term = label_term + np.sum(np.multiply(E_xWxt, L)) / 2
    FE = x_term + w_term + tau_term + label_term
    return FE


def crowd_clust(params: Params, L: List[int], label_id: List[Label], prior: Prior, n_iter: int):
    FE = np.zeros(n_iter)
    post: Post = create_post(params, L, prior)

    post = update_delta(post, label_id)

    relevant_labels = compute_relevant(params, label_id)
    for i in range(n_iter):
        post = update_post(post, L, label_id, prior, relevant_labels)
        FE[i] = compute_free_energy(post, L, label_id, prior)
        print(f'Iteration: {i} Free Energy: {FE[i]}')

    return post, FE
