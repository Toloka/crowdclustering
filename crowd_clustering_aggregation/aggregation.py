import json
import math
from collections import defaultdict
from itertools import combinations
from typing import Tuple, List, Dict, Optional
from urllib.request import urlopen

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

from crowd_clustering_aggregation.MB_VDP import MB_VDP
from crowd_clustering_aggregation.crowd_clust import crowd_clust
from crowd_clustering_aggregation.data_structures import Label, Params, Prior, VdpPrior, AggregationAssignment


def get_assignments_list(
    assignments: pd.DataFrame,
    input_column: str,
    result_column: str = 'OUTPUT:result',
    assignment_id_column: str = 'ASSIGNMENT:assignment_id',
    worker_id_column: str = 'ASSIGNMENT:worker_id',
    golden_column: Optional[str] = None,
) -> Tuple[List[str], List[AggregationAssignment]]:
    assignments_list = list()
    images = set()
    for i, row in assignments.iterrows():
        if '[' in row[result_column]:
            output_list = json.loads(row[result_column].replace('""', '"').replace('\\', ''))
        else:
            output_list = json.loads('[' + row[result_column].replace('""', '"').replace('\\', '') + ']')

        current_images = json.loads('[' + row[input_column] + ']')
        current_outputs = [x['group'] for x in output_list]

        images.update(current_images)
        dd = defaultdict(list)
        for label, image in zip(current_outputs, current_images):
            dd[label].append(image)
        clusters = frozenset(frozenset(x) for x in dd.values())

        assignment_item = AggregationAssignment(images=frozenset(current_images),
                                                outputs=current_outputs,
                                                assignment_id=row[assignment_id_column],
                                                worker_id=row[worker_id_column],
                                                clusters=clusters
                                                )

        if golden_column is not None:
            if '[' in row[golden_column]:
                golden_list = json.loads(row[golden_column].replace('""', '"').replace('\\', ''))
            else:
                golden_list = json.loads('[' + row[golden_column].replace('""', '"').replace('\\', '') + ']')
            assignment_item.golden = [x['group'] for x in golden_list]

            dd = defaultdict(list)
            for label, image in zip(assignment_item.golden, current_images):
                dd[label].append(image)
            assignment_item.golden_clusters = frozenset(frozenset(x) for x in dd.values())

        assignments_list.append(assignment_item)

    return list(images), assignments_list


def prepare_for_aggregation(
    all_pictures_list: List[str],
    assignment_list: List[AggregationAssignment]
) -> Tuple[List[int], List[Tuple[int, int, int]], Dict[int, str]]:
    id_to_img = {i: x for i, x in enumerate(all_pictures_list)}
    img_to_id = {x: i for i, x in enumerate(all_pictures_list)}
    worker_to_id = {x: i for i, x in enumerate(list(set(x.worker_id for x in assignment_list)))}
    indices = list()
    comparisons = list()

    for assignment in assignment_list:
        for pair in combinations(assignment.images, 2):
            pair_in_same_cluster = -1
            for cluster in assignment.clusters:
                if pair[0] in cluster and pair[1] in cluster:
                    pair_in_same_cluster = 1
            indices.append((img_to_id[pair[0]], img_to_id[pair[1]], worker_to_id[assignment.worker_id]))
            comparisons.append(pair_in_same_cluster)

    return comparisons, indices, id_to_img


def generate_prior(data: np.ndarray) -> VdpPrior:
    xi_0 = 0.01
    eta_p = 1

    D, N = data.shape

    covariance = np.cov(data)

    m_0 = np.mean(data, axis=1)[:, np.newaxis]
    max_eig = max(np.linalg.eigvals(covariance))

    eta_0 = eta_p * D
    B_0 = eta_0 * max_eig * np.eye(D) * xi_0

    return VdpPrior(xi_0=xi_0, alpha=1, B_0=B_0, eta_0=eta_0, m_0=m_0)


def clustering_aggregation(assignments_raw: pd.DataFrame,
                           input_column: str,
                           prior: Prior) -> Tuple[Dict[int, List[str]], Dict[int, str]]:
    if input_column not in assignments_raw:
        raise ValueError('input_column should be present in DataFrame object')

    N = assignments_raw[input_column].unique().shape[-1]
    J = assignments_raw['ASSIGNMENT:worker_id'].unique().shape[-1]
    D = 4
    params = Params(N=N, D=D, J=J)
    all_pictures_list, ass_list = get_assignments_list(assignments_raw, input_column=input_column)

    L: List[int]
    label_id: List[Label]
    L, label_id, id_to_img = prepare_for_aggregation(all_pictures_list, ass_list)

    post, FE = crowd_clust(params, L, label_id, prior, 10)
    vdp_prior = generate_prior(post.mu_x)

    clustering_results = MB_VDP(post.mu_x, math.ceil(N / 2), math.floor(N / 2), vdp_prior)

    item_clusters = clustering_results.q_z.singlets.argmax(axis=1).tolist()

    cluster_dict = defaultdict(list)
    for i, x in enumerate(item_clusters):
        cluster_dict[x].append(id_to_img[i])

    return cluster_dict, id_to_img


def draw_cluster(images, plot_filename):
    plt.figure(figsize=(250, 250))
    plt.figure()
    for i, image_name in enumerate(images):
        img = Image.open(urlopen(image_name))
        ax = plt.subplot(len(images) // 5 + 1, 5, i + 1)
        ax.imshow(img)
        ax.axis('off')
    plt.savefig(plot_filename)
    plt.show()


if __name__ == '__main__':
    assignments_raw = pd.read_csv('results.csv')
    prior = Prior(1, 5, 10, 1)
    cluster_dict, id_to_img = clustering_aggregation(assignments_raw, 'INPUT:images', prior)

    for i, cluster in cluster_dict.items():
        print(str('-------------' + str(i) + '-------------').center(100))
        draw_cluster(cluster, f'clust_crowd_shoes_{i + 1}.png')
