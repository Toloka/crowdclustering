from crowdkit.aggregation import DawidSkene
import json
from collections import Counter
from matplotlib import pyplot as plt

def calculate_quality(assignments_raw, cluster_dict, intruder_hits):
    assignments_df = assignments_raw.rename(
        columns={'INPUT:images': 'task', 'ASSIGNMENT:worker_id': 'worker', 'OUTPUT:answer': 'label'}
    )

    img_to_clust = dict()
    for cluster, images_list in cluster_dict.items():
        for img in images_list:
            img_to_clust[img] = cluster

    dawid_skene = DawidSkene(10)
    aggregated_answers = dict(dawid_skene.fit_predict(assignments_df))
    answers = dict()
    for key, value in aggregated_answers.items():
        img_list = tuple(x.strip() for x in json.loads('[' + key + ']'))
        img_index = int(value) - 1
        intrudor = img_list[img_index]
        intrudor_clust = img_to_clust[intrudor]
        answers[img_list] = intrudor_clust

    ground_truth = dict()
    for hit in intruder_hits:
        url_and_cluster = [(x, img_to_clust[x]) for x in hit]
        key = tuple(x[0] for x in url_and_cluster)
        if len(Counter(x[1] for x in url_and_cluster).most_common()) != 1:
            intruder_cluster = Counter(x[1] for x in url_and_cluster).most_common()[1][0]
        else:
            intruder_cluster = Counter(x[1] for x in url_and_cluster).most_common()[0][0]
        main_cluster = Counter(x[1] for x in url_and_cluster).most_common()[0][0]
        ground_truth[key] = (intruder_cluster, main_cluster)

    gt = Counter(x[1] for x in ground_truth.values())
    clusters_accuracy = dict()
    main_clusters = dict()
    for images, answer in answers.items():
        main_cluster = ground_truth[images][1]
        intruder_cluster = ground_truth[images][0]
        if main_cluster not in clusters_accuracy:
            clusters_accuracy[main_cluster] = 0
        if intruder_cluster == answer:
            clusters_accuracy[main_cluster] += 1

    print('mean accuracy:', sum(clusters_accuracy.values()) / len(ground_truth))

    for clust in clusters_accuracy:
        clusters_accuracy[clust] /= gt[clust]

    plt.figure(figsize=(8, 5))
    plt.bar(clusters_accuracy.keys(), clusters_accuracy.values(), tick_label=list(clusters_accuracy.keys()))
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.ylabel('accuracy')
    plt.xlabel('main cluster')
    plt.title('Accuracy for each "main" cluster')