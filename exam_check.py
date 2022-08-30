import logging
from decimal import Decimal

from toloka.client import TolokaClient, Pool, Skill

from crowd_clustering_aggregation.aggregation import get_assignments_list


def exam_check(toloka_client: TolokaClient, exam_pool: Pool, skill: Skill, n_exams: int = 3):
    assignments_raw = toloka_client.get_assignments_df(pool_id=exam_pool.id, status=['SUBMITTED'])[
        ['INPUT:images', 'OUTPUT:result', 'GOLDEN:result', 'ASSIGNMENT:link', 'ASSIGNMENT:assignment_id',
         'ASSIGNMENT:worker_id', 'ASSIGNMENT:status']]

    _, assignments_list = get_assignments_list(assignments_raw, input_column='INPUT:images',
                                               golden_column='GOLDEN:result')

    for assignment in assignments_list:
        skill_value = 0
        correct_clusters = len(assignment.clusters & assignment.golden_clusters)
        ground_truth_clusters = len(assignment.golden_clusters)

        if assignment.clusters == assignment.golden_clusters:
            try:
                toloka_client.accept_assignment(assignment.assignment_id, 'Excellent')
            except Exception as e:
                logging.info(e)
            skill_value = 100

        if correct_clusters == 0:
            try:
                toloka_client.reject_assignment(assignment.assignment_id, 'Not correct')
            except Exception as e:
                logging.info(e)
            skill_value = 0

        if correct_clusters != 0 and correct_clusters != ground_truth_clusters:
            try:
                toloka_client.reject_assignment(assignment.assignment_id, 'Partly correct')
            except Exception as e:
                logging.info(e)
            skill_value = Decimal(correct_clusters / ground_truth_clusters * 100)

        try:
            current_value = list(toloka_client.get_user_skills(user_id=assignment.worker_id,
                                                               skill_id=skill.id))[0].exact_value or 0
        except Exception as e:
            logging.info(e)
            current_value = 0

        toloka_client.set_user_skill(
            skill_id=skill.id, user_id=assignment.worker_id,
            value=Decimal(min(float(current_value) + float(skill_value / n_exams), 100))
        )
