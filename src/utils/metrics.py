import numpy as np


def discounted_cumulative_gain(relevances, k):
    relevances = np.asanyarray(relevances)[:k]
    if relevances.size:
        denominators = np.log2(np.arange(2, relevances.size + 2))

        # calculate the numerators: 2^relevances - 1
        numerators = 2**relevances - 1

        # perform element-wise division and then sum
        dcg = np.sum(numerators / denominators)
        return float(dcg)

    return 0.0


def normalized_dcg(y_true, y_score, k):
    # calculate dcg for teacher
    ideal_order = np.argsort(y_true)[::-1]
    ideal_dcg = discounted_cumulative_gain(np.take(y_true, ideal_order), k)

    if ideal_dcg == 0.0:
        return 0.0

    # calculate normalized dcg for students
    predicted_order = np.argsort(y_score)[::-1]
    predicted_dcg = discounted_cumulative_gain(np.take(y_true, predicted_order), k)

    return predicted_dcg / ideal_dcg


def mean_reciprocal_rank(relevance):
    for i, rel in enumerate(relevance, start=1):
        if rel:
            return 1.0 / i

    return 0.0


def precision_at_k(relevance, k):
    return np.sum(relevance[:k]) / k
