import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.feature_extraction.text import CountVectorizer


def _get_count_vectorizer():
    cv_params = {}
    return CountVectorizer(**cv_params)


def _cosine_similarity(a_i, a_j):
    cs = np.dot(a_i, a_j) / (np.linalg.norm(a_i) * np.linalg.norm(a_j))
    return cs


def _cosine_distance(a_i, a_j):
    cs = _cosine_similarity(a_i, a_j)
    cd = 1 - cs
    return cd


def run_representative_sample_test(all, samples, penalty=0):
    """
    Parameters
    ----------
    all : pd.DataFrame
        df with columns 'doc', 'topic_label'.
        this df contains all the docs.

    samples : pd.DataFrame
        df with columns 'doc', 'topic_label'.
        this df only contains the samples.

    penalty: float
        the penalty term for words that are in the all df and not in the samples df.
        Defaults to 0.

    Returns
    ----------
    metrics : tuple
        a tuple containing the final Cosine Similarity, Cosine Distance and Representative Percentage.
    """
    cv = _get_count_vectorizer()
    all_cv = cv.fit_transform(all.doc)

    # Get words and word counts for all docs
    all_word_ct = pd.DataFrame(pd.DataFrame(all_cv.todense()).sum(axis=0))
    all_word_ct.columns = ["count"]
    all_idx_word_map = pd.DataFrame(
        {"word": cv.vocabulary_.keys()}, index=cv.vocabulary_.values()
    ).sort_index()
    all_idx_word_count_map = pd.merge(
        all_idx_word_map, all_word_ct, how="left", left_index=True, right_index=True
    )
    all_idx_word_count_map.columns = ["all_word", "all_word_count"]

    topics_cs_list = []
    topics_cd_list = []

    for topic_label in all.topic_label.unique():
        cv = _get_count_vectorizer()
        samples_cv = cv.fit_transform(samples.doc)

        # Get words and word counts for the sample docs
        samples_word_ct = pd.DataFrame(pd.DataFrame(samples_cv.todense()).sum(axis=0))
        samples_word_ct.columns = ["count"]
        samples_idx_word_map = pd.DataFrame(
            {"word": cv.vocabulary_.keys()}, index=cv.vocabulary_.values()
        ).sort_index()
        samples_idx_word_count_map = pd.merge(
            samples_idx_word_map,
            samples_word_ct,
            how="left",
            left_index=True,
            right_index=True,
        )
        samples_idx_word_count_map.columns = ["samples_word", "samples_word_count"]

        # Map all docs to sample docs by the word
        tmp = pd.merge(
            all_idx_word_count_map,
            samples_idx_word_count_map,
            how="left",
            left_on="all_word",
            right_on="samples_word",
        )

        # Apply penalty to samples
        tmp.samples_word_count = tmp.samples_word_count.fillna(penalty)

        # Get comparison arrays and compute cosine distance
        original_v = np.asarray(tmp.all_word_count)
        sample_v = np.asarray(tmp.samples_word_count)

        # Cosine Similarity
        this_topic_cs = _cosine_similarity(original_v, sample_v)
        topics_cs_list.append(this_topic_cs)

        # Cosine Distance
        this_topic_cd = _cosine_distance(original_v, sample_v)
        topics_cd_list.append(this_topic_cd)

    # Get final Cosine Similarity
    final_cs = sum(topics_cs_list) / len(topics_cs_list)

    # Get final cosine distance
    final_cd = sum(topics_cd_list) / len(topics_cd_list)

    # Final representative level metric as a pct
    repr_pct = ((2 - final_cd) / 2) * 100

    metrics = (final_cs, final_cd, repr_pct)

    return metrics
