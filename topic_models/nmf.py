from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

### NOTE: Important, need to convert the processed docs to array before
# inputting into NMF

# docs_arr = np.asarray(docs)


def _get_tfidf_vectorizer():
    tfidf_params = {"min_df": 0.0008, "max_df": 0.90, "max_features": 500, "norm": "l1"}
    return TfidfVectorizer(**tfidf_params)


def run_nmf(docs, num_topics):
    """
    W is the Document-Topic matrix.
    Each row in W represents the Document and the entries represents the Document's rank in a Topic.
    H is the Topic-Word matrix (weighting).
    Each column in H represents a Word and the entries represents the Word's rank in a Topic.
    Matrix multiplication of the factored components, W x H results in the input Document-Word matrix.

    Parameters
    ----------
    docs : np.array
        An array of documents. Note that each document is a string of the processed text.

    num_topics : int
        Number of topics to learn.

    Returns
    ----------
    nmf : sklearn.estimator
        The fitted nmf sklearn estimator instance.

    tfidf_feature_names: list[str]
        Vocabulary to aid visualisation.

    W: np.array
        Document-Topic matrix.

    H: np.array
        Topic-Word matrix.

    """
    nmf_params = {
        "n_components": num_topics,
        "alpha_W": 3.108851387228361e-05,
        "alpha_H": 8.312434671077156e-05,
        "l1_ratio": 0.3883534426209613,
        "beta_loss": "kullback-leibler",
        "init": "nndsvda",
        "solver": "mu",
        "max_iter": 1000,
        "random_state": 4013,
        "tol": 0.0001,
    }

    tfidf_vectorizer = _get_tfidf_vectorizer()
    tfidf = tfidf_vectorizer.fit_transform(docs)

    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

    nmf = NMF(**nmf_params)
    nmf.fit(tfidf)

    W = nmf.fit_transform(tfidf)
    H = nmf.components_

    return nmf, tfidf_feature_names, W, H


def plot_top_words(
    model, feature_names, n_top_words=10, title="Topics in NMF Model with n Top Words"
):
    """
    Parameters
    ----------
    model : sklearn.estimator
        The fitted nmf estimator.

    feature_names : np.array
        The feature names used for training (Selected by TF-IDF Vectorizer).

    n_top_words : int
        The number of top words to show for each topic in plot.

    title : str
        The main title of the plot.
    """
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


def plot_top_words_v2(
    model, feature_names, n_top_words=10, title="Topics in NMF Model with n Top Words"
):
    """
    Parameters
    ----------
    model : sklearn.estimator
        The fitted nmf estimator.

    feature_names : np.array
        The feature names used for training (Selected by TF-IDF Vectorizer).

    n_top_words : int
        The number of top words to show for each topic in plot.

    title : str
        The main title of the plot.
    """
    H = model.components_
    num_topics = H.shape[0]

    for start_topic_idx in range(0, num_topics, 5):
        end_topic_idx = min(start_topic_idx + 5, num_topics)
        H_w = H[start_topic_idx:end_topic_idx]

        plot_w = 5
        if end_topic_idx == num_topics:
            plot_w = H_w.shape[0]

        fig, axes = plt.subplots(1, plot_w, figsize=(30, 15), sharex=True)
        axes.flatten()

        for topic_idx, topic in enumerate(H_w):
            top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
            top_features = [feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]

            ax = axes[topic_idx]
            ax.barh(top_features, weights, height=0.7)
            ax.set_title(
                f"Topic {start_topic_idx + topic_idx + 1}", fontdict={"fontsize": 30}
            )
            ax.invert_yaxis()
            ax.tick_params(axis="both", which="major", labelsize=20)
            for i in "top right left".split():
                ax.spines[i].set_visible(False)

            if start_topic_idx == 0:
                fig.suptitle(title, fontsize=40)

        plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
        plt.show()


def get_all_docs_nmf(docs, W, H, feature_names):
    """
    Parameters
    ----------
    docs : np.array
        An array of documents. Note that each document is a string of the processed text.
        This is same as input into run_nmf.

    W : np.array
        Document-Topic matrix.

    H : np.array
        Topic-Word matrix.

    feature_names : np.array
        The feature names used for training (Selected by TF-IDF Vectorizer).

    Returns
    ----------
    doc_topic_df : pd.DataFrame
        df with 4 columns, 'doc', 'topic_label', 'topic_score' and 'top_words'.
    """
    doc_topic_df = pd.DataFrame(W)

    num_topics = doc_topic_df.shape[1]

    doc_topic_df["topic_label"] = doc_topic_df.apply(lambda r: r.argmax(), axis=1)
    doc_topic_df["topic_score"] = doc_topic_df.apply(
        lambda r: r[:num_topics].max(), axis=1
    )
    doc = list(docs[doc_topic_df.index.start : doc_topic_df.index.stop])
    doc_topic_df["doc"] = doc
    doc_topic_df["top_words"] = ""

    for topic_idx, topic in enumerate(H):
        n_top_words = 10
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = ",".join([feature_names[i] for i in top_features_ind])
        doc_topic_df.top_words[doc_topic_df.topic_label == topic_idx] = top_features

    doc_topic_df = doc_topic_df[["doc", "topic_label", "topic_score", "top_words"]]

    return doc_topic_df


def get_top_docs_nmf(docs, W, H, feature_names, k):
    """
    Parameters
    ----------
    docs : np.array
        An array of documents. Note that each document is a string of the processed text.
        This is same as input into run_nmf.

    W : np.array
        Document-Topic matrix.

    H : np.array
        Topic-Word matrix.

    feature_names : np.array
        The feature names used for training (Selected by TF-IDF Vectorizer).

    k : int
        The top k number of docs will be taken from each topic's docs.

    Returns
    ----------
    top_docs_df : pd.DataFrame
        df with 4 columns, 'doc', 'topic_label', 'topic_score' and 'top_words'.
        df only contains the top k docs with highest topic scores.
    """
    df = get_all_docs_nmf(docs, W, H, feature_names)

    tmp = df.groupby("topic_label").topic_score.nlargest(k)
    docs_idx = []
    for topic_label in df.topic_label.unique():
        docs_idx.extend(list(tmp[topic_label].index))

    top_docs_df = df.iloc[docs_idx]

    return top_docs_df
