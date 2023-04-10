import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer


def run_bertopic(docs, num_topics):
    """
    Runs BERTopic on provided documents (docs) and outputs topics (num_topics)

    Args:
    docs -> List of documents
    num_topics -> int

    Returns:
    - Trained "model" that can be used to return visualizations and stats
    """
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    vectorizer_model = CountVectorizer(stop_words="english")

    model = BERTopic(
        top_n_words=10,
        nr_topics=num_topics,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
    )

    topics, probabilities = model.fit_transform(docs)
    
    return model


def run_bertopic_auto(docs):
    """
    Runs BERTopic on provided documents (docs) and outputs topics (num_topics)

    Args:
    docs -> List of documents

    Returns:
    - Trained "model" that can be used to return visualizations and stats
    """
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    vectorizer_model = CountVectorizer(stop_words="english")

    model = BERTopic(
        top_n_words=10,
        nr_topics="auto",
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
    )

    topics, probabilities = model.fit_transform(docs)

    return model

def get_number_of_topics_bert(model):
    """
    Args:
    - model: bertopic model

    Returns:
    - Number of topics (-1 because noise is also counted as a topic)
    """
    return len(model.get_topics()) - 1


def get_top_docs_bert(df, model, k):
    """
    Args:
    - df: pandas dataframe
    - model: bertopic model
    - k: how many sample to be extracted per topic

    Returns:
    - samples: Array of sample documents
    - topic_labels: Array of corresponding topic labels
    - topic_words: String of representative topic_words for topic
    """
    samples = []
    topic_labels = []
    topic_words = []
    representative_docs = model.representative_docs_
    for topic_num, documents in representative_docs.items():
        topic_word = " ".join(list(map(lambda x: x[0], model.get_topic(topic_num))))
        for index, doc in enumerate(documents):
            if index > k:
                break
            sample = df.loc[df["processed"] == doc]["text"].values[0]
            samples.append(sample)
            topic_labels.append(topic_num)
            topic_words.append(topic_word)

    data = {"doc": samples, "topic_label": topic_labels, "topic_words": topic_words}
    return pd.DataFrame(data)


def get_all_docs_bert(docs, model):
    """
    Args:
    - docs: List of documents
    - model: bertopic model

    Returns:
    - Dataframe with columns 'doc', 'topic_label'. This is all docs from the dataset (docs)
    If topic label is -1 then it means it was classified as noise
    """
    data = {"doc": docs, "topic_label": model.topics_}
    return pd.DataFrame(data)
