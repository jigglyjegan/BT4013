import pandas as pd
from top2vec import Top2Vec
import matplotlib.pyplot as plt
import plotly.express as px

# docs is a string array of documents, numTopic is an integer
def runTop2Vec(docs):
    return Top2Vec(docs)


def runTop2VecReduced(model, numTopics):
    return model.hierarchical_topic_reduction(numTopics)


# get topic words for all topics
def getTopicWords(model):
    return model.topic_words_reduced


# print all wordclouds
def printWordCloud(model, numTopic):
    for i in range(numTopic):
        Top2Vec.generate_topic_wordcloud(
            model, i, background_color="black", reduced=True
        )


# print topic word score barchart
def printWordBarReduced(model, numTopic):
    # for i in range(numTopic):
    topic_names = model.topic_words_reduced[numTopic][:5]
    topic_probs = model.topic_word_scores_reduced[numTopic][:5]
    df_topics = pd.DataFrame(topic_names).rename(columns={0: "Topic Words"})
    df_probs = pd.DataFrame(topic_probs).rename(columns={0: "Probability"})
    df = pd.concat([df_topics, df_probs], axis=1)
    fig = px.bar(df, y="Probability", x="Topic Words", text_auto=".2s", title=numTopic)
    fig.update_traces(
        textfont_size=12, textangle=0, textposition="outside", cliponaxis=False
    )
    return fig


def printWordBar(model, numTopic):
    # for i in range(numTopic):
    topic_names = model.topic_words[numTopic][:5]
    topic_probs = model.topic_word_scores[numTopic][:5]
    df_topics = pd.DataFrame(topic_names).rename(columns={0: "Topic Words"})
    df_probs = pd.DataFrame(topic_probs).rename(columns={0: "Probability"})
    df = pd.concat([df_topics, df_probs], axis=1)
    fig = px.bar(
        df,
        y="Probability",
        x="Topic Words",
        text_auto=".2s",
        title="Topic " + str(numTopic),
    )
    fig.update_traces(
        textfont_size=12, textangle=0, textposition="outside", cliponaxis=False
    )
    return fig


def printWordBar(model, numTopic):
    # for i in range(numTopic):
    topic_names = model.topic_words[numTopic][:5]
    topic_probs = model.topic_word_scores[numTopic][:5]
    df_topics = pd.DataFrame(topic_names).rename(columns={0: "Topic Words"})
    df_probs = pd.DataFrame(topic_probs).rename(columns={0: "Probability"})
    df = pd.concat([df_topics, df_probs], axis=1)
    fig = px.bar(
        df,
        y="Probability",
        x="Topic Words",
        text_auto=".2s",
        title="Topic " + str(numTopic),
    )
    fig.update_traces(
        textfont_size=12, textangle=0, textposition="outside", cliponaxis=False
    )
    return fig


def get_top_documents_Top2Vec(df, model, num_topics, k):
    samples = []
    topic_labels = []
    topic_words = []
    for topic_num in range(0, num_topics):
        documents, document_scores, document_ids = model.search_documents_by_topic(
            topic_num=topic_num, num_docs=k
        )
        for index, doc in enumerate(documents):
            if index > k:
                break
            sample = df.loc[df["processed"] == doc]["text"].values[0]
            samples.append(sample)
        for label in range(0, k):
            topic_labels.append(topic_num)
            topic_words.append(model.topic_words[topic_num][:10])
    data = {"doc": samples, "topic_label": topic_labels, "topic_words": topic_words}
    return pd.DataFrame(data)


def get_top_documents_Top2Vec_reduced(df, model, num_topics, k):
    samples = []
    topic_labels = []
    topic_words = []
    for topic_num in range(0, num_topics):
        documents, document_scores, document_ids = model.search_documents_by_topic(
            topic_num=topic_num, num_docs=k, reduced=True
        )
        for index, doc in enumerate(documents):
            if index > k:
                break
            sample = df.loc[df["processed"] == doc]["text"].values[0]
            samples.append(sample)
        for label in range(0, k):
            topic_labels.append(topic_num)
            topic_words.append(model.topic_words_reduced[topic_num][:10])
    data = {"doc": samples, "topic_label": topic_labels, "topic_words": topic_words}
    return pd.DataFrame(data)


def get_all_docs_top2vec_reduced(docs, model):
    id_array = []
    for i in range(0, len(docs)):
        id_array.append(i)
    topic_nums = model.get_documents_topics(id_array, reduced=True)[0]
    data = {"doc": docs, "topic_label": topic_nums}
    df_all = pd.DataFrame(data)
    return pd.DataFrame(data)


def get_all_docs_top2vec(docs, model):
    id_array = []
    for i in range(0, len(docs)):
        id_array.append(i)
    topic_nums = model.get_documents_topics(id_array)[0]
    data = {"doc": docs, "topic_label": topic_nums}
    df_all = pd.DataFrame(data)
    return pd.DataFrame(data)
