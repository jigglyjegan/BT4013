from similarity_metric.rep_metric_cosine_similarity import *
import streamlit as st
import math
from topic_models.data import *
from topic_models.bertopic import *
from topic_models.lda import *
from topic_models.top2vec import *
from topic_models.nmf import *
import streamlit.components.v1 as components

st.set_page_config(layout="wide")

with open("styles.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

if "currentPage" not in st.session_state:
    st.session_state.currentPage = "main_page"

st.title("Text Pre Processing")
main_page = st.empty()
insight_page = st.empty()

# For changing pages
def change_page(page):
    st.session_state.currentPage = page


# For input widget within insight page to get number of sample per topic
def set_topic_model(model):
    total_sample_size = st.session_state["number_of_topics"] * st.session_state["k"]
    size_of_data_set = len(st.session_state["docs"])
    if total_sample_size < size_of_data_set:
        if st.session_state["k"]:
            st.session_state.topicModel = model
            change_page("download_page")
        else:
            st.warning("Set number of topics.")
    else:
        st.warning(
            "Please set a lower number of samples per topic. The max you can set is: "
            + str(size_of_data_set)
        )


# For checkbox widget to toggle usage of model
def set_model_usage(
    session_state_name, current_session_state_value, model_decide_topics_session_state
):
    if current_session_state_value:
        st.session_state[session_state_name] = False
    else:
        # Case where the person selects model decide but then wants to add nmf/lda
        if session_state_name == "use_lda" or session_state_name == "use_nmf":
            st.session_state["model_decide_topics"] = False
        st.session_state[session_state_name] = True


def allow_model_to_decide(model_decide_topics_session_state):
    if model_decide_topics_session_state:
        st.session_state["model_decide_topics"] = False
    else:
        st.session_state["use_lda"] = False
        st.session_state["use_nmf"] = False
        st.session_state["use_bert"] = True
        st.session_state["use_top2vec"] = True
        st.session_state["model_decide_topics"] = True


# Main page
if st.session_state.currentPage == "main_page":
    main_page = st.container()
    with main_page:
        col1, col2, col3 = st.columns([0.5, 0.1, 0.5])
        col2.image("csv.png", use_column_width=True)
        st.markdown(
            "<h2 style='text-align: center;font-size: 24px;'>Preprocess your text data</h2>",
            unsafe_allow_html=True,
        )

        # Initalise session states for model usage, defaults to True.
        if "use_bert" not in st.session_state:
            st.session_state["use_bert"] = True
        if "use_lda" not in st.session_state:
            st.session_state["use_lda"] = True
        if "use_top2vec" not in st.session_state:
            st.session_state["use_top2vec"] = True
        if "use_nmf" not in st.session_state:
            st.session_state["use_nmf"] = True

        # Initalise session state for model deciding number of topics:
        if "model_decide_topics" not in st.session_state:
            st.session_state["model_decide_topics"] = False

        # Checkboxes for selecting which models to use
        use_bert, use_lda, use_top2vec, use_nmf = st.columns([1, 1, 1, 1])
        use_bert.checkbox(
            "Use BERTopic Model",
            value=st.session_state["use_bert"],
            on_change=set_model_usage,
            args=(
                "use_bert",
                st.session_state["use_bert"],
                st.session_state["model_decide_topics"],
            ),
        )
        use_lda.checkbox(
            "Use LDA Model",
            value=st.session_state["use_lda"],
            on_change=set_model_usage,
            args=(
                "use_lda",
                st.session_state["use_lda"],
                st.session_state["model_decide_topics"],
            ),
            disabled=st.session_state["model_decide_topics"],
        )
        use_top2vec.checkbox(
            "Use Top2Vec Model",
            value=st.session_state["use_top2vec"],
            on_change=set_model_usage,
            args=(
                "use_top2vec",
                st.session_state["use_top2vec"],
                st.session_state["model_decide_topics"],
            ),
        )
        use_nmf.checkbox(
            "Use NMF Model",
            value=st.session_state["use_nmf"],
            on_change=set_model_usage,
            args=(
                "use_nmf",
                st.session_state["use_nmf"],
                st.session_state["model_decide_topics"],
            ),
            disabled=st.session_state["model_decide_topics"],
        )

        # Input for number of topics
        if st.session_state["model_decide_topics"]:
            number_of_topics = st.number_input(
                "Insert number of Topics, decimals will be rounded down.",
                disabled=True,
            )
        else:
            number_of_topics = st.number_input(
                "Insert number of Topics, decimals will be rounded down.",
                min_value=1,
                max_value=999,
                value=3,
            )

        # Allow model to decide number of topics
        st.checkbox(
            "Allow model to decide number of topics. Note that only the Top2Vec and BERTopic allow this feature.",
            value=st.session_state["model_decide_topics"],
            on_change=allow_model_to_decide,
            args=(st.session_state["model_decide_topics"],),
        )

        # File uploader
        uploaded_file = st.file_uploader("", type=["csv", "xlsx"], key="enabled")

        # add logic to ensure that number of topics is not None
        if uploaded_file is not None:
            if st.session_state["model_decide_topics"]:
                col1, col2 = st.columns([0.5, 0.5])
                if st.session_state["use_bert"]:
                    col1.write("Awaiting BERTopic Process to Begin")
                if st.session_state["use_top2vec"]:
                    col2.write("Awaiting Top2Vec Process to Begin")

                df = load_data(uploaded_file, uploaded_file.name)
                df, docs, docs_tokenized = preprocess_data(df)
                st.session_state["dataframe"] = df
                st.session_state["docs"] = docs
                st.session_state["docs_tokenized"] = docs_tokenized

                # Bert logic
                if st.session_state["use_bert"]:
                    col1.write("Running BERTopic.....")
                    bert = run_bertopic_auto(docs)
                    st.session_state["bert"] = bert
                    col1.write("BERTopic Model Completed")

                # top2vec logic
                if st.session_state['use_top2vec']:
                    col2.write("Running Top2Vec.....")
                    top2vec = runTop2Vec(docs)
                    st.session_state["top2vec"] = top2vec
                    col2.write("Top2Vec Model Completed")

                insight1, insight2, insight3 = st.columns([1, 0.5, 1])
                insight = insight2.button(
                    "Click here to focus on the insights that has be found!",
                    on_click=change_page,
                    args=("insight_page",),
                )

            elif number_of_topics:

                # Column for in progress text
                col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25])
                if st.session_state["use_bert"]:
                    col1.write("Awaiting BERTopic Process to Begin")
                if st.session_state["use_lda"]:
                    col2.write("Awaiting LDA  Process to Begin")
                if st.session_state["use_top2vec"]:
                    col3.write("Awaiting Top2Vec Process to Begin")
                if st.session_state["use_nmf"]:
                    col4.write("Awaiting NMF Process to Begin")

                number_of_topics = math.floor(number_of_topics)
                st.session_state["number_of_topics"] = number_of_topics

                df = load_data(uploaded_file, uploaded_file.name)
                df, docs, docs_tokenized = preprocess_data(df)
                st.session_state["dataframe"] = df
                st.session_state["docs"] = docs
                st.session_state["docs_tokenized"] = docs_tokenized

                # Bert logic
                if st.session_state["use_bert"]:
                    col1.write("Running BERTopic.....")
                    if st.session_state["model_decide_topics"]:
                        bert = run_bertopic_auto(docs)
                    else:
                        bert = run_bertopic(docs, number_of_topics)
                    st.session_state["bert"] = bert
                    col1.write("BERTopic Model Completed")

                # Lda logic
                if st.session_state["use_lda"]:
                    col2.write("Running LDA.....")
                    lda_model, bow_corpus, dictionary = run_lda(
                        docs_tokenized, number_of_topics
                    )
                    st.session_state["lda"] = lda_model
                    st.session_state["bow_corpus"] = bow_corpus
                    st.session_state["lda_dictionary"] = dictionary
                    col2.write("LDA Model Completed")

                # top2vec logic
                if st.session_state['use_top2vec']:
                    col3.write("Running Top2Vec.....")
                    top2vec = runTop2Vec(docs)
                    runTop2VecReduced(top2vec, number_of_topics)
                    st.session_state["top2vec"] = top2vec
                    col3.write("Top2Vec Model Completed")

                # nmf logic
                if st.session_state["use_nmf"]:
                    col4.write("Running NMF.....")
                    nmf, tfidf_feature_names, W, H = run_nmf(docs, number_of_topics)
                    st.session_state["nmf"] = nmf
                    st.session_state["tfidf_feature_names"] = tfidf_feature_names
                    st.session_state["running_nmf"] = False
                    st.session_state["W"] = W
                    st.session_state["H"] = H
                    col4.write("NMF Model Completed")

                insight1, insight2, insight3 = st.columns([1, 0.5, 1])
                insight = insight2.button(
                    "Click here to focus on the insights that has be found!",
                    on_click=change_page,
                    args=("insight_page",),
                )
            else:
                st.warning("Please insert the number of topics.g")

# Insights page
if st.session_state["currentPage"] == "insight_page":
    insight_page = st.container()
    if not st.session_state["model_decide_topics"]:
        number_of_topics = st.session_state["number_of_topics"]

    with insight_page:

        # WordCloud
        st.write("Word Cloud for Entire Dataset")
        col1, col2, col3 = st.columns([1, 1, 1])
        col2.pyplot(wordcloud(st.session_state["docs_tokenized"]))

        # BERT
        if st.session_state["use_bert"]:
            bert = st.session_state["bert"]
            bert_expander = st.expander("BERTopic")
            bert_expander.write(
                bert.visualize_barchart().update_layout(
                    autosize=False, width=670, height=400
                )
            )

        # Top2Vec
        if st.session_state['use_top2vec']:
            top2vec = st.session_state['top2vec']
            top2vec_expander = st.expander("Top2Vec")
            if st.session_state["model_decide_topics"]:
                top2Vec_num_topics = top2vec.get_num_topics()
                for i in range(top2Vec_num_topics):
                    if i == 10:
                        break
                    fig = printWordBar(top2vec, i)
                    top2vec_expander.plotly_chart(fig, use_container_width=True)
            else:
                for i in range(number_of_topics):
                    if i == 10:
                        break
                    fig = printWordBarReduced(top2vec, i)
                    top2vec_expander.plotly_chart(fig, use_container_width=True)


        # LDA
        if st.session_state["use_lda"]:
            lda = st.session_state["lda"]
            with st.expander("LDA"):
                col1, col2, col3 = st.columns([0.5, 0.1, 0.5])
                components.html(
                    visualize_chart_lda(
                        lda,
                        st.session_state["bow_corpus"],
                        st.session_state["lda_dictionary"],
                    ),
                    width=1300,
                    height=800,
                    scrolling=True,
                )

        # NMF
        if st.session_state["use_nmf"]:
            nmf = st.session_state["nmf"]
            tfidf_feature_names = st.session_state["tfidf_feature_names"]
            NMF_expander = st.expander("NMF")
            NMF_expander.pyplot(
                plot_top_words_v2(
                    nmf,
                    tfidf_feature_names,
                    10,
                    "Topics in NMF model (KL Divergence Loss)",
                )
            )

        # Ask for how many datapoints you want her topic, k.
        k = st.number_input(
            "Insert number of datapoints, you want for each topic, decimals will be rounded down.",
            min_value=1,
            max_value=100,
            value=5,
        )
        st.session_state["k"] = k

        # Process sampled dataset and similarity score
        if st.session_state["use_bert"]:
            bert = st.session_state["bert"]
            bert_sample_df = get_top_docs_bert(st.session_state["dataframe"], bert, k)
            bert_labeled_csv = df_to_csv(bert_sample_df)

        if st.session_state['use_top2vec']:
            top2vec = st.session_state['top2vec']
            if st.session_state["model_decide_topics"]:
                top2vec_sample_df = get_top_documents_Top2Vec(st.session_state["dataframe"], top2vec, top2vec.get_num_topics(), k)
            else:
                top2vec_sample_df = get_top_documents_Top2Vec_reduced(st.session_state["dataframe"], top2vec, number_of_topics, k)
            top2vec_labeled_csv = df_to_csv(top2vec_sample_df)

        if st.session_state["use_lda"]:
            lda = st.session_state["lda"]
            lda_sample_df = get_top_documents_lda(
                st.session_state["dataframe"],
                st.session_state["bow_corpus"],
                lda,
                st.session_state["number_of_topics"],
                k,
            )
            lda_labeled_csv = df_to_csv(lda_sample_df)

        if st.session_state["use_nmf"]:
            nmf = st.session_state["nmf"]
            nmf_sample_df = get_top_docs_nmf(
                st.session_state["docs"],
                st.session_state["W"],
                st.session_state["H"],
                st.session_state["tfidf_feature_names"],
                k,
            )
            nmf_labeled_csv = df_to_csv(nmf_sample_df)

        # Similarity Scores
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        if st.session_state["use_bert"]:
            if st.session_state["model_decide_topics"]:
                num_topics = col1.write(
                    "Number of topics decided by BERTopic model: {}".format(get_number_of_topics_bert(bert))
                )

            bert_similarity_score = col1.write(
                "Similarity Score: "
                + "{:.2f}".format(
                    run_representative_sample_test(
                        get_all_docs_bert(st.session_state["docs"], bert),
                        bert_sample_df,
                    )[1]
                )
            )

            bert_similarity_percentage = col1.write(
                "Similarity Percentage: "
                + "{:.2f}".format(
                    run_representative_sample_test(
                        get_all_docs_bert(st.session_state["docs"], bert),
                        bert_sample_df,
                    )[2]
                )
                + "%"
            )

        if st.session_state["use_top2vec"]:
            if st.session_state["model_decide_topics"]:
                df_all_top2vec = get_all_docs_top2vec(st.session_state["docs"], top2vec)
                num_topics = col2.write(
                    "Number of topics decided by Top2Vec model: {}".format(top2vec.get_num_topics())
                )
            else: 
                df_all_top2vec = get_all_docs_top2vec_reduced(st.session_state["docs"], top2vec)

            top2vec_similarity_score = col2.write(
                "Similarity Score: "
                + "{:.2f}".format(
                    run_representative_sample_test(
                        df_all_top2vec,
                        top2vec_sample_df,
                    )[1]
                )
            )

            top2vec_similarity_percentage = col2.write(
                "Similarity Percentage: "
                + "{:.2f}".format(
                    run_representative_sample_test(
                        df_all_top2vec,
                        top2vec_sample_df,
                    )[2]
                )
                + "%"
            )

        if st.session_state["use_lda"]:
            lda_similarity_score = col3.write(
                "Similarity Score: "
                + "{:.2f}".format(
                    run_representative_sample_test(
                        get_all_docs_lda(
                            st.session_state["dataframe"],
                            st.session_state["bow_corpus"],
                            lda,
                        ),
                        lda_sample_df,
                    )[1]
                )
            )

            lda_similarity_percentage = col3.write(
                "Similarity Percentage: "
                + "{:.2f}".format(
                    run_representative_sample_test(
                        get_all_docs_lda(
                            st.session_state["dataframe"],
                            st.session_state["bow_corpus"],
                            lda,
                        ),
                        lda_sample_df,
                    )[2]
                )
                + "%"
            )

        if st.session_state["use_nmf"]:
            nmf_similarity_score = col4.write(
                "Similarity Score: "
                + "{:.2f}".format(
                    run_representative_sample_test(
                        get_all_docs_nmf(
                            st.session_state["docs"],
                            st.session_state["W"],
                            st.session_state["H"],
                            st.session_state["tfidf_feature_names"],
                        ),
                        nmf_sample_df,
                    )[1]
                )
            )

            nmf_similarity_percentage = col4.write(
            "Similarity Percentage: "
            + "{:.2f}".format(
                run_representative_sample_test(
                    get_all_docs_nmf(
                        st.session_state["docs"],
                        st.session_state["W"],
                        st.session_state["H"],
                        st.session_state["tfidf_feature_names"],
                    ),
                    nmf_sample_df,
                )[2]
            )
            + "%"
            )

        # Generate buttons to go to download sample csv
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        if st.session_state["use_bert"]:
            generate_with_bert = col1.download_button(
                label="Download dataset generated with BERTopic",
                data=bert_labeled_csv,
                file_name="bertopic_output.csv",
                mime="text/csv",
            )
        if st.session_state["use_top2vec"]:
            generate_with_top2vec = col2.download_button(
                "Download dataset generated with Top2Vec",
                data=top2vec_labeled_csv,
                file_name="top2vec_output.csv",
                mime="text/csv",
            )    
        if st.session_state["use_lda"]:
            generate_with_lda = col3.download_button(
                label="Download dataset generated with LDA",
                data=lda_labeled_csv,
                file_name="lda_output.csv",
                mime="text/csv",
            )
        if st.session_state["use_nmf"]:
            generate_with_nmf = col4.download_button(
                label="Download dataset generated with NMF",
                data=nmf_labeled_csv,
                file_name="nmf_output.csv",
                mime="text/csv",
            )

        go_back = st.button(
            "Go Back to Main Page", on_click=change_page, args=("main_page",)
        )