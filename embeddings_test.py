import requests
import json
import torch 
import os
import spacy 
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
import seaborn as sns

from sklearn.manifold import TSNE
from numpy.linalg import norm
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from nltk.corpus import stopwords
from string import punctuation
from scipy.spatial import distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from transformers import BertTokenizer, BertModel
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from boilerpy3 import extractors
from mpl_toolkits.mplot3d import Axes3D


DIMENSIONS = 300
GLOVE_VOCABULARY = set()

COSINE = 1
EUCLIDEAN = 2

relevant_urls = ["https://en.wikipedia.org/wiki/Atmospheric_science", "https://en.wikipedia.org/wiki/Earth_observation", "https://en.wikipedia.org/wiki/Remote_sensing", "https://en.wikipedia.org/wiki/Earth_science", "https://en.wikipedia.org/wiki/Flood" ] # "https://geo.arc.nasa.gov/"
irrelevant_urls = ["https://en.wikipedia.org/wiki/Music", "https://en.wikipedia.org/wiki/Almond", "https://en.wikipedia.org/wiki/Train", "https://en.wikipedia.org/wiki/Moth", "https://en.wikipedia.org/wiki/England"] #  "https://lasermania.com/", "https://carolinescakes.com/"
all_urls = relevant_urls + irrelevant_urls
max_url_length = max(len(url) for url in all_urls)  
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()

def load_glove_model(glove_file):
    embeddings = {}

    with open(glove_file, 'r', encoding="utf8") as file:
        for i, line in tqdm(enumerate(file)):
            split_line = line.split(" ")
            embedding = np.array(split_line[-DIMENSIONS:], dtype=np.float64)
            word = split_line[0]
            GLOVE_VOCABULARY.add(word)
            embeddings[word.lower()] = embedding

    return embeddings


def calc_embedding_from_tokens(glove_embedding, documents):
    combined_embedding = np.zeros(DIMENSIONS, dtype=np.float64)
    unknown_tokens = set()
    
    num_tokens = 0
    for token in documents:
        if token in GLOVE_VOCABULARY:
            combined_embedding += glove_embedding[token]
            num_tokens += 1
        else:
            # print(f"Token {token} doesnt exist in GloVe 42B uncased")
            unknown_tokens.add(token)
    with open("data/unknown_tokens.txt", "w+") as file:
        for token in unknown_tokens:
                file.write(token + "\n")

    return np.array(combined_embedding/num_tokens)


def calc_multiple_embeddings_from_tokens(glove_embeddings, documents):
    embeddings = []
    for doc in documents:
        embedding = np.zeros(DIMENSIONS, dtype=np.float64)
        for token in doc:
            if token in GLOVE_VOCABULARY:
                embedding += glove_embeddings[token]
        embeddings.append(embedding)
    return embeddings


def get_abstracts(filename, num_abstracts=None):
    abstracts = []
    with open(filename, "r") as file:
        data = json.load(file)

    for i, paper in enumerate(data):
        try:
            abstracts.append(paper["abstract"])
        except:
            pass
        if num_abstracts:
            if i > num_abstracts:
                break

    return abstracts


def process_documents_combined(documents, apply_stemming=False, remove_punctuation=True):
    tokenized_documents = [word_tokenize(text) for text in documents]
    lowered_documents = [word.lower() for doc in tokenized_documents for word in doc]

    stop_words = set(stopwords.words('english'))
    tokenized_documents = [word for word in lowered_documents if word not in stop_words]

    if remove_punctuation:
        tokenized_documents = [word for word in tokenized_documents if word not in punctuation]

    if apply_stemming:
        stemmer = PorterStemmer()
        return [stemmer.stem(token) for token in tokenized_documents]
    else:
        return tokenized_documents


def process_documents_seperate(documents):
    tokenized_docs = [word_tokenize(text) for text in documents]
    lowered_docs = []

    stop_words = set(stopwords.words('english'))
    for doc in tokenized_docs:
        lowered_doc = [word.lower() for word in doc if word not in stop_words]
        lowered_doc = [word for word in lowered_doc if word not in punctuation]
        lowered_docs.append(lowered_doc)

    return lowered_docs

def process_documents_tfidf(documents):
    tokenized_docs = [word_tokenize(text) for text in documents]
    lowered_docs = []

    stop_words = set(stopwords.words('english'))
    for doc in tokenized_docs:
        lowered_doc = [word.lower() for word in doc if word not in stop_words]
        lowered_doc = [word for word in lowered_doc if word not in punctuation]
        lowered_docs.append(" ".join(lowered_doc))

    return lowered_docs


def create_mean_embedding(glove_embeddings=None):
    if glove_embeddings is None:
        print("Loading GloVe embeddings")
        glove_embeddings = load_glove_model('data/glove.42B.300d/glove.42B.300d.txt')
    print("Loading documents")
    abstracts = get_abstracts("data/raw_paper_abstracts.json")
    print("Processing documents")
    processed_abstracts = process_documents_combined(abstracts)
    print("Creating document embedding")
    combined_embedding = calc_embedding_from_tokens(glove_embeddings, processed_abstracts)
    with open("data/corpus_embedding.txt", "w+") as file:
        file.write("combined ")
        combined_embedding.tofile(file, sep=" ")


def create_multiple_embeddings(glove_embeddings=None):
    if glove_embeddings is None:
        print("Loading GloVe embeddings")
        glove_embeddings = load_glove_model('data/glove.42B.300d/glove.42B.300d.txt')
    print("Loading documents")
    abstracts = get_abstracts("data/raw_paper_abstracts.json")
    print("Processing documents")
    processed_abstracts = process_documents_seperate(abstracts)
    print("Creating document embedding")
    embeddings = calc_multiple_embeddings_from_tokens(glove_embeddings, processed_abstracts)
    np.save("data/multiple_embeddings.npy", embeddings)


def calc_similarity(corpus_embedding, document_embedding, mode=EUCLIDEAN):
    if mode == COSINE:
        dot_product = np.dot(corpus_embedding, document_embedding)
        norm_corpus = norm(corpus_embedding)
        norm_doc = norm(document_embedding)
        return dot_product / (norm_corpus * norm_doc)
    elif mode == EUCLIDEAN:
        return distance.euclidean(corpus_embedding, document_embedding)


def get_embeddings(name=""): 
    with open("data/corpus_embedding.txt", "r") as file:
        embeddings = {}
        for line in file:
            split_line = line.split(" ")
            embedding_type = split_line[0]
            embedding = split_line[-DIMENSIONS:]
            embeddings[embedding_type] = np.array(embedding, dtype=np.float64)
    if name != "":
        return embeddings[name]
    else:
        return embeddings


def get_tokens_from_website(url, process_text=True):


    # res = requests.get(url)
    # soup = BeautifulSoup(res.text, "html.parser")
    # text = soup.get_text(separator="\n", strip=True)
    
    extractor = extractors.ArticleExtractor()
    text = extractor.get_content_from_url(url)
    if process_text:
        processed_text = process_documents_combined([text])
        return processed_text
    return text


def test_mean_embedding(glove_embeddings):
    create_mean_embedding(glove_embeddings)
    corpus_embeddings = get_embeddings()

    print("Testing relevant URLs")
    for url in relevant_urls:
        tokens = get_tokens_from_website(url)
        document_embedding = calc_embedding_from_tokens(glove_embeddings, tokens)
        similarity = calc_similarity(corpus_embeddings["combined"], document_embedding)
        print(f"  Similarity between {url} and corpus is: {similarity:.4f}")

    print("Testing irrelevant URLs")
    for url in irrelevant_urls:
        tokens = get_tokens_from_website(url)
        document_embedding = calc_embedding_from_tokens(glove_embeddings, tokens)
        similarity = calc_similarity(corpus_embeddings["combined"], document_embedding)
        print(f"  Similarity between {url} and corpus is: {similarity:.4f}")


def test_multiple_embedding(glove_embeddings):
    create_multiple_embeddings(glove_embeddings)
    embeddings = np.load("data/multiple_embeddings.npy")
    
    print("Testing relevant URLs")
    for url in relevant_urls:
        tokens = get_tokens_from_website(url)
        document_embedding = calc_embedding_from_tokens(glove_embeddings, tokens)
        similarities = []
        for embedding in embeddings:
            similarity = calc_similarity(embedding, document_embedding)
            similarities.append(similarity)
        print("Similarity for url: ", url)
        print(f"    Highest Similarity: {max(similarities):.4f}")
        print(f"    Smallest Similarity: {min(similarities):.4f}")
        print(f"    Average Similarity: {np.mean(similarities):.4f}")
        print("-"*20)

    print("+"*75, "\n")

    similarities = []
    print("Testing irrelevant URLs")
    for url in irrelevant_urls:
        tokens = get_tokens_from_website(url)
        document_embedding = calc_embedding_from_tokens(glove_embeddings, tokens)
        similarities = []
        for embedding in embeddings:
            similarity = calc_similarity(embedding, document_embedding)
            similarities.append(similarity)
        print("Similarity for url: ", url)
        print(f"    Highest Similarity: {max(similarities):.4f}")
        print(f"    Smallest Similarity: {min(similarities):.4f}")
        print(f"    Average Similarity: {np.mean(similarities):.4f}")
        print("-"*20)


def test_tfidf_weighted_mean_embedding(embedding_type="glove"):
    abstracts = process_documents_tfidf(get_abstracts("data/raw_paper_abstracts.json"))
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=0.05)
    corpus_tfidf = tfidf_vectorizer.fit_transform(abstracts)
    print(corpus_tfidf.shape)
    
    if embedding_type == "bert":
        embeddings = get_bert_embeddings()
    elif embedding_type == "glove":
        embeddings = get_embeddings()

    print(embeddings.shape)



def test_taxonomy_embedding(glove_embeddings):
    pass


def test_sentence_embedding(glove_embeddings):
    pass


def test_tfidf_baseline_embedding():
    abstracts = process_documents_tfidf(get_abstracts("data/raw_paper_abstracts.json"))
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=0.05)
    corpus_tfidf = tfidf_vectorizer.fit_transform(abstracts)

    print("Testing relevant URLs")
    for url in relevant_urls:
        text = " ".join(get_tokens_from_website(url))
        document_tfidf = tfidf_vectorizer.transform([text])
        similarity = cosine_similarity(corpus_tfidf, document_tfidf).flatten()
        print(f"  Mean Similarity between  {url.ljust(max_url_length)} and corpus is: {np.mean(similarity):.4f}")
        print(f"  Max Similarity between   {url.ljust(max_url_length)} and corpus is: {np.max(similarity):.4f}")

    print("Testing irrelevant URLs")
    for url in irrelevant_urls:
        text = " ".join(get_tokens_from_website(url))
        document_tfidf = tfidf_vectorizer.transform([text])
        similarity = cosine_similarity(corpus_tfidf, document_tfidf).flatten()
        print(f"  Mean Similarity between  {url.ljust(max_url_length)} and corpus is: {np.mean(similarity):.4f}")
        print(f"  Max Similarity between   {url.ljust(max_url_length)} and corpus is: {np.max(similarity):.4f}")

def get_bert_text_embedding(document):
    inputs = bert_tokenizer(document, return_tensors='pt', truncation=True, max_length=512, padding="max_length")
    outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(1)
    return embeddings

def get_bert_embeddings(embedding_file=None, abstracts=None):

    if not embedding_file:
        embedding_file = "data/bert_embeddings.npy"
    if not os.path.exists(embedding_file):
        if not abstracts:
            abstracts = get_abstracts("data/raw_paper_abstracts.json")
        print(len(abstracts))
        corpus_embedding = torch.stack([get_bert_text_embedding(text) for text in tqdm(abstracts)]).squeeze(1).detach().numpy()
        np.save(embedding_file, corpus_embedding)
    else:
        corpus_embedding = np.load(embedding_file)
    
    return corpus_embedding

def test_bert_embedding(corpus_embedding, distance_function=cosine_similarity):
    print("Testing relevant URLs")
    for url in relevant_urls:
        text = get_tokens_from_website(url, process_text=False)
        text_embedding = get_bert_text_embedding(text)
        similarities = distance_function(corpus_embedding, text_embedding.detach().numpy()).flatten()
        sorted_similarities = np.sort(similarities)[::-1]
        print(f"  Mean Similarity between  {url.ljust(max_url_length)} and corpus is: {np.mean(similarities):.4f}")
        print(f"  Max Similarity between   {url.ljust(max_url_length)} and corpus is: {np.max(similarities):.4f}")
        print(f"  min Similarity between   {url.ljust(max_url_length)} and corpus is: {np.min(similarities):.4f}")
        print(f"  Top 5 mean Similarity between   {url.ljust(max_url_length)} and corpus is: {np.mean(sorted_similarities[:5]):.4f}")

        print("--")

    print("Testing irrelevant URLs")
    for url in irrelevant_urls:
        text = get_tokens_from_website(url, process_text=False)
        text_embedding = get_bert_text_embedding(text)
        similarities = distance_function(corpus_embedding, text_embedding.detach().numpy()).flatten()
        sorted_similarities = np.sort(similarities)[::-1]
        print(f"  Mean Similarity between  {url.ljust(max_url_length)} and corpus is: {np.mean(similarities):.4f}")
        print(f"  Max Similarity between   {url.ljust(max_url_length)} and corpus is: {np.max(similarities):.4f}")
        print(f"  min Similarity between   {url.ljust(max_url_length)} and corpus is: {np.min(similarities):.4f}")
        print(f"  Top 5 mean Similarity between   {url.ljust(max_url_length)} and corpus is: {np.mean(sorted_similarities[:5]):.4f}")
        print("--")


def test_bert_embedding_knn(corpus_embedding, num_neighbors=5):
    knn = NearestNeighbors(n_neighbors=num_neighbors, metric='cosine')
    knn.fit(corpus_embedding)


    print("Testing relevant URLs")
    for url in relevant_urls:
        text = get_tokens_from_website(url, process_text=False)
        text_embedding = get_bert_text_embedding(text).detach().numpy()
        distances, indices = knn.kneighbors(text_embedding)
        print(f"  URL: {url}")
        print(f"  Closest Neighbor: Document {indices[0][0]} at distance: {distances[0][0]}")
        print(f"  Mean Distance to Top {num_neighbors} Neighbors:         {distances[0].mean()}")
        # mean_neighbor_embedding = corpus_embedding[indices[0]].mean(axis=0)
        # similarities = cosine_similarity(corpus_embedding, mean_neighbor_embedding)
        # print(f"  Mean Similarity between  of top k neighbors and corpus is: {np.mean(similarities):.4f}")
        # print(f"  Max Similarity between   of top k neighbors and corpus is: {np.max(similarities):.4f}")

    print("#"*25)
    print("Testing irrelevant URLs")
    for url in irrelevant_urls:
        text = get_tokens_from_website(url, process_text=False)
        text_embedding = get_bert_text_embedding(text).detach().numpy()
        distances, indices = knn.kneighbors(text_embedding)
        print(f"  URL: {url}")
        print(f"  Closest Neighbor: Document {indices[0][0]} at distance: {distances[0][0]}")
        print(f"  Mean Distance to Top Neighbors:                        {distances[0].mean()}")
        # mean_neighbor_embedding = corpus_embedding[indices[0]].mean(axis=0)
        # similarities = cosine_similarity(corpus_embedding, mean_neighbor_embedding)
        # print(f"  Mean Similarity between  of top k neighbors and corpus is: {np.mean(similarities):.4f}")
        # print(f"  Max Similarity between   of top k neighbors and corpus is: {np.max(similarities):.4f}")

def test_embedding_knn(glove_embeddings, num_neighbors=5):

    embeddings = np.load("data/multiple_embeddings.npy")

    knn = NearestNeighbors(n_neighbors=num_neighbors, metric='cosine')
    knn.fit(embeddings)

    print("Testing relevant URLs")
    for url in relevant_urls:
        text = get_tokens_from_website(url)
        text_embedding = calc_embedding_from_tokens(glove_embeddings, text)
        distances, indices = knn.kneighbors(text_embedding)
        print(f"  URL: {url}")
        print(f"  Closest Neighbor: Document {indices[0][0]} at distance: {distances[0][0]}")
        print(f"  Mean Distance to Top {num_neighbors} Neighbors:         {distances[0].mean()}")
        # mean_neighbor_embedding = corpus_embedding[indices[0]].mean(axis=0)
        # similarities = cosine_similarity(corpus_embedding, mean_neighbor_embedding)
        # print(f"  Mean Similarity between  of top k neighbors and corpus is: {np.mean(similarities):.4f}")
        # print(f"  Max Similarity between   of top k neighbors and corpus is: {np.max(similarities):.4f}")
        # print(f"  Sum Similarities between of top k neighbors and corpus is: {np.sum(similarities):.4f}")

    print("#"*25)
    print("Testing irrelevant URLs")
    for url in irrelevant_urls:
        text = get_tokens_from_website(url)
        text_embedding = calc_embedding_from_tokens(glove_embeddings, text)
        distances, indices = knn.kneighbors(text_embedding)
        print(f"  URL: {url}")
        print(f"  Closest Neighbor: Document {indices[0][0]} at distance: {distances[0][0]}")
        print(f"  Mean Distance to Top Neighbors:                        {distances[0].mean()}")
        # mean_neighbor_embedding = corpus_embedding[indices[0]].mean(axis=0)
        # similarities = cosine_similarity(corpus_embedding, mean_neighbor_embedding)
        # print(f"  Mean Similarity between  of top k neighbors and corpus is: {np.mean(similarities):.4f}")
        # print(f"  Max Similarity between   of top k neighbors and corpus is: {np.max(similarities):.4f}")
        # print(f"  Sum Similarities between of top k neighbors and corpus is: {np.sum(similarities):.4f}")

def sbert():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    abstracts = get_abstracts("data/raw_paper_abstracts.json", num_abstracts=10)
    spacy_model = spacy.load("en_core_web_sm")

    abstract_embeddings = []
    for abstract in tqdm(abstracts):
        doc = spacy_model(abstract)
        sentences = [s.text for s in doc.sents]
        s_embeddings = model.encode(sentences)
        abstract_embeddings.append(np.mean(s_embeddings))

    print("Testing relevant URLs")
    for url in relevant_urls:
        text = " ".join(get_tokens_from_website(url, process_text=False))

        doc = spacy_model(text)
        sentences = [s.text for s in doc.sents]
        s_embeddings = model.encode(sentences)
        # s_embeddings_mean = np.mean(s_embeddings, axis=0).reshape(1, -1)

        similarities = cosine_similarity(abstract_embeddings, s_embeddings)
        print(f"  Mean Similarity between  of top k neighbors and corpus is: {np.mean(similarities):.4f}")
        print(f"  Max Similarity between   of top k neighbors and corpus is: {np.max(similarities):.4f}")
        # print(f"  Sum Similarities between of top k neighbors and corpus is: {np.sum(similarities):.4f}")

    print("#"*25)
    print("Testing irrelevant URLs")
    for url in irrelevant_urls:
        text = " ".join(get_tokens_from_website(url, process_text=False))

        doc = spacy_model(text)
        sentences = [s.text for s in doc.sents]
        s_embeddings = model.encode(sentences)

        similarities = cosine_similarity(abstract_embeddings, s_embeddings)
        print(f"  Mean Similarity between  of top k neighbors and corpus is: {np.mean(similarities):.4f}")
        print(f"  Max Similarity between   of top k neighbors and corpus is: {np.max(similarities):.4f}")
        # print(f"  Sum Similarities between of top k neighbors and corpus is: {np.sum(similarities):.4f}")

def analyze_reference_corpus(embedding_file="", title=""):
    if not embedding_file:
        embeddings = np.load("data/bert_embeddings.npy")
    else:
        embeddings= np.load(embedding_file)

    similarity_matrix = cosine_similarity(embeddings)
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap='viridis')
    plt.title('Heatmap of Cosine Similarity between Abstracts')
    plt.xlabel('Abstract Index')
    plt.ylabel('Abstract Index')
    plt.savefig(f'figures/abstract_similarities{title}.png', dpi=300)


    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300, random_state=42)

    tsne_results = tsne.fit_transform(embeddings)

    plt.figure(figsize=(16,10))
    plt.scatter(tsne_results[:,0], tsne_results[:,1])
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Visualization of BERT Embeddings')
    plt.savefig(f'figures/tsne_bert_embeddings{title}.png', dpi=300)


    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(embeddings)
    print(len(embedding), len(embeddings))
    outlier_indices = np.where((embedding[:,0] < -10) & (embedding[:,1] < 3))[0]
    print(outlier_indices)
    abstracts = get_abstracts("data/raw_paper_abstracts.json")
    with open("data/outlier_abstracts.txt", "w+") as file:
        for index in outlier_indices:
            file.write(abstracts[index].strip().replace("\n", "").replace("\r", "").replace("\t", "") + "\n")

    plt.figure(figsize=(15,10))
    plt.scatter(embedding[:,0], embedding[:,1])
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title('UMAP Visualization of BERT Embeddings')

    plt.savefig(f'figures/umap_bert_embeddings{title}.png', dpi=300)


    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=500, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)

    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tsne_results[:,0], tsne_results[:,1], tsne_results[:,2])
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')
    ax.set_title('3D t-SNE Visualization of BERT Embeddings')
    plt.savefig(f'figures/3d_tsne_bert_embeddings{title}.png', dpi=300)


    reducer = umap.UMAP(n_components=3, transform_seed=42)
    embedding = reducer.fit_transform(embeddings)

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2])
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.set_zlabel('UMAP Dimension 3')
    ax.set_title('3D UMAP Visualization of BERT Embeddings')
    plt.savefig(f'figures/3d_umap_bert_embeddings{title}.png', dpi=300)


def filter_abstracts(return_abstracts=False):
    abstracts = get_abstracts("data/raw_paper_abstracts.json")
    embeddings = np.load("data/multiple_embeddings.npy")
    similarity_matrix = cosine_similarity(embeddings)

    similarity_threshold = 0.5
    row_sim = np.mean(similarity_matrix, axis=0)
    indices_to_keep = np.where(row_sim > similarity_threshold)[0]

    filtered_abstracts = [abstracts[i] for i in indices_to_keep]
    if return_abstracts:
        return filtered_abstracts
    get_bert_embeddings("data/filtered_bert_embeddings.npy", filtered_abstracts)



def test_bert_embedding2(corpus_embedding, distance_function=cosine_similarity):


    print("Testing relevant URLs")
    for url in relevant_urls:
        text = get_tokens_from_website(url, process_text=False)
        text_embedding = get_bert_text_embedding(text)
        similarity = distance_function(corpus_embedding, text_embedding.detach().numpy()).flatten()
        print(f"  Mean Similarity between  {url.ljust(max_url_length)} and corpus is: {np.mean(similarity):.4f}")
        print(f"  Max Similarity between   {url.ljust(max_url_length)} and corpus is: {np.max(similarity):.4f}")
        print("--")

    print("Testing irrelevant URLs")
    for url in irrelevant_urls:
        text = get_tokens_from_website(url, process_text=False)
        text_embedding = get_bert_text_embedding(text)
        similarity = distance_function(corpus_embedding, text_embedding.detach().numpy()).flatten()
        print(f"  Mean Similarity between  {url.ljust(max_url_length)} and corpus is: {np.mean(similarity):.4f}")
        print(f"  Max Similarity between   {url.ljust(max_url_length)} and corpus is: {np.max(similarity):.4f}")
        print("--")


def main():
    # TFIDF baseline
    #print("Test TFIDF baseline: ")
    #test_tfidf_baseline_embedding()
    #print("#"*100)
    filter_abstracts()
    get_bert_embeddings("data/bert_embeddings.npy")
    
    analyze_reference_corpus("data/bert_embeddings.npy", "")
    analyze_reference_corpus("data/filtered_bert_embeddings.npy", "_filtered")

    # sbert()

    # bert based embeddings
    print("Test BERT-based embeddings: ")
    test_bert_embedding(get_bert_embeddings("data/filtered_bert_embeddings.npy"), cosine_similarity)
    print("#"*100)

    # bert based embeddings, KNN 
    print("Test BERT-based embeddings with KNN: ")
    test_bert_embedding_knn(get_bert_embeddings("data/filtered_bert_embeddings.npy"), num_neighbors=3)
    print("#"*100)

    quit()


    glove_embeddings = load_glove_model("data/glove.42B.300d/glove.42B.300d.txt")

    # glove embedding, KNN
    # print("Test GloVe embedding, KNN")
    # test_embedding_knn(glove_embeddings)
    # print("#"*100)

    # mean over all embeddings
    print("Test mean embedding: ")
    test_mean_embedding(glove_embeddings)
    print("#"*100)


    # one embedding for each abstract
    print("Test one embedding for each abstract")
    test_multiple_embedding(glove_embeddings)
    print("#"*100)


    # TFIDF weighted mean embeddings
    # https://www.sciencedirect.com/science/article/pii/S0957417421012264
    print("Test TFIDF weighted mean embeddings")
    test_tfidf_weighted_mean_embedding(glove_embeddings)
    print("#"*100)

    
    # one embedding for each level 1 taxonomy concept (atmosphere, disasters, land, built environment, marine, security)
    # print("Test taxonomy embedding")
    # test_taxonomy_embedding(glove_embeddings)
    # print("#"*100)


    # 



if __name__ == "__main__":
    main()



