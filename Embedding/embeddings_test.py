import requests
import json
import torch 
import os
import spacy 
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
import seaborn as sns
import time
import random
import hashlib
import h5py

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
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from boilerpy3 import extractors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn import linear_model
from sklearn.ensemble import IsolationForest


DIMENSIONS = 300
GLOVE_VOCABULARY = set()

COSINE = 1
EUCLIDEAN = 2


def get_urls(relevant=True):
    if relevant:
        with open("relevant_urls.txt", "r") as file:
            url_list = [line.strip() for line in file]
    else:
        with open("irrelevant_urls.txt", "r") as file:
            url_list = [line.strip() for line in file]
    return url_list


relevant_urls = ["https://en.wikipedia.org/wiki/Atmospheric_science", "https://en.wikipedia.org/wiki/Earth_observation", "https://en.wikipedia.org/wiki/Remote_sensing", "https://en.wikipedia.org/wiki/Earth_science", "https://en.wikipedia.org/wiki/Flood" ] # "https://geo.arc.nasa.gov/"
irrelevant_urls = ["https://en.wikipedia.org/wiki/Music", "https://en.wikipedia.org/wiki/Almond", "https://en.wikipedia.org/wiki/Train", "https://en.wikipedia.org/wiki/Moth", "https://en.wikipedia.org/wiki/England"] #  "https://lasermania.com/", "https://carolinescakes.com/"

relevant_urls = get_urls(relevant=True)
irrelevant_urls = get_urls(relevant=False)

all_urls = relevant_urls + irrelevant_urls
max_url_length = max(len(url) for url in all_urls)  
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=0.05)

def get_tfidf_of_corpus():
    abstracts = process_documents_tfidf(filter_abstracts(return_abstracts=True))
    corpus_tfidf = tfidf_vectorizer.fit_transform(abstracts)
    return corpus_tfidf

def get_tokens_from_website(url, process_text=True, relevant=True, save=True):
    # res = requests.get(url)
    # soup = BeautifulSoup(res.text, "html.parser")
    # text = soup.get_text(separator="\n", strip=True)
    text = ""
    try:
        extractor = extractors.ArticleExtractor()
        text = extractor.get_content_from_url(url)
        if process_text:
            processed_text = process_documents_combined([text])
            return processed_text
    except:
        pass
    
    if save:
        filename = "relevant_text.txt" if relevant else "irrelevant_text.txt"
        with open(filename, "a+") as file:
            file.write(text.strip().replace("\n", " ").replace("\r", " ").replace("\t", " ") + "\n")

    return text


def get_texts(relevant=True):
    filename = "relevant_text.txt" if relevant else "irrelevant_text.txt"
    with open(filename, "r") as file:
        texts = [line.strip() for line in file]
    return texts


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
    abstracts = process_documents_tfidf(filter_abstracts(return_abstracts=True))
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=0.05)
    corpus_tfidf = tfidf_vectorizer.fit_transform(abstracts)

    print("Testing relevant URLs")
    for url in relevant_urls:
        text = " ".join(get_tokens_from_website(url))
        document_tfidf = tfidf_vectorizer.transform([text])
        similarity = cosine_similarity(corpus_tfidf, document_tfidf).flatten()
        # print(f"  Mean Similarity between  {url.ljust(max_url_length)} and corpus is: {np.mean(similarity):.4f}")
        print(f"  Max Similarity between   {url.ljust(max_url_length)} and corpus is: {np.max(similarity):.4f}")

    print("#"*25)
    print("Testing irrelevant URLs")
    for url in irrelevant_urls:
        text = " ".join(get_tokens_from_website(url))
        document_tfidf = tfidf_vectorizer.transform([text])
        similarity = cosine_similarity(corpus_tfidf, document_tfidf).flatten()
        # print(f"  Mean Similarity between  {url.ljust(max_url_length)} and corpus is: {np.mean(similarity):.4f}")
        print(f"  Max Similarity between   {url.ljust(max_url_length)} and corpus is: {np.max(similarity):.4f}")

# def get_bert_text_embedding(document):
#     inputs = bert_tokenizer(document, return_tensors='pt', truncation=True, max_length=512, padding="max_length")
#     outputs = bert_model(**inputs)
#     embeddings = outputs.last_hidden_state.mean(1)
#     return embeddings
        
def get_bert_text_embedding(document, cache_file="data/bert_embeddings_cache_large.json"):
    doc_hash = hashlib.sha256(document.encode('utf-8')).hexdigest()
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}

    if doc_hash in cache:
        cached_embedding = torch.tensor(cache[doc_hash])
        return cached_embedding
    else:
        inputs = bert_tokenizer(document, return_tensors='pt', truncation=True, max_length=512, padding="max_length")
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(1)
        
        with open(cache_file, 'w') as f:
            cache[doc_hash] = embeddings.detach().numpy().tolist()
            json.dump(cache, f)
        
        return embeddings.detach()
    

def get_bert_embeddings(embedding_file=None, abstracts=None):
    if ".hdf5" in embedding_file:
        with h5py.File(embedding_file, 'r') as hdf5_file:
            embeddings = []
            for key in hdf5_file.keys():
                embeddings.append(np.array(hdf5_file[key]))
            embeddings = np.array(embeddings).squeeze(1)
            print(embeddings.shape)
        return embeddings
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
        print(corpus_embedding.shape)
    
    return corpus_embedding

def test_bert_embedding(corpus_embedding, distance_function=cosine_similarity, plot=True):
    irrelevant_embeddings = []
    relevant_embeddings = []
    relevant_sims = []
    irrelevant_sims = []
    relevant_texts = get_texts(relevant=True)[:10]



    print("Testing relevant URLs")
    for i, text in enumerate(relevant_texts):
        text_embedding = get_bert_text_embedding(text)
        relevant_embeddings.append(text_embedding.detach().numpy())
        similarities = distance_function(corpus_embedding, text_embedding.detach().numpy()).flatten()
        sorted_similarities = np.sort(similarities)[::-1]
        # print(f"  Mean Similarity between  {relevant_urls[i].ljust(max_url_length)} and corpus is: Mean {np.mean(similarities):.4f}")
        # print(f"  Max Similarity between   {relevant_urls[i].ljust(max_url_length)} and corpus is: Max  {np.max(similarities):.4f}")
        # print(f"  min Similarity between   {relevant_urls[i].ljust(max_url_length)} and corpus is: Min  {np.min(similarities):.4f}")
        print(f"  Top 5 mean between       {relevant_urls[i].ljust(max_url_length)} and corpus is: TopK {np.mean(sorted_similarities[::-1][:5]):.4f}")
        relevant_sims.append(np.mean(sorted_similarities[::-1][:5]))
    print("#"*25)

    print("Testing irrelevant URLs")
    irrelevant_texts = get_texts(relevant=False)[:10]
    for i, text in enumerate(irrelevant_texts):
        text_embedding = get_bert_text_embedding(text)
        irrelevant_embeddings.append(text_embedding.detach().numpy())
        similarities = distance_function(corpus_embedding, text_embedding.detach().numpy()).flatten()
        sorted_similarities = np.sort(similarities)[::-1]
        # print(f"  Mean Similarity between  {irrelevant_urls[i].ljust(max_url_length)} and corpus is: Mean {np.mean(similarities):.4f}")
        # print(f"  Max Similarity between   {irrelevant_urls[i].ljust(max_url_length)} and corpus is: Max  {np.max(similarities):.4f}")
        # print(f"  min Similarity between   {irrelevant_urls[i].ljust(max_url_length)} and corpus is: Min  {np.min(similarities):.4f}")
        print(f"  Top 5 mean between       {irrelevant_urls[i].ljust(max_url_length)} and corpus is: TopK {np.mean(sorted_similarities[::-1][:5]):.4f}")
        irrelevant_sims.append(np.mean(sorted_similarities[::-1][:5]))

    print("Mean total relevant sims: ", np.array(relevant_sims).mean())
    print("Mean total irrelevant sims: ", np.array(irrelevant_sims).mean())
    print("Median total relevant sims: ", np.median(relevant_sims))
    print("Median total irrelevant sims: ", np.median(irrelevant_sims))


    if plot:
        reducer = umap.UMAP(random_state=42)
        umap_data = reducer.fit_transform(corpus_embedding)
        plt.figure(figsize=(15,10))
        plt.scatter(umap_data[:,0], umap_data[:,1], label="Corpus")
        for embedding in relevant_embeddings:
            data = reducer.transform(embedding)
            plt.scatter(data[:,0], data[:,1], color="green", label="Relevant")
        for embedding in irrelevant_embeddings:
            data = reducer.transform(embedding)
            plt.scatter(data[:,0], data[:,1], color="red", label="Irrelevant")
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.title('UMAP Visualization of BERT Embeddings')
        plt.legend()
        plt.savefig(f'figures/umap_bert_embeddings_total.png', dpi=300) 


def test_bert_embedding_knn(corpus_embedding, num_neighbors=5, plot=True):
    knn = NearestNeighbors(n_neighbors=num_neighbors)
    knn.fit(corpus_embedding)


    if plot:
        plt.figure(figsize=(10, 8))
        plt.scatter(corpus_embedding[:, 0], corpus_embedding[:, 1], color='gray', label='Corpus')


    print("Testing relevant URLs")
    relevant_texts = get_texts(relevant=True)[:10]
    for i, text in enumerate(relevant_texts):        
        text_embedding = get_bert_text_embedding(text).detach().numpy()
        distances, indices = knn.kneighbors(text_embedding, n_neighbors=num_neighbors)
        print(f"  URL: {relevant_urls[i]}")
        print(f"  Closest Neighbor: Document {indices[0][0]} at distance: {distances[0][0]:.4f}")
        print(f"  Mean Distance to Top {num_neighbors} Neighbors:         {distances[0].mean():.4f}")
        # mean_neighbor_embedding = corpus_embedding[indices[0]].mean(axis=0)
        # similarities = cosine_similarity(corpus_embedding, mean_neighbor_embedding)
        # print(f"  Mean Similarity between  of top k neighbors and corpus is: {np.mean(similarities):.4f}")
        # print(f"  Max Similarity between   of top k neighbors and corpus is: {np.max(similarities):.4f}")
        if plot:
            plt.scatter(text_embedding[0, 0], text_embedding[0, 1], color='blue', label='Relevant')
            plt.scatter(corpus_embedding[indices[0]][:, 0], corpus_embedding[indices[0]][:, 1], color='blue', alpha=0.5)
            

    print("#"*25)
    print("Testing irrelevant URLs")
    irrelevant_texts = get_texts(relevant=False)[:10]
    for i, text in enumerate(irrelevant_texts):
        text_embedding = get_bert_text_embedding(text).detach().numpy()
        distances, indices = knn.kneighbors(text_embedding, n_neighbors=num_neighbors)
        print(f"  URL: {irrelevant_urls[i]}")
        print(f"  Closest Neighbor: Abstract {indices[0][0]} at distance: {distances[0][0]:.4f}")
        print(f"  Mean Distance to Top {num_neighbors} Neighbors:         {distances[0].mean():.4f}")
        # mean_neighbor_embedding = corpus_embedding[indices[0]].mean(axis=0)
        # similarities = cosine_similarity(corpus_embedding, mean_neighbor_embedding)
        # print(f"  Mean Similarity between  of top k neighbors and corpus is: {np.mean(similarities):.4f}")
        # print(f"  Max Similarity between   of top k neighbors and corpus is: {np.max(similarities):.4f}")
        if plot:
            plt.scatter(text_embedding[0, 0], text_embedding[0, 1], color='red', label='Irrelevant')
            plt.scatter(corpus_embedding[indices[0]][:, 0], corpus_embedding[indices[0]][:, 1], color='red', alpha=0.5)
            

    if plot:
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('Neighbors of Relevant and Irrelevant URLs with Corpus Embedding')
        plt.legend()
        plt.savefig('figures/bert_embeddings_knn_total.png', dpi=300, bbox_inches='tight')

def test_embedding_knn(glove_embeddings, num_neighbors=5):
    #FIXME
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

def test_bert_classification():
    abstracts = filter_abstracts(return_abstracts=True)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    labels = torch.tensor([1 for _ in range(len(abstracts))])

    inputs = tokenizer(abstracts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    outputs = model(**inputs, labels=labels)
    loss, logits = outputs.loss, outputs.logits

    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 4
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            model.zero_grad()
            
            outputs = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            
            optimizer.step()
        
        print(f"Epoch {epoch} loss: {total_loss / len(train_dataloader)}")


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
    #FIXME use bert embeddings ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    embeddings = np.load("data/multiple_embeddings.npy")
    similarity_matrix = cosine_similarity(embeddings)

    similarity_threshold = 0.5
    row_sim = np.mean(similarity_matrix, axis=0)
    indices_to_keep = np.where(row_sim > similarity_threshold)[0]
    filtered_abstracts = [abstracts[i] for i in indices_to_keep]
    if return_abstracts:
        return filtered_abstracts
    get_bert_embeddings("data/filtered_bert_embeddings.npy", filtered_abstracts)

def calc_similarities_glove(glove_embedding, corpus_embeddings, text):
    text_embedding = calc_embedding_from_tokens(glove_embedding, process_documents_combined([text]))
    similarities = []
    for embedding in corpus_embeddings:
        similarity = calc_similarity(embedding, list(text_embedding), mode=COSINE)
        similarities.append(similarity)
    sorted_similarities = np.sort(similarities)[::-1]
    return sorted_similarities

def calc_similarities_bert(glove_embedding, corpus_embedding, text):
    text_embedding = get_bert_text_embedding(text)
    similarities = euclidean_distances(corpus_embedding, text_embedding.detach().numpy()).flatten()
    sorted_similarities = np.sort(similarities)
    return sorted_similarities

def calc_similarities_tfidf(glove_embedding, corpus_tfidf, text):
    document_tfidf = tfidf_vectorizer.transform([text])
    similarities = cosine_similarity(corpus_tfidf, document_tfidf).flatten()
    sorted_similarities = np.sort(similarities)[::-1]
    return sorted_similarities

def test_precision_threshold(corpus_embedding, similarity_func=None, glove_embedding=None):
    reference_corpus = filter_abstracts(return_abstracts=True)
    relevant_texts = get_texts(relevant=True)
    irrelevant_texts = get_texts(relevant=False)
    num_relevant = len(relevant_texts)
    num_irrelevant = len(irrelevant_texts)

    document_urls = relevant_urls + irrelevant_urls
    documents = relevant_texts + irrelevant_texts
    combined = list(zip(documents, document_urls))
    random.shuffle(combined)

    documents, document_urls = zip(*combined)

    relevant_threshold_set = random.sample(reference_corpus, int(len(reference_corpus)*0.2))
    sims_5 = []
    sims_25 = []
    sims_high = []
    sims_low = []
    sims = []

    top_5 = int(len(reference_corpus)*0.05)
    top_25 = int(len(reference_corpus)*0.25)

    for i, text in tqdm(enumerate(relevant_threshold_set)):
        sorted_similarities = similarity_func(glove_embedding, corpus_embedding, text)
        sims_5.append(np.mean(sorted_similarities[:top_5]))
        sims_25.append(np.mean(sorted_similarities[:top_25]))
        sims_high.append(sorted_similarities[-1])
        sims_low.append(sorted_similarities[0])
        sims.extend(sorted_similarities)

    relevant_threshold_mean_5 = np.array(sims_5).mean()
    relevant_threshold_mean_25 = np.array(sims_25).mean()
    relevant_threshold_median_5 = np.median(sims_5)
    relevant_threshold_median_25 = np.median(sims_25)
    relevant_threshold_total_mean = np.array(sims).mean()
    relevant_threshold_total_median = np.median(sims)


    performance_stats = {
        'mean_5': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'threshold':relevant_threshold_mean_5},
        'mean_25': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'threshold':relevant_threshold_mean_25},
        'median_5': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'threshold':relevant_threshold_median_5},
        'median_25': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'threshold':relevant_threshold_median_25},
        'mean_total': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'threshold':relevant_threshold_total_mean},
        'median_total': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'threshold':relevant_threshold_total_median}
    }
    total_relevant_docs_identified = 0
    for i, text in tqdm(enumerate(documents)):
        sorted_similarities = similarity_func(glove_embedding, corpus_embedding, text)
        relevance_prediction = np.mean(sorted_similarities[:5])

        ground_truth_text_relevance = text in relevant_texts
        # print(f"Relevance of {document_urls[i].ljust(max_url_length)} is: {relevance_prediction:.3f}")
        for key, value in performance_stats.items():
            is_relevant = relevance_prediction <= value["threshold"]
            if is_relevant and ground_truth_text_relevance:
                performance_stats[key]['tp'] += 1
            elif is_relevant and not ground_truth_text_relevance:
                performance_stats[key]['fp'] += 1
            elif not is_relevant and ground_truth_text_relevance:
                performance_stats[key]['fn'] += 1
            elif not is_relevant and not ground_truth_text_relevance:
                performance_stats[key]['tn'] += 1
        total_relevant_docs_identified += is_relevant


    for key, stats in performance_stats.items():
        precision = (stats['tp'] / (stats['tp'] + stats['fp'])) if stats['tp'] + stats['fp'] > 0 else 0
        recall = (stats['tp'] / (stats['tp'] + stats['fn'])) if stats['tp'] + stats['fn'] > 0 else 0
        print(f"TP: {stats['tp']} - FP: {stats['fp']} - FN: {stats['fn']} - TN: {stats['tn']}")
        print(f"{key} - Precision: {precision:.3f}, Recall: {recall:.3f}")
    print(f"Total relevant documents found: {total_relevant_docs_identified} out of {num_relevant}. (False Positives included)")

    
def update_stats(prediction, stats, ground_truth):
    is_relevant = False if prediction == -1 else True
    if is_relevant and ground_truth:
        stats['tp'] += 1
    elif is_relevant and not ground_truth:
        stats['fp'] += 1
    elif not is_relevant and ground_truth:
        stats['fn'] += 1
    elif not is_relevant and not ground_truth:
        stats['tn'] += 1


def print_stats(stats, title=""):
    precision = (stats['tp'] / (stats['tp'] + stats['fp'])) if stats['tp'] + stats['fp'] > 0 else 0
    recall = (stats['tp'] / (stats['tp'] + stats['fn'])) if stats['tp'] + stats['fn'] > 0 else 0
    print(f"TP: {stats['tp']} - FP: {stats['fp']} - FN: {stats['fn']} - TN: {stats['tn']}")
    print(f"{title} - Precision: {precision:.3f}, Recall: {recall:.3f}")


def test_precision_novelty_detection(corpus_embedding, embedding_type="", glove_embedding=None):

    svm_model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.01)
    svm_model.fit(corpus_embedding)

    lof_model = LocalOutlierFactor(n_neighbors=4, novelty=True)
    lof_model.fit(corpus_embedding)

    iso_model = IsolationForest(random_state=42)
    iso_model.fit(corpus_embedding)

    sgd_model = linear_model.SGDOneClassSVM(nu=0.05)
    sgd_model.fit(corpus_embedding)

    relevant_texts = get_texts(relevant=True)
    irrelevant_texts = get_texts(relevant=False)
    num_relevant = len(relevant_texts)
    num_irrelevant = len(irrelevant_texts)

    document_urls = relevant_urls + irrelevant_urls
    documents = relevant_texts + irrelevant_texts
    combined = list(zip(documents, document_urls))
    random.shuffle(combined)

    unseen_documents, document_urls = zip(*combined)
    lof_stats = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    svm_stats = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    iso_stats = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    sgd_stats = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

    for document in tqdm(unseen_documents):
        if embedding_type == "tfidf":
            document_embedding  = tfidf_vectorizer.transform([document])
        elif embedding_type == "bert":
            document_embedding = get_bert_text_embedding(document)
        elif embedding_type == "glove":
            document_embedding = np.array(calc_embedding_from_tokens(glove_embedding, process_documents_combined([document]))).reshape(1, -1)
        else:
            print("Specify embedding type")
            exit()
        lof_prediction = lof_model.predict(document_embedding)[0]
        svm_prediction = svm_model.predict(document_embedding)[0]
        iso_prediction = iso_model.predict(document_embedding)[0]
        sgd_prediction = sgd_model.predict(document_embedding)[0]

        ground_truth = document in relevant_texts
        update_stats(lof_prediction, lof_stats, ground_truth)
        update_stats(svm_prediction, svm_stats, ground_truth)
        update_stats(iso_prediction, iso_stats, ground_truth)
        update_stats(sgd_prediction, sgd_stats, ground_truth)

    print_stats(lof_stats, title="lof")
    print_stats(svm_stats, title="svm")
    print_stats(iso_stats, title="iso")
    print_stats(sgd_stats, title="sgd")

def hp_search_lof():
    corpus_embedding = get_bert_embeddings("data/filtered_bert_embeddings.npy")

    relevant_texts = get_texts(relevant=True)
    irrelevant_texts = get_texts(relevant=False)
    document_urls = relevant_urls + irrelevant_urls
    documents = relevant_texts + irrelevant_texts
    combined = list(zip(documents, document_urls))
    random.shuffle(combined)
    unseen_documents, document_urls = zip(*combined)

    n_neighbors_options = [2, 4, 8, 12, 16, 20, 40, 50, 75, 100]
    leaf_size_options = [30, 40, 50, 200]

    for leaf_size in tqdm(leaf_size_options):
        for num_neighbor in tqdm(n_neighbors_options):
            lof_stats = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
            lof_model = LocalOutlierFactor(n_neighbors=num_neighbor, leaf_size=leaf_size, novelty=True)
            lof_model.fit(corpus_embedding)

            for document in tqdm(unseen_documents):
                document_embedding = get_bert_text_embedding(document)
                lof_prediction = lof_model.predict(document_embedding)[0]

                ground_truth = document in relevant_texts
                update_stats(lof_prediction, lof_stats, ground_truth)
            print(f"Hyperparams: neighbors: {num_neighbor} - leaf_size: {leaf_size}")
            print_stats(lof_stats, title="lof")

def hp_search_iso():
    corpus_embedding = get_bert_embeddings("data/filtered_bert_embeddings.npy")

    relevant_texts = get_texts(relevant=True)
    irrelevant_texts = get_texts(relevant=False)
    document_urls = relevant_urls + irrelevant_urls
    documents = relevant_texts + irrelevant_texts
    combined = list(zip(documents, document_urls))
    random.shuffle(combined)
    unseen_documents, document_urls = zip(*combined)

    n_estimators_options = [10, 50, 100, 200, 300, 400, 500, 700, 1000]

    for num_estimators in tqdm(n_estimators_options):
        lof_stats = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

        iso_model = IsolationForest(random_state=42, n_estimators=num_estimators)
        iso_model.fit(corpus_embedding)

        for document in tqdm(unseen_documents):
            document_embedding = get_bert_text_embedding(document)
            lof_prediction = iso_model.predict(document_embedding)[0]

            ground_truth = document in relevant_texts
            update_stats(lof_prediction, lof_stats, ground_truth)
        print(f"Hyperparams: estimators: {num_estimators}")
        print_stats(lof_stats, title="iso")

def random_benchmark():
    relevant_texts = get_texts(relevant=True)
    irrelevant_texts = get_texts(relevant=False)
    document_urls = relevant_urls + irrelevant_urls
    documents = relevant_texts + irrelevant_texts
    combined = list(zip(documents, document_urls))
    random.shuffle(combined)
    unseen_documents, document_urls = zip(*combined)

    stats = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    model_predict = lambda: random.choice([-1, 1])

    for document in tqdm(unseen_documents):
        prediction = model_predict()
        ground_truth = document in relevant_texts
        update_stats(prediction, stats, ground_truth)

    print_stats(stats, title="random")

def main():
    # Abstract preprocessing / Visualization
    # filter_abstracts()
    # get_bert_embeddings("data/bert_embeddings.npy")
    # analyze_reference_corpus("data/bert_embeddings.npy", "")
    # analyze_reference_corpus("data/filtered_bert_embeddings.npy", "_filtered")

    for _ in range(5):
        random_benchmark()

    #glove_embeddings = load_glove_model("data/glove.42B.300d/glove.42B.300d.txt")
    for _ in range(1):
        print("Threshold Selection approach")
        # print("TFIDF Results:")
        # test_precision_threshold(get_tfidf_of_corpus(), similarity_func=calc_similarities_tfidf)
        # print("GloVe Results:")
        # test_precision_threshold(np.load("data/multiple_embeddings.npy"), similarity_func=calc_similarities_glove, glove_embedding=glove_embeddings)
        # print("BERT Results:")
        # test_precision_threshold(get_bert_embeddings("data/corpus_embedding_d_0.hdf5"), similarity_func=calc_similarities_bert)
        
        print("Novelty Detection Approach")
        # print("TFIDF Results:")
        # test_precision_novelty_detection(get_tfidf_of_corpus(), embedding_type="tfidf")
        # print("GloVe Results:")
        # test_precision_novelty_detection(np.load("data/multiple_embeddings.npy"), embedding_type="glove", glove_embedding=glove_embeddings)
        print("BERT Results:")
        test_precision_novelty_detection(get_bert_embeddings("data/corpus_embedding_d_0.hdf5"), embedding_type="bert")
        print("-"*250)
        
    # hp_search_iso()
    # hp_search_lof()

    # TFIDF baseline
    # print("Test TFIDF baseline: ")
    # test_tfidf_baseline_embedding()
    # print("#"*100)
    # bert based embeddings
    print("Test BERT-based embeddings: ")
    test_bert_embedding(get_bert_embeddings("data/corpus_embedding_d_0.hdf5"), euclidean_distances)
    print("#"*100)
    # bert based embeddings, KNN 
    # print("Test BERT-based embeddings with KNN: ")
    # test_bert_embedding_knn(get_bert_embeddings("data/filtered_bert_embeddings.npy"), num_neighbors=3)
    # print("#"*100)

    exit()

    # bert classification
    test_bert_classification()



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



if __name__ == "__main__":
    main()



