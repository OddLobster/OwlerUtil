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
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from boilerpy3 import extractors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle




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
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()


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

def test_bert_embedding(corpus_embedding, distance_function=cosine_similarity, plot=True):
    irrelevant_embeddings = []
    relevant_embeddings = []
    print("Testing relevant URLs")
    for url in relevant_urls:
        text = get_tokens_from_website(url, process_text=False, relevant=True, save=True)
        text_embedding = get_bert_text_embedding(text)
        relevant_embeddings.append(text_embedding.detach().numpy())
        similarities = distance_function(corpus_embedding, text_embedding.detach().numpy()).flatten()
        sorted_similarities = np.sort(similarities)[::-1]
        print(f"  Mean Similarity between  {url.ljust(max_url_length)} and corpus is: Mean {np.mean(similarities):.4f}")
        print(f"  Max Similarity between   {url.ljust(max_url_length)} and corpus is: Max  {np.max(similarities):.4f}")
        print(f"  min Similarity between   {url.ljust(max_url_length)} and corpus is: Min  {np.min(similarities):.4f}")
        print(f"  Top 5 mean between       {url.ljust(max_url_length)} and corpus is: TopK {np.mean(sorted_similarities[::-1][:5]):.4f}")
        print("--")
    print("#"*25)

    print("Testing irrelevant URLs")
    for url in irrelevant_urls:
        text = get_tokens_from_website(url, process_text=False, relevant=False, save=True)
        text_embedding = get_bert_text_embedding(text)
        irrelevant_embeddings.append(text_embedding.detach().numpy())
        similarities = distance_function(corpus_embedding, text_embedding.detach().numpy()).flatten()
        sorted_similarities = np.sort(similarities)[::-1]
        print(f"  Mean Similarity between  {url.ljust(max_url_length)} and corpus is: Mean {np.mean(similarities):.4f}")
        print(f"  Max Similarity between   {url.ljust(max_url_length)} and corpus is: Max  {np.max(similarities):.4f}")
        print(f"  min Similarity between   {url.ljust(max_url_length)} and corpus is: Min  {np.min(similarities):.4f}")
        print(f"  Top 5 mean between       {url.ljust(max_url_length)} and corpus is: TopK {np.mean(sorted_similarities[::-1][:5]):.4f}")
        print("--")

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
    for url in relevant_urls:
        text = get_tokens_from_website(url, process_text=False)
        text_embedding = get_bert_text_embedding(text).detach().numpy()
        distances, indices = knn.kneighbors(text_embedding, n_neighbors=num_neighbors)
        print(f"  URL: {url}")
        print(f"  Closest Neighbor: Document {indices[0][0]} at distance: {distances[0][0]:.4f}")
        print(f"  Mean Distance to Top {num_neighbors} Neighbors:         {distances[0].mean():.4f}")
        # mean_neighbor_embedding = corpus_embedding[indices[0]].mean(axis=0)
        # similarities = cosine_similarity(corpus_embedding, mean_neighbor_embedding)
        # print(f"  Mean Similarity between  of top k neighbors and corpus is: {np.mean(similarities):.4f}")
        # print(f"  Max Similarity between   of top k neighbors and corpus is: {np.max(similarities):.4f}")
        if plot:
            plt.scatter(text_embedding[0, 0], text_embedding[0, 1], color='blue', label='Relevant')
            plt.scatter(corpus_embedding[indices[0]][:, 0], corpus_embedding[indices[0]][:, 1], color='blue', alpha=0.5)
            
            # circle = Circle((text_embedding[0, 0], text_embedding[0, 1]), np.mean(distances[0]), fill=False, color='blue', linestyle='--')
            # plt.gca().add_patch(circle)

    print("#"*25)
    print("Testing irrelevant URLs")
    for url in irrelevant_urls:
        text = get_tokens_from_website(url, process_text=False)
        text_embedding = get_bert_text_embedding(text).detach().numpy()
        distances, indices = knn.kneighbors(text_embedding, n_neighbors=num_neighbors)
        print(f"  URL: {url}")
        print(f"  Closest Neighbor: Abstract {indices[0][0]} at distance: {distances[0][0]:.4f}")
        print(f"  Mean Distance to Top {num_neighbors} Neighbors:         {distances[0].mean():.4f}")
        # mean_neighbor_embedding = corpus_embedding[indices[0]].mean(axis=0)
        # similarities = cosine_similarity(corpus_embedding, mean_neighbor_embedding)
        # print(f"  Mean Similarity between  of top k neighbors and corpus is: {np.mean(similarities):.4f}")
        # print(f"  Max Similarity between   of top k neighbors and corpus is: {np.max(similarities):.4f}")
        if plot:
            plt.scatter(text_embedding[0, 0], text_embedding[0, 1], color='red', label='Irrelevant')
            plt.scatter(corpus_embedding[indices[0]][:, 0], corpus_embedding[indices[0]][:, 1], color='red', alpha=0.5)
            
            # Draw circle around the irrelevant URL and its neighbors
            # circle = Circle((text_embedding[0, 0], text_embedding[0, 1]), np.mean(distances[0]), fill=False, color='red', linestyle='--')
            # plt.gca().add_patch(circle)

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


def main():
    # Abstract preprocessing / Visualization
    # filter_abstracts()
    # get_bert_embeddings("data/bert_embeddings.npy")
    # analyze_reference_corpus("data/bert_embeddings.npy", "")
    # analyze_reference_corpus("data/filtered_bert_embeddings.npy", "_filtered")


    # TFIDF baseline
    # print("Test TFIDF baseline: ")
    # test_tfidf_baseline_embedding()
    # print("#"*100)
    # bert based embeddings
    print("Test BERT-based embeddings: ")
    test_bert_embedding(get_bert_embeddings("data/filtered_bert_embeddings.npy"), euclidean_distances)
    print("#"*100)
    # bert based embeddings, KNN 
    print("Test BERT-based embeddings with KNN: ")
    test_bert_embedding_knn(get_bert_embeddings("data/filtered_bert_embeddings.npy"), num_neighbors=3)
    print("#"*100)


    # bert classification
    test_bert_classification()

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



