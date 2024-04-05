import numpy as np
import requests
import json
import torch 
import os

from numpy.linalg import norm
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from nltk.corpus import stopwords
from string import punctuation
from scipy.spatial import distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel

DIMENSIONS = 300
GLOVE_VOCABULARY = set()

COSINE = 1
EUCLIDEAN = 2

relevant_urls = ["https://en.wikipedia.org/wiki/Atmospheric_science", "https://en.wikipedia.org/wiki/Geomatics", "https://en.wikipedia.org/wiki/Remote_sensing", "https://en.wikipedia.org/wiki/Earth_science", "https://en.wikipedia.org/wiki/Flood" ] # "https://geo.arc.nasa.gov/"
irrelevant_urls = ["https://en.wikipedia.org/wiki/Music", "https://en.wikipedia.org/wiki/Almond", "https://en.wikipedia.org/wiki/Train", "https://en.wikipedia.org/wiki/Moth", "https://en.wikipedia.org/wiki/England"] #  "https://lasermania.com/", "https://carolinescakes.com/"
all_urls = relevant_urls + irrelevant_urls
max_url_length = max(len(url) for url in all_urls)  

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


def get_abstracts(filename):
    abstracts = []
    with open(filename, "r") as file:
        data = json.load(file)

    for i, paper in enumerate(data):
        try:
            abstracts.append(paper["abstract"])
        except:
            pass
        # if i > 5:
        #     break

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
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    text = soup.get_text(strip=True)
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


def test_tfidf_weighted_mean_embedding(glove_embeddings):
    pass


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
        # print(f"  Mean Similarity between {url.ljust(max_url_length)} and corpus is: {np.mean(similarity):.4f}")
        print(f"  Max Similarity between {url.ljust(max_url_length)} and corpus is: {np.max(similarity):.4f}")
        # print(f"  Sum Similarities between {url.ljust(max_url_length)} and corpus is: {np.sum(similarity):.4f}")

    print("Testing irrelevant URLs")
    for url in irrelevant_urls:
        text = " ".join(get_tokens_from_website(url))
        document_tfidf = tfidf_vectorizer.transform([text])
        similarity = cosine_similarity(corpus_tfidf, document_tfidf).flatten()
        # print(f"  Mean Similarity between {url.ljust(max_url_length)} and corpus is: {np.mean(similarity):.4f}")
        print(f"  Max Similarity between {url.ljust(max_url_length)} and corpus is: {np.max(similarity):.4f}")
        # print(f"  Sum Similarities between {url.ljust(max_url_length)} and corpus is: {np.sum(similarity):.4f}")

def test_bert_embedding():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    def get_bert_embedding(document):
        inputs = tokenizer(document, return_tensors='pt', truncation=True, max_length=512)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings
    abstracts = get_abstracts("data/raw_paper_abstracts.json")

    embedding_file = "data/bert_embeddings.npy"
    if not os.path.exists(embedding_file):
        corpus_embedding = torch.stack([get_bert_embedding(text) for text in abstracts]).squeeze(1).detach().numpy()
        np.save(embedding_file)
    else:
        corpus_embedding = np.load(embedding_file)

    print("Testing relevant URLs")
    for url in relevant_urls:
        text = get_tokens_from_website(url, process_text=False)
        text_embedding = get_bert_embedding(text)
        similarity = cosine_similarity(corpus_embedding, text_embedding.detach().numpy()).flatten()
        print(f"  Mean Similarity between  {url.ljust(max_url_length)} and corpus is: {np.mean(similarity):.4f}")
        print(f"  Max Similarity between   {url.ljust(max_url_length)} and corpus is: {np.max(similarity):.4f}")
        print(f"  Sum Similarities between {url.ljust(max_url_length)} and corpus is: {np.sum(similarity):.4f}")

    print("Testing irrelevant URLs")
    for url in irrelevant_urls:
        text = get_tokens_from_website(url, process_text=False)
        text_embedding = get_bert_embedding(text)
        similarity = cosine_similarity(corpus_embedding, text_embedding.detach().numpy()).flatten()
        print(f"  Mean Similarity between  {url.ljust(max_url_length)} and corpus is: {np.mean(similarity):.4f}")
        print(f"  Max Similarity between   {url.ljust(max_url_length)} and corpus is: {np.max(similarity):.4f}")
        print(f"  Sum Similarities between {url.ljust(max_url_length)} and corpus is: {np.sum(similarity):.4f}")


    


def main():
    # glove_embeddings = load_glove_model("data/glove.42B.300d/glove.42B.300d.txt")
    # test_embedding = calc_embedding_from_tokens(glove_embeddings, process_documents_combined(get_abstracts("data/raw_paper_abstracts.json")))
    # print("Should be one: ", calc_similarity(test_embedding, test_embedding))



    # TFIDF baseline
    print("Test TFIDF baseline: ")
    #test_tfidf_baseline_embedding()
    print("#"*100)


    # bert based embeddings
    print("Test BERT-based embeddings: ")
    test_bert_embedding()
    print("#"*100)


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
    print("Test taxonomy embedding")
    test_taxonomy_embedding(glove_embeddings)
    print("#"*100)


    # 



if __name__ == "__main__":
    main()



