import numpy as np
import json
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

DIMENSIONS = 300

def load_glove_model(glove_file):
    embeddings = {}

    with open(glove_file, 'r', encoding="utf8") as file:
        for line in tqdm(file):
            split_line = line.split(" ")
            embedding = np.array(split_line[-DIMENSIONS:], dtype=np.float64)
            word = split_line[0]
            embeddings[word.lower()] = embedding

    return embeddings


def calc_combined_embedding(glove_embedding, documents):
    combined_embedding = np.zeros(DIMENSIONS, dtype=np.float64)
    failed_tokens = set()
    
    num_tokens = 0
    for token in documents:
        try:
            combined_embedding += glove_embedding[token]
            num_tokens += 1
        except:
            failed_tokens.add(token)
    print(combined_embedding)
    with open("data/unknown_tokens.txt", "w+") as file:
        for token in failed_tokens:
                file.write(token + "\n")

    return combined_embedding/num_tokens


def get_abstracts(filename):
    abstracts = []
    with open(filename, "r") as file:
        data = json.load(file)

    for paper in data:
        try:
            abstracts.append(paper["abstract"])
        except:
            pass

    return abstracts


def process_documents(documents, apply_stemming=False):
    processed_documents = []
    tokenized_documents = [word_tokenize(text) for text in documents]

    if apply_stemming:
        stemmer = PorterStemmer()
        for document in tokenized_documents:
            for token in document:
                processed_documents.append(stemmer.stem(token))
    else:
        processed_documents = [word.lower() for doc in tokenized_documents for word in doc]

    return processed_documents


def create_combined_embedding():
    print("Loading GloVe embeddings")
    glove_embeddings = load_glove_model('data/glove.42B.300d/glove.42B.300d.txt')
    print("Loading documents")
    abstracts = get_abstracts("data/raw_paper_abstracts.json")
    print("Processing documents")
    processed_abstracts = process_documents(abstracts)
    print("Creating document embedding")
    combined_embedding = calc_combined_embedding(glove_embeddings, processed_abstracts)
    print(combined_embedding)
    with open("data/embeddings.txt", "w+") as file:
        file.write("combined ")
        combined_embedding.tofile(file, sep=" ")



create_combined_embedding()
