import numpy as np
import json
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from nltk.corpus import stopwords
from string import punctuation

DIMENSIONS = 300

def load_glove_model(glove_file):
    embeddings = {}

    with open(glove_file, 'r', encoding="utf8") as file:
        for i, line in tqdm(enumerate(file)):
            split_line = line.split(" ")
            embedding = np.array(split_line[-DIMENSIONS:], dtype=np.float64)
            word = split_line[0]
            embeddings[word.lower()] = embedding

    return embeddings


def calc_embedding_from_tokens(glove_embedding, documents):
    combined_embedding = np.zeros(DIMENSIONS, dtype=np.float64)
    failed_tokens = set()
    
    num_tokens = 0
    for token in documents:
        try:
            combined_embedding += glove_embedding[token]
            num_tokens += 1
        except:
            # print(f"Token {token} doesnt exist in GloVe 42B uncased")
            failed_tokens.add(token)
    with open("data/unknown_tokens.txt", "w+") as file:
        for token in failed_tokens:
                file.write(token + "\n")

    return np.array(combined_embedding/num_tokens)


def calc_multiple_embeddings_from_tokens(glove_embeddings, documents):
    embeddings = []
    for doc in documents:
        embedding = np.zeros(DIMENSIONS, dtype=np.float64)
        for token in doc:
            try:
                embedding += glove_embeddings[token]
            except:
                pass
        embeddings.append(embedding)
    return embeddings


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


def process_documents_seperate(documents, apply_stemming=False):
    tokenized_docs = [word_tokenize(text) for text in documents]
    lowered_docs = []

    stop_words = set(stopwords.words('english'))
    for doc in tokenized_docs:
        lowered_doc = [word.lower() for word in doc if word not in stop_words]
        lowered_docs.append(lowered_doc)

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


def main():
    create_mean_embedding()


if __name__ == "__main__":
    main()


