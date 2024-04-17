import json
import numpy as np
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm 
from numpy.linalg import norm

DIMENSIONS = 300

glove_vocabulary = set()

# taxonomy based on the image of https://earsc-portal.eu/display/EOwiki/EO+Taxonomy
# could create more fine grained taxonomy from table of page 37
domain_taxonomy = {
    'security & safety': ['customs', 'borders', 'health', 'epidemics', 'diseases', 'food security', 'production', 'meteorology'],
    'marine & maritime': ['coastal', 'metocean', 'biodiversity', 'marine', 'ecosystems', 'fisheries', 'marine', 'pollution', 'sea-ice', 'icebergs', 'shipping'],
    'atmosphere & climate': ['atmosphere', 'climate', 'change'],
    'disasters & geohazards': ['floods', 'landslides', 'earthquakes', 'fires', 'volcanoes'],
    'land': ['agriculture', 'biodiversity',  'land', 'ecosystems', 'inland', 'water', 'forests', 'snow',  'ice', 'topography', 'motion', 'land', 'cover', 'geology'],
    'built environment': ['waste', 'urban', 'areas', 'infrastructure', 'transport', 'networks']
}

def process_abstracts(documents):
    tokenized_docs = [word_tokenize(text) for text in documents]
    lowered_docs = []

    stop_words = set(stopwords.words('english'))
    for doc in tokenized_docs:
        lowered_doc = [word.lower() for word in doc if word not in stop_words]
        lowered_doc = [word for word in lowered_doc if word not in punctuation]
        lowered_docs.append(lowered_doc)

    return lowered_docs


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


def load_glove_model(glove_file):
    embeddings = {}

    with open(glove_file, 'r', encoding="utf8") as file:
        for i, line in tqdm(enumerate(file)):
            split_line = line.split(" ")
            embedding = np.array(split_line[-DIMENSIONS:], dtype=np.float64)
            word = split_line[0]
            glove_vocabulary.add(word)
            embeddings[word.lower()] = embedding

    return embeddings


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


def calc_similarity(corpus_embedding, document_embedding):
    dot_product = np.dot(corpus_embedding, document_embedding)
    norm_corpus = norm(corpus_embedding)
    norm_doc = norm(document_embedding)
    return dot_product / (norm_corpus * norm_doc)


def main():
    clean_abstracts = process_abstracts(get_abstracts("data/raw_paper_abstracts.json"))
    glove_embeddings = load_glove_model("data/glove.42B.300d/glove.42B.300d.txt")
    embeddings = calc_multiple_embeddings_from_tokens(glove_embeddings, clean_abstracts)

    for i, embedding in enumerate(embeddings):
        similarities = {}
        for domain in domain_taxonomy:
            similarities[domain] = {keyword:calc_similarity(embedding, glove_embeddings[keyword]) for keyword in domain_taxonomy[domain] if keyword in glove_vocabulary}
        print(similarities)
        if i > 10:
            break

    
    # ????? approach ?????
    # classified_abstracts = {k:[] for k in domain_taxonomy}
    # for abstract in clean_abstracts:
    #     domain_keyword_count = {}
    #     for domain in domain_taxonomy:
    #         domain_keyword_count[domain] = {word:abstract.count(word) for word in domain_taxonomy[domain]}
    #     print(domain_keyword_count)
    #     break


if __name__ == "__main__":
    main()