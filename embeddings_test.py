def calc_similarity(corpus_embedding, document_embedding):
    

DIMENSIONS = 300
with open("data/embeddings.txt", "r") as file:
    embeddings = {}
    for line in file:
        split_line = line.split(" ")
        embedding_type = split_line[0]
        embedding = split_line[-DIMENSIONS:]
        embeddings[embedding_type] = embedding

combined_embedding = embeddings["combined"]

