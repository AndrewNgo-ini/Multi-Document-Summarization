import os

path = "/Users/ngohieu/textsum/VietnameseMDS/clusters"

def read_label_tok(is_flatten):
    all_sentences = {}
    pwd = path
    clusters_dir = os.listdir(pwd)
    for cluster in clusters_dir:
        documents = os.listdir(pwd + '/' + cluster)
        valid_documents = list(filter(lambda x: "ref" in x, documents))
        valid_documents = list(filter(lambda x: "tok" in x, valid_documents))
        cluster_sentences = []
        for vd in valid_documents:
            file = open(pwd + '/' + cluster + '/' + vd).read().strip()
            if is_flatten:
                sentences = file.split("\n")
                cluster_sentences.extend(sentences)
            else:
                cluster_sentences.append(file)

        all_sentences[cluster] = cluster_sentences
    return all_sentences

def read_data_tok():
    all_sentences = {}
    pwd = path
    clusters_dir = os.listdir(pwd)
    for cluster in clusters_dir:
        documents = os.listdir(pwd + '/' + cluster)
        valid_documents = list(filter(lambda x: "body.tok.txt" in x, documents))
        cluster_sentences = []
        for vd in valid_documents:
            file = open(pwd + '/' + cluster + '/' + vd).read().strip()
            sentences = file.split("\n")
            cluster_sentences.extend(sentences)
        all_sentences[cluster] = cluster_sentences
    return all_sentences

def read_data():
    all_sentences = {}  
    pwd = path
    clusters_dir = os.listdir(pwd)
    for cluster in clusters_dir:
        documents = os.listdir(pwd + '/' + cluster)
        valid_documents = list(filter(lambda x: "body.txt" in x, documents))
        cluster_sentences = []
        for vd in valid_documents:
            file = open(pwd + '/' + cluster + '/' + vd).read()
            sentences = file.split(".")
            cluster_sentences.extend(sentences)
        all_sentences[cluster] = cluster_sentences
    return all_sentences

def read_label():
    all_sentences = {}
    pwd = path
    clusters_dir = os.listdir(pwd)
    for cluster in clusters_dir:
        documents = os.listdir(pwd + '/' + cluster)
        valid_documents = list(filter(lambda x: "ref" in x, documents))
        valid_documents = list(filter(lambda x: "tok" not in x, valid_documents))
        cluster_sentences = []
        for vd in valid_documents:
            file = open(pwd + '/' + cluster + '/' + vd).read()
            cluster_sentences.append(file)
        all_sentences[cluster] = cluster_sentences
    return all_sentences