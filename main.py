from transformers import BertTokenizer,BertModel, AutoModel, AutoTokenizer
from sklearn.cluster import KMeans
import os 
import torch
import numpy as np
import evaluate
import torch.nn as nn
import pickle 
from utils import *
import hdbscan

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    
class custom_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertModel.from_pretrained("NlpHUST/vibert4news-base-cased")
        self.pool = MeanPooling()
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs['attention_mask'])
        return feature
    def forward(self, inputs):
        feature = self.feature(inputs)
        return feature

if __name__== "__main__":
    #tokenizer= BertTokenizer.from_pretrained("NlpHUST/vibert4news-base-cased")
    tokenizer= AutoTokenizer.from_pretrained("vinai/phobert-base")
    #bert_model = custom_model()
    #bert_model = BertModel.from_pretrained("NlpHUST/vibert4news-base-cased")
    bert_model = AutoModel.from_pretrained("vinai/phobert-base")
    #bert_model = AutoModel.from_pretrained("/home/hieungo3/Multi-Document-Summarization/my_awesome_model/checkpoint-4800")
    rouge = evaluate.load('rouge')
    #all_sentences = read_data()
    # all_labels = read_label()
    all_sentences = read_data_tok()
    all_labels = read_label_tok(is_flatten=False)
    
    with open('val_cluster.pkl', 'rb') as file:
        val_cluster = pickle.load(file)
    stop = 0
    cluster_results = []
    for cluster_name, cluster_sentences in all_sentences.items():
        if cluster_name in val_cluster:
            print("Calculate", cluster_name)
            inputs = tokenizer(cluster_sentences, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                #features = bert_model(inputs)
                features = bert_model(**inputs).last_hidden_state[:, 0, :]

            kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(features)
            # Retrieve sentence closest to centroid for each cluster
            cluster_representatives = []
            for i in range(len(kmeans.cluster_centers_)):
                centroid = kmeans.cluster_centers_[i]
                distances = [np.linalg.norm(embedding - centroid) for embedding in features]
                closest_index = np.argmin(distances)
                closest_text = cluster_sentences[closest_index]
                cluster_representatives.append(closest_text)

            prediction_holder = []
            # Print cluster representatives
            for i, representative in enumerate(cluster_representatives):
                #print(f"Cluster {i+1} representative: {representative}")
                prediction_holder.append(representative.strip())
            prediction = "\n".join(prediction_holder)

            # Calculate ROUGE on each cluster
            cluster_results.append(
                rouge.compute(
                    predictions=[prediction],
                    references=[all_labels[cluster_name]]
                )
            )
            print("Prediction: \n", prediction)
            for i, label in enumerate(all_labels[cluster_name]):
                print(f"Labels {i}: \n", label)

            stop += 1
            if stop == 1:
                break 
    
    # Final mean ROUGE scores
    s = 0
    s2 = 0
    for i in range(len(cluster_results)):
        s += cluster_results[i]["rouge1"] 
        s2 += cluster_results[i]["rouge2"] 
    mean_s = s/len(cluster_results)
    mean_s2 = s2/len(cluster_results)
    print('Rouge 1:' , mean_s)
    print('Rouge 2:', mean_s2)
