from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse

import torch
from tqdm import tqdm

def parse_args():
    """
    Parse the following arguments for a default parser
    """
    parser = argparse.ArgumentParser(
        description="Running experiments"
    )
    parser.add_argument(
        "--m",
        dest="model_name",
        help="model_name",
        default="dinov2",
        type=str,
    )
    parser.add_argument(
        "--d",
        dest="dataset",
        help="dataset",
        default="coco",
        type=str,
    )
    parser.add_argument(
        "-t",
        dest="text",
        help="text",
        action='store_true',
    )
    parser.add_argument(
        "--nc",
        dest="n_clusters",
        help="number of clusters",
        default=320,
        type=int,
    )
    return parser.parse_args()

def cluster_text(model, dataset, n_clusters):
    source_base_large = torch.load(f"data/{dataset}_{model}_text.pt")
    clustering_matrix = source_base_large.cpu().numpy()
    kmeans = KMeans(n_clusters= n_clusters, random_state=0).fit(clustering_matrix)
    
    cos_sims = cosine_similarity(kmeans.cluster_centers_, clustering_matrix)
    closest_indices = np.argsort(cos_sims, axis=1)[:,-1:]
    closest_indices = list(set(closest_indices.reshape(-1)))
    choose = closest_indices
    
    torch.save(choose, f'data/{dataset}_{model}_text_cluster.pt')

def cluster_img(model, dataset, n_clusters):
    source_base_large = torch.load(f"data/{dataset}_{model}_img.pt")
    clustering_matrix = source_base_large.cpu().numpy()
    kmeans = KMeans(n_clusters= n_clusters, random_state=0).fit(clustering_matrix)
    
    cos_sims = cosine_similarity(kmeans.cluster_centers_, clustering_matrix)
    closest_indices = np.argsort(cos_sims, axis=1)[:,-1:]
    closest_indices = list(set(closest_indices.reshape(-1)))
    choose = closest_indices
    
    torch.save(choose, f'data/{dataset}_{model}_img_cluster.pt')


if __name__ == "__main__":
    args = parse_args()

    model = args.model_name
    dataset = args.dataset
    text = args.text
    n_clusters = args.n_clusters

    if text:
        cluster_text(model, dataset, n_clusters)
    else:
        cluster_img(model, dataset, n_clusters)

    

    
    
    

    
        
    



