import os
import time
import argparse
import numpy as np

from PIL import Image
from sklearn.cluster import KMeans


def get_args_parser():
    parser = argparse.ArgumentParser('relabel found classes to num_classes in total', add_help=False)
    # Directory to save annotations and results
    parser.add_argument('--annotation_dir', type=str, help='annotation dir')
    # K-Means params
    parser.add_argument('--num_clusters', default=40, type=int, help='number of clusters for K-Means')
    
    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    # collect clusters on the training dataset
    clusters = []

    print("Collecting clusters on the training dataset", flush=True)
    print("-" * 10, flush=True)
    for category in os.listdir(f"{args.annotation_dir}/train"):
        for file_name in os.listdir(f"{args.annotation_dir}/train/{category}"):
            if 'clusters' in file_name:
                image_clusters = np.load(f"{args.annotation_dir}/train/{category}/{file_name}")
                clusters.append(image_clusters)
    # concatenate the clusters
    clusters = np.concatenate(clusters, axis=0)
            
    print(f"Fitting K-Means algorithm", flush=True)
    # define K-Means parameters
    K = args.num_clusters
    # fit K-Means
    kmeans = KMeans(n_clusters=args.num_clusters)
    kmeans.fit(clusters)

    # relabel annotations
    for partition in ['train', 'test']:
        print(f"Processing {partition} dataset", flush=True)
        print("-" * 10, flush=True)
        for category in os.listdir(f"{args.annotation_dir}/{partition}"):
            for file_name in os.listdir(f"{args.annotation_dir}/{partition}/{category}"):
                if 'clusters' in file_name:
                    # load found clusters 
                    centers = np.load(f"{args.annotation_dir}/{partition}/{category}/{file_name}")
                    # load annotation
                    segm = np.load(f"{args.annotation_dir}/{partition}/{category}/{file_name[:-13]}_segm.npy")

                    new_labels = kmeans.predict(centers)
                    for i, new_label in enumerate(new_labels):
                        segm[segm == i] = new_label

                    # save resulting annotation
                    np.save(f"{args.annotation_dir}/{partition}/{category}/{file_name[:-13]}.npy", segm.astype(np.uint8))
                    # remove redundant files
                    os.remove(f"{args.annotation_dir}/{partition}/{category}/{file_name}")
                    os.remove(f"{args.annotation_dir}/{partition}/{category}/{file_name[:-13]}_segm.npy")
