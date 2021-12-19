import os
import cv2
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
        # Morphology
    parser.add_argument('--closure_kernel_size', default=5, type=int, help='size of the closure kernel')
    parser.add_argument('--opening_kernel_size', default=5, type=int, help='size of the opening kernel')

    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    # collect clusters on the training dataset
    clusters = []
    # get closure and opening kernel
    closure_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.closure_kernel_size,) * 2)
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.opening_kernel_size,) * 2)

    print("Collecting clusters on the training dataset", flush=True)
    print("-" * 10, flush=True)
    for file_name in os.listdir(args.annotation_dir):
        if 'clusters' in file_name:
            image_clusters = np.load(f"{args.annotation_dir}/{file_name}")
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
    print("-" * 10, flush=True)
    for file_name in os.listdir(args.annotation_dir):
        if 'clusters' in file_name:
            # load found clusters 
            centers = np.load(f"{args.annotation_dir}/{file_name}")
            # load annotation
            segm = np.load(f"{args.annotation_dir}/{file_name[:-13]}_segm.npy")

            new_labels = kmeans.predict(centers)
            for i, new_label in enumerate(new_labels):
                segm[segm == i] = new_label

            # cast to uint8
            segm = segm.astype(np.uint8)
            # apply closure to the mask
            segm_closure = cv2.morphologyEx(segm, cv2.MORPH_CLOSE, closure_kernel)
            # apply opening to the segm
            segm_opening = cv2.morphologyEx(segm_closure, cv2.MORPH_OPEN, opening_kernel)
            # save resulting annotation
            np.save(f"{args.annotation_dir}/{file_name[:-13]}.npy", segm_opening.astype(np.uint8))
            # remove redundant files
            os.remove(f"{args.annotation_dir}/{file_name}")
            os.remove(f"{args.annotation_dir}/{file_name[:-13]}_segm.npy")
