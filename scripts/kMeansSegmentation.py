import os
import cv2
import time
import argparse
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser('kNN segmentation with sklearn to determine clusters', add_help=False)
    # Paths to the datasets
    parser.add_argument('--train_dir', type=str, help='path to train dir')
    parser.add_argument('--test_dir', type=str, help='path to test dir')
    # K-Means params
    parser.add_argument('--num_clusters', default=5, type=int, help='number of clusters for K-Means')
    parser.add_argument('--attempts', default=10, type=int, help='number of attempts K-Means')
    # Directory to save annotations and results
    parser.add_argument('--annotation_dir', type=str, help='annotation dir')
    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    # get dataset dirs
    train_dir = args.train_dir
    test_dir  = args.test_dir

    # make directory for annotations
    for partition in ['train', 'test']:
        os.makedirs(f"{args.annotation_dir}/{partition}")

    # define K-Means parameters
    K = args.num_clusters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = args.attempts

    start = time.time()

    # process dataset
    for partition, data_dir in zip(['train', 'test'], [train_dir, test_dir]):
        print(f"Processing {partition} dataset", flush=True)
        print("-" * 10, flush=True)
        for category in os.listdir(data_dir):
            # make directory for class annotation if needed
            os.makedirs(f"{args.annotation_dir}/{partition}/{category}")
            for image_name in os.listdir(f"{data_dir}/{category}"):
                img = cv2.imread(f"{data_dir}/{category}/{image_name}")
                # BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # reshape image to (N, 3)
                img_flat = img.reshape((-1,3)).astype(np.float32)
                # get clusters
                ret, label, center = cv2.kmeans(img_flat, K, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
                # reshape labe to image.shape
                label = label.reshape(img.shape[:2])
                # save resulting annotation
                np.save(f"{args.annotation_dir}/{partition}/{category}/{image_name[:-4]}_segm.npy", label)
                # save found clusters 
                np.save(f"{args.annotation_dir}/{partition}/{category}/{image_name[:-4]}_clusters.npy", center)
            
    end = time.time()
    print(f"K-Means clustering on the whole dataset took {(end - start):.2f} seconds")
