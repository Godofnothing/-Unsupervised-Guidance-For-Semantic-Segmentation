import os
import cv2
import argparse
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser('make smoother ground masks', add_help=False)
    # Directory to save annotations and results
    parser.add_argument('--annotation_dir', type=str, help='annotation dir')
    # Morphology
    parser.add_argument('--closure_kernel_size', default=5, type=int, help='size of the closure kernel')
    parser.add_argument('--opening_kernel_size', default=5, type=int, help='size of the opening kernel')
    
    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    # get closure and opening kernel
    closure_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.closure_kernel_size,) * 2)
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.opening_kernel_size,) * 2)

    # relabel annotations
    for partition in ['train', 'test']:
        print(f"Processing {partition} dataset", flush=True)
        print("-" * 10, flush=True)
        for category in os.listdir(f"{args.annotation_dir}/{partition}"):
            for file_name in os.listdir(f"{args.annotation_dir}/{partition}/{category}"):
                mask = np.load(f"{args.annotation_dir}/{partition}/{category}/{file_name}")
                # apply closure to the mask
                mask_closure = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closure_kernel)
                # apply opening to the mask
                mask_opening = cv2.morphologyEx(mask_closure, cv2.MORPH_OPEN, opening_kernel)
                # save resulting mask
                np.save(f"{args.annotation_dir}/{partition}/{category}/{file_name}", mask_opening)
