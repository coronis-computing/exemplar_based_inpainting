"""Console script for exemplar_based_inpainting."""
import argparse
import sys
import cv2
from exemplar_based_inpainting.simple_image_masker import simple_image_masker
from exemplar_based_inpainting.inpainter import Inpainter
import numpy as np


def main():
    """Console script for exemplar_based_inpainting."""
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help="The input image to inpaint")
    parser.add_argument('-o', '--out_image', default="", dest="out_image", help="The output inpainted image")
    parser.add_argument('--mask', type=str, default="", help="The mask file, of the same size as the input image, with the areas to inpaint as white.")
    parser.add_argument('--patch_size', '-p', type=int, default=9, help="The size of the inpainting patches.")
    parser.add_argument('--search_original_source_only', action='store_true', help="Do not search for filling patches in the growing inpainted area.")
    parser.add_argument('--search_color_space', type=str, default="bgr", help="Color space to use when searching for the next best filler patch. Options available: \"bgr\", \"hsv\", \"lab\", \"gray\". In case \"gray\" is selected, the input image will also be loaded assuming it is a graycale image. Defaults to \"bgr\".")
    parser.add_argument('--patch_preference', type=str, default="any", help="In case there are multiple patches in the image with the same similarity, this parameter decides which one to choose. Options: 'closest' (the one closest to the query patch in the front), 'any', 'random'")
    parser.add_argument('--similarity_measure', type=str, default="sqdiff", help="Similarity measure to use when looking for similar patches to the ones in the filling front. Available: \"sqdiff\", \"sqdiff_normed\", \"ccorr\", \"ccorr_normed\", \"ccoeff\", \"ccoeff_normed\"")
    parser.add_argument('--hide_progress_bar', action='store_true', help="Hides the progress bar.")
    parser.add_argument('--plot_progress', action='store_true', help="Plots the inpainting process if set.")    
    parser.add_argument('--plot_progress_wait_ms', type=int, default=1, help="How many milliseconds to wait after plotting each update to the progress on screen.")
    parser.add_argument('--hide_result', action='store_true', help="Disables the plotting of the inpainting result after finishing the process.")
    parser.add_argument('--out_progress_dir', type=str, default="", help="Stores the inpainting progress as individual images in the specified directory.")
    parser.add_argument('--out_mask', type=str, default="", help="The output mask file, only used when --mask is not set, to store the mask that you drawn manually.")
    args = parser.parse_args()

    # Load the image
    if args.search_color_space == "gray":
        img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(args.image)
    if img is None:
        print("The image file was not found")
        return -1

    # Load the mask or mark it manually
    if not args.mask:
        print("Please select the area to inpaint in the emerging window.")
        mask = simple_image_masker(img)
        if args.out_mask:
            print("Saving the selected mask to a file")
            cv2.imwrite(args.out_mask, mask)
    else:
        mask = cv2.imread(args.mask, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print("The mask file was not found")
            return -1
        if len(mask.shape) == 3:
            print("[WARNING] Input mask is not a single-channel image, taking a single channel")
            mask = mask[:, :, 0]
    
    # Inpaint
    inpainter = Inpainter(args.patch_size, 
                          args.search_original_source_only, 
                          args.search_color_space,
                          args.plot_progress, 
                          args.out_progress_dir, 
                          not args.hide_progress_bar, 
                          args.patch_preference,
                          args.similarity_measure,
                          args.plot_progress_wait_ms)
    inpainted_img = inpainter.inpaint(img, mask)

    # Write the result
    if args.out_image:
        cv2.imwrite(args.out_image, inpainted_img)

    # Show result?
    if not args.hide_result:
        print("Showing the result (press any key to end)")
        cv2.imshow("Exemplar-based inpainting result", inpainted_img)
        cv2.waitKey()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
