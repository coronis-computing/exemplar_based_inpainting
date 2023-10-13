"""Console script for exemplar_based_inpainting."""
import argparse
import sys
import cv2
from exemplar_based_inpainting.simple_image_masker import simple_image_masker
from exemplar_based_inpainting.inpainter import Inpainter


def main():
    """Console script for exemplar_based_inpainting."""
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help="The input image to inpaint")
    parser.add_argument('--mask', type=str, default="", help="The mask file, of the same size as the input image, with the areas to inpaint as white.")
    parser.add_argument('--patch_size', '-p', type=int, default=9, help="The size of the inpainting patches.")
    parser.add_argument('--plot_progress', action='store_true', help="Plots the inpainting process if set.")
    parser.add_argument('--out_mask', type=str, default="", help="The output mask file, only used when --mask is not set. Use it if you want to store the mask that you drawn manually.")
    args = parser.parse_args()

    # Load the image
    img = cv2.imread(args.image)
    if img is None:
        print("The mask file was not found")
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
    
    # Inpaint
    inpainter = Inpainter(args.patch_size, args.plot_progress)
    inpainter.inpaint(img, mask)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
