# Command line tool

After installation, you should have the `exemplar_based_inpainting` command line tool available. 

The only required parameter is the input image to inpaint. We provide some examples to test the tool in the `data` folder of this project, but you should be able to use it on any image of your choice.

If you do not want to store the results, you can call it as follows:

```
exemplar_based_inpainting <image_path>
```

In this case, the user will be presented with a simple masking tool to manually select the region of the image to inpaint. If you had this mask already computed, you can also pass it with the optional parameter `--mask`. The result will be presented on screen when the process finishes (this can be disabled via the `--hide_result` flag). Alternatively, the result can be stored to the file specified by the `-o` option, and the inpainting process can also be visualized with the `--plot_progress` flag or stored to disk using the `--out_progress_dir` option. 

In order to get the full list of parameters, you can run the app with the `--help` flag:

```
usage: exemplar_based_inpainting [-h] [-o OUT_IMAGE] [--mask MASK] [--patch_size PATCH_SIZE]
                                 [--search_original_source_only] [--search_color_space SEARCH_COLOR_SPACE]
                                 [--patch_preference PATCH_PREFERENCE] [--hide_progress_bar] [--plot_progress]
                                 [--hide_result] [--out_progress_dir OUT_PROGRESS_DIR] [--out_mask OUT_MASK]
                                 image

positional arguments:
  image                 The input image to inpaint

options:
  -h, --help            show this help message and exit
  -o OUT_IMAGE, --out_image OUT_IMAGE
                        The output inpainted image
  --mask MASK           The mask file, of the same size as the input image, with the areas to inpaint as white.
  --patch_size PATCH_SIZE, -p PATCH_SIZE
                        The size of the inpainting patches.
  --search_original_source_only
                        Do not search for filling patches in the growing inpainted area.
  --search_color_space SEARCH_COLOR_SPACE
                        Color space to use when searching for the next best filler patch
  --patch_preference PATCH_PREFERENCE
                        In case there are multiple patches in the image with the same similarity, this parameter
                        decides which one to choose. Options: 'closest' (the one closest to the query patch in the
                        front), 'any', 'random'
  --hide_progress_bar   Hides the progress bar.
  --plot_progress       Plots the inpainting process if set.
  --hide_result         Disables the plotting of the inpainting result after finishing the process.
  --out_progress_dir OUT_PROGRESS_DIR
                        Stores the inpainting progress as individual images in the specified directory.
  --out_mask OUT_MASK   The output mask file, only used when --mask is not set, to store the mask that you drawn
                        manually.
```

You can see that most of them are self-explanatory. However, the `--patch_preference` may need a bit of explanation. This parameter decides which patch to select when there are multiple source patches that have the same similarity to the current patch to fill in the front. Since we are only comparing the *valid* pixels within a patch, it is likely than more than one part of an image has the same similarity value, specially in images with regions of constant colors. Therefore, similarity score alone does not disambiguate which patch is better, and we need to follow an *heuristic*. The available heuristics are:

* `any`: selects any patch (in this case, the one that was found first in the image traversal when computing the SSD).
* `closest`: selects the patch with the highest similarity score that is closest to the query point on the filling front.
* `random`: selects a random patch among the ones sharing the highest similarity score.

Finally, another parameter worth mentining is `--search_color_space`, which indicates the color space in which the patch similarity check will be performed.