import cv2
import numpy as np
from exemplar_based_inpainting.image_utils import image_gradients, get_roi, roi_area, roi_shape, fill_roi, to_three_channels
import random
import timeit
import os
from enum import Enum
from rich.progress import Progress

class PatchPreference(Enum):
    ANY = 0
    CLOSEST = 1
    RANDOM = 2

class SearchColorSpace(Enum):
    BGR = 0
    HSV = 1
    LAB = 2
    GRAY = 3

class Inpainter():
    def __init__(self, 
                 patch_size=9, 
                 search_original_source_only=False,
                 search_color_space="bgr",
                 plot_progress=False, 
                 out_progress_dir="",                  
                 show_progress_bar=True, 
                 patch_preference="closest",
                 similarity_measure="sqdiff",
                 plot_progress_wait_ms=500):
        """Inpainter Constructor. 

        Args:
            patch_size (int, optional): Size of the inpainting patch. Defaults to 9.
            search_original_source_only (bool, optional): If true, just the original source image - mask will be searched for inpainting patches. If false, the growing inpainting area will also be taken into account. Defaults to False.
            search_color_space (str, optional): Color space to use when searching for the next best filler patch. Options available: "bgr", "hsv", "lab", "gray". In case gray is selected, the input image must also be grayscale. Defaults to "bgr".
            plot_progress (bool, optional): Activates/deactivates the plotting of the inpainting process. Defaults to False.
            out_progress_dir (str, optional): Set to a directory to get the same output as in "plot_progress=True" but stored to files. Defaults to "".
            show_progress_bar (bool, optional): Activates/deactivates the progress bar. Defaults to True.
            patch_preference (str, optional): When more than a patch has the same similarity score, this parameter selects which one to choose. Available: "any", "closest", "random". Defaults to "closest".
            plot_progress_wait_ms (int, optional): How many milliseconds to wait after plotting each update to the progress on screen. Defaults to 500."

        Raises:
            ValueError: Wrong patch preference.
            ValueError: Wrong color space.
        """
        self.patch_size = patch_size
        self.half_patch_size = (self.patch_size-1)//2
        self.search_original_source_only = search_original_source_only
        self.plot_progress = plot_progress
        self.plot_progress_wait_ms = plot_progress_wait_ms
        self.out_progress_dir = out_progress_dir
        if self.out_progress_dir:
            if not os.path.exists(self.out_progress_dir):
                os.makedirs(self.out_progress_dir)
        if patch_preference == "any":
            self.patch_preference = PatchPreference.ANY
        elif patch_preference == "closest":
            self.patch_preference = PatchPreference.CLOSEST
        elif patch_preference == "random":
            self.patch_preference = PatchPreference.RANDOM
        else:
            raise ValueError("Unknown patch preference \"" + patch_preference + "\"")
        if search_color_space == "bgr":
            self.search_color_space = SearchColorSpace.BGR
        elif search_color_space == "hsv":
            self.search_color_space = SearchColorSpace.HSV
        elif search_color_space == "lab":
            self.search_color_space = SearchColorSpace.LAB
        elif search_color_space == "gray":
            self.search_color_space = SearchColorSpace.GRAY
        else:
            raise ValueError("Unknown search color space \"" + search_color_space + "\"")            
        if similarity_measure == "sqdiff":
            self.similarity_measure = cv2.TM_SQDIFF
        elif similarity_measure == "sqdiff_normed":
            self.similarity_measure = cv2.TM_SQDIFF_NORMED
        elif similarity_measure == "ccorr":
            self.similarity_measure = cv2.TM_CCORR
        elif similarity_measure == "ccorr_normed":
            self.similarity_measure = cv2.TM_CCORR_NORMED
        elif similarity_measure == "ccoeff":
            self.similarity_measure = cv2.TM_CCOEFF
        elif similarity_measure == "ccoeff_normed":
            self.similarity_measure = cv2.TM_CCOEFF_NORMED
        else:
            raise ValueError("Unknown similarity measure \"" + similarity_measure + "\"")
        self.show_progress_bar = show_progress_bar

    def _initialize(self, image, mask):
        """Initializes the inpainting problem
        
        
        Args:
            image (numpy.array): image to inpaint, in BGR color space.
            mask (numpy.array): mask containing the area to inpaint. Should be a binary image (0 == source area, 255 == to inpaint).
        """        
        #self.image = image.astype('uint8')
        self.image = image
        if self.search_color_space == SearchColorSpace.GRAY and len(self.image.shape) > 2:
            raise ValueError("If you select the color space to be gray, the input image is also expected to be grayscale (i.e., a single-channel image)")

        self.mask = mask.astype('uint8')
        self.mask = (self.mask > 128).astype('uint8') # Important: we change the change mask to be 0s and 1s!

        # Non initialized attributes
        self.working_image = np.copy(self.image)
        self.working_mask = np.copy(self.mask)        
        self.front = None
        self.confidence = (1 - self.mask).astype(float)
        self.data = np.zeros(self.image.shape[:2])
        self.priority = None
        self.num_pixels = self.image.shape[0] * self.image.shape[1]
        self.num_pixels_to_fill = np.count_nonzero(self.working_mask)        

        # Remove the target region from the image
        inverse_mask = (1-self.working_mask)
        if self.search_color_space == SearchColorSpace.GRAY:
            self.working_image = self.working_image * inverse_mask
        else:
            inverse_mask_3 = to_three_channels(inverse_mask)
            self.working_image = self.working_image * inverse_mask_3

    def inpaint(self, image, mask):        
        """Inpaint using the exemplar-based inpainting algorithm

        This method implements the algorithm described in:

        1. `Criminisi A, Pérez P, Toyama K. Region filling and object removal by exemplar-based image inpainting[J]. IEEE Transactions on image processing, 2004, 13(9): 1200-1212.`

        Args:
            image (numpy.array): image to inpaint, in BGR color space.
            mask (numpy.array): mask containing the area to inpaint. Should be a binary image (0 == source area, 255 == to inpaint).

        Raises:
            ValueError: The image and mask must be of the same size.

        Returns:
            (numpy.array): inpainted image.
        """
        # Initialization
        if image.shape[:2] != mask.shape:
            raise ValueError("The input image and mask must be of the same size.")        
        self._initialize(image, mask)

        # Inpainting        
        with Progress() as progress:
            task = progress.add_task("Inpainting...", total=self.num_pixels_to_fill, visible=self.show_progress_bar, transient=self.show_progress_bar)
            remaining = self.num_pixels_to_fill
            self.iter = 0
            while remaining != 0:
                self._find_front()            
                self._update_priority()                
                hp_pixel = self._highest_priority_pixel()
                best_source_patch = self._find_source_patch(hp_pixel)

                if self.plot_progress or self.out_progress_dir:
                    self._plot_current_state(hp_pixel, best_source_patch)

                self._update_image(hp_pixel, best_source_patch)

                self.iter += 1
                remaining = np.count_nonzero(self.working_mask)

                progress.update(task, completed=self.num_pixels_to_fill-remaining)
        
        if self.plot_progress:
            cv2.destroyWindow("Inpainting process")

        return self.working_image

    def _find_front(self):
        self.front = (cv2.Laplacian(self.working_mask, -1) > 0).astype('uint8')        

    def _plot_current_state(self, hp_pixel, best_source_patch):
        if self.working_image.dtype == np.float32:
            disp_img = self.working_image * 255
            disp_img = disp_img.astype(np.uint8)
        else:
            disp_img = self.working_image.copy()
        if self.search_color_space == SearchColorSpace.GRAY:
            disp_img = to_three_channels(disp_img)
        disp_img = cv2.drawMarker(disp_img, (hp_pixel[1], hp_pixel[0]), (255, 0, 0), cv2.MARKER_TILTED_CROSS, self.patch_size)
        disp_img = cv2.rectangle(disp_img, (best_source_patch[1][0], best_source_patch[0][0]), (best_source_patch[1][1], best_source_patch[0][1]), (0, 255, 0), 1)
        disp_img[self.front > 0] = np.array([0, 0, 255], dtype=np.uint8)        
        if self.out_progress_dir:
            cv2.imwrite(os.path.join(self.out_progress_dir, "inpainting_step_{:08d}.png".format(self.iter)), disp_img)
        
        if self.plot_progress:
            cv2.namedWindow("Inpainting process", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Inpainting process", disp_img)
            cv2.waitKey(self.plot_progress_wait_ms)        

    def _update_priority(self):
        self._update_confidence()
        self._update_data()
        self.priority = self.confidence * self.data * self.front.astype(np.float64)

    def _update_confidence(self):
        new_confidence = np.copy(self.confidence)
        front_positions = np.argwhere(self.front == 1)
        for point in front_positions:
            patch = self._get_patch(point)
            new_confidence[point[0], point[1]] = sum(sum(
                get_roi(self.confidence, patch)
            ))/roi_area(patch)
        self.confidence = new_confidence

    def _update_data(self):
        front_isophotes = self._compute_gradients_ignoring_mask()
        front_normals = self._compute_front_normals()

        front_data = front_isophotes*front_normals
        self.data = np.sqrt(
            front_data[:, :, 0]**2 + front_data[:, :, 1]**2
        ) + 0.001 

    def _compute_front_normals(self):
        [ny, nx] = np.gradient(self.working_mask.astype(float))
        height, width = nx.shape[:2]
        norm = np.sqrt(ny**2 + nx**2)
        norm[norm == 0] = 1
        nx = nx/norm
        ny = ny/norm
        return np.dstack((-ny, nx))
    
    def _highest_priority_pixel(self):
        point = np.unravel_index(self.priority.argmax(), self.priority.shape)
        return point

    def _get_patch(self, point):
        height, width = self.working_image.shape[:2]
        patch = [
            [
                max(0, point[0] - self.half_patch_size),
                min(point[0] + self.half_patch_size, height-1)
            ],
            [
                max(0, point[1] - self.half_patch_size),
                min(point[1] + self.half_patch_size, width-1)
            ]
        ]
        return patch

    def _find_source_patch(self, target_pixel):
        # Get the target patch from the inpainting image
        target_patch = self._get_patch(target_pixel)
        template = get_roi(self.working_image, target_patch)

        # Convert to another color space?
        if self.search_color_space == SearchColorSpace.BGR or self.search_color_space == SearchColorSpace.GRAY:
            working_image = self.working_image
        elif self.search_color_space == SearchColorSpace.HSV:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
            working_image = cv2.cvtColor(self.working_image, cv2.COLOR_BGR2HSV)
        elif self.search_color_space == SearchColorSpace.LAB:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2LAB)
            working_image = cv2.cvtColor(self.working_image, cv2.COLOR_BGR2LAB)        
        
        # Find the best match using template matching
        template_mask = 1-get_roi(self.working_mask, target_patch)
        tm = cv2.matchTemplate(working_image, template, self.similarity_measure, None, template_mask.astype(np.uint8)*255)        

        # Do not include any patch from the source image that may have a pixel within the original mask
        struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, (self.patch_size, self.patch_size))
        # dilated_mask = cv2.dilate(self.working_mask.astype(np.uint8)*255, struct_element)        
        
        if self.search_original_source_only:
            # Prevent taking texture from the inpainted area
            valid_mask = cv2.filter2D((1-self.mask.astype(np.int16)), -1, struct_element, anchor=(0,0)) == ((self.patch_size)*(self.patch_size))
            # valid_mask = valid_mask | valid_mask_ext
        else:
            valid_mask = cv2.filter2D((1-self.working_mask.astype(np.int16)), -1, struct_element, anchor=(0,0)) == ((self.patch_size)*(self.patch_size))
        
        # Set the invalid areas to a high/low value (depending on the similarity measure), so that they are not selected
        valid_mask = valid_mask[0:tm.shape[0], 0:tm.shape[1]]
        if self.similarity_measure == cv2.TM_SQDIFF or self.similarity_measure == cv2.TM_SQDIFF_NORMED:
            tm[~valid_mask] = np.max(tm)+1
        else:
            tm[~valid_mask] = np.min(tm)-1

        # Find the best score (minimum value when using an SSD-line as measure, maximum otherwise)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(tm)
        if self.similarity_measure == cv2.TM_SQDIFF or self.similarity_measure == cv2.TM_SQDIFF_NORMED:
            best_val = min_val
            best_loc = min_loc
        else:
            best_val = max_val
            best_loc = max_loc
        
        vals_coords = cv2.findNonZero((tm == best_val).astype(np.uint8))

        if len(vals_coords) != 1:
            # In case there are more than a single instance of the minimum, we need to select one based on the user-set heuristic
            if self.patch_preference == PatchPreference.ANY:
                # Choose any (the first one in the list)
                best_loc = vals_coords[0][0]
            elif self.patch_preference == PatchPreference.CLOSEST:
                # Choose the one closer to the current pixel
                distances = np.sqrt((vals_coords[:,:,0] - target_pixel[1]) ** 2 + (vals_coords[:,:,1] - target_pixel[0]) ** 2)
                nearest_index = np.argmin(distances)
                best_loc = vals_coords[nearest_index][0]
            elif self.patch_preference == PatchPreference.RANDOM:
                # Random choice
                ind = random.randrange(len(vals_coords))
                best_loc = vals_coords[ind][0]
            else:
                raise ValueError("Unknown patch preference")

        best_patch = [[best_loc[1], best_loc[1]+self.patch_size-1], [best_loc[0], best_loc[0]+self.patch_size-1]]
        return best_patch
    
    def _update_image(self, target_pixel, source_patch):
        # Get the target patch around the target pixel on the front
        target_patch = self._get_patch(target_pixel)

        # Update the confidence after adding this patch
        pixels_positions = np.argwhere(
            get_roi(
                self.working_mask,
                target_patch
            ) == 1
        ) + [target_patch[0][0], target_patch[1][0]]
        patch_confidence = self.confidence[target_pixel[0], target_pixel[1]]
        for point in pixels_positions:
            self.confidence[point[0], point[1]] = patch_confidence

        # Mix the pixels from source and target patches together
        mask = get_roi(self.working_mask, target_patch)
        if self.search_color_space != SearchColorSpace.GRAY:
            mask = to_three_channels(mask)
        source_data = get_roi(self.working_image, source_patch)
        target_data = get_roi(self.working_image, target_patch)

        if target_data.shape != source_data.shape:
            # In some cases, the target patch may get over the borders. 
            # In such cases, the target_data will be smaller than the source_data (source_data is enforced to be within the image in _find_source_patch)
            source_data = source_data[0:target_data.shape[0], 0:target_data.shape[1]]
        
        new_data = source_data*mask + target_data*(1-mask)

        fill_roi(
            self.working_image,
            target_patch,
            new_data
        )
        fill_roi(
            self.working_mask,
            target_patch,
            0
        )

    def _compute_gradients_ignoring_mask(self):
        working_image_float = self.working_image.astype(np.float64)/255.0
        height, width = working_image_float.shape[:2]
        inds = [*range(1, height)]
        inds.append(height-1)
        gy = working_image_float[inds, :].astype(np.float64)
        gy = gy - working_image_float
        inds = [*range(1, width)]
        inds.append(width-1)
        gx = working_image_float[:, inds].astype(np.float64)
        gx = gx - working_image_float
        front_positions = np.argwhere(self.front == 1)        
        for point in front_positions:
            r = point[0] # Row 
            c = point[1] # Column

            # Ys
            if r+1 < height and self.working_mask[r+1, c] == 0:
                pass # Already computed above (gy = gy - self.working_image)
            elif r-1 >= 0 and self.working_mask[r-1,c] == 0:
                gy[r, c] = working_image_float[r-1, c] - working_image_float[r, c]
            else:
                gy[r, c] = 0

            # Xs
            if c+1 < width and self.working_mask[r, c+1] == 0:
                pass # Already computed above (gx = gx - self.working_image)
            elif c-1 >= 0 and self.working_mask[r, c-1] == 0:
                gx[r, c] = working_image_float[r, c-1] - working_image_float[r, c]
            else:
                gx[r, c] = 0
        if len(working_image_float.shape) == 3:                        
            mask_3d = to_three_channels(1-self.working_mask)
            gx = gx*mask_3d
            gy = gy*mask_3d        
            gx = np.sum(gx, 2)/working_image_float.shape[2]
            gy = np.sum(gy, 2)/working_image_float.shape[2]
        else:
            gx = gx*(1-self.working_mask)
            gy = gy*(1-self.working_mask)
        return np.dstack((gx, gy))