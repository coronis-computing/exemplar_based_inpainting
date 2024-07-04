import cv2
import numpy as np
# from matplotlib import pyplot as plt
from exemplar_based_inpainting.image_utils import image_gradients, get_roi, roi_area, roi_shape, fill_roi, to_three_channels
from scipy.ndimage.filters import convolve
from skimage.filters import laplace
from skimage.color import rgb2gray, rgb2lab
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

class Inpainter():
    def __init__(self, 
                 patch_size=9, 
                 search_original_source_only=False,
                 search_color_space="bgr",
                 plot_progress=False, 
                 out_progress_dir="",                  
                 show_progress_bar=True, 
                 patch_preference="closest"):
        """Inpainter Constructor. 
        
        Args:
            patch_size (int, optional): Size of the inpainting patch. Defaults to 9.
            plot_progress (bool, optional): Activates/deactivates the plotting of the inpainting process. Defaults to False.
            out_progress_dir (str, optional): Set to a directory to get the same output as in "plot_progress=True" but stored to files. Defaults to "".
            show_progress_bar (bool, optional): Activates/deactivates the progress bar. Defaults to True.
            patch_preference (str, optional): In case there are multiple patches in the image with the same similarity, this parameter decides which one to choose. Options: 'closest' (the one closest to the query patch in the front), 'any', 'random'. Defaults to "closest".

        Raises:
            ValueError: _description_
        """
        self.patch_size = patch_size
        self.half_patch_size = (self.patch_size-1)//2
        self.search_original_source_only = search_original_source_only
        self.plot_progress = plot_progress
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
        else:
            raise ValueError("Unknown search color space \"" + search_color_space + "\"")            
        self.show_progress_bar = show_progress_bar

    def _initialize(self, image, mask):
        """Initializes the inpainting problem
        
        Args:
            image (numpy.array): image to inpaint, in BGR color space.
            mask (numpy.array): mask containing the area to inpaint. Should be a binary image (0 == source area, 255 == to inpaint).
        """        
        self.image = image.astype('uint8')
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
        rgb_inverse_mask = to_three_channels(inverse_mask)
        self.working_image = self.working_image * rgb_inverse_mask

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
                # best_source_patch = self._find_source_patch_2(hp_pixel)

                if self.plot_progress or self.out_progress_dir:
                    self._plot_current_state(hp_pixel, best_source_patch)

                self._update_image(hp_pixel, best_source_patch)

                self.iter += 1
                remaining = np.count_nonzero(self.working_mask)

                progress.update(task, completed=self.num_pixels_to_fill-remaining)
            
        return self.working_image

    def _find_front(self):
        self.front = (cv2.Laplacian(self.working_mask, -1) > 0).astype('uint8')
        # # laplacian_kernel = np.ones((3, 3), dtype = np.float32)
        # # laplacian_kernel[1, 1] = -8
        # laplacian_kernel = np.zeros((3, 3), dtype = np.float32)
        # laplacian_kernel[0, 1] = -1
        # laplacian_kernel[1, 0] = -1
        # laplacian_kernel[1, 2] = -1
        # laplacian_kernel[2, 1] = -1
        # laplacian_kernel[1, 1] = 4
        # self.front = (cv2.filter2D(self.working_mask, cv2.CV_32F, laplacian_kernel)> 0).astype('uint8')

        # front2 = (laplace(self.working_mask) > 0).astype('uint8')

        # fronts = np.zeros((self.front.shape[0], self.front.shape[1], 3))
        # fronts[:, :, 0] = self.front
        # fronts[:, :, 1] = front2
        # fronts[:, :, 2] = front_new
        # plt.imshow(fronts)
        # plt.title('front'), plt.xticks([]), plt.yticks([])
        # plt.show()

    def _plot_current_state(self, hp_pixel, best_source_patch):
        disp_img = self.working_image.copy()
        disp_front = self.front.copy()
        disp_front = to_three_channels(disp_front*255)
        disp_img = cv2.drawMarker(disp_img, (hp_pixel[1], hp_pixel[0]), (255, 0, 0), cv2.MARKER_TILTED_CROSS, self.patch_size)
        disp_front = cv2.drawMarker(disp_front, (hp_pixel[1], hp_pixel[0]), (0, 0, 255))
        disp_img = cv2.rectangle(disp_img, (best_source_patch[1][0], best_source_patch[0][0]), (best_source_patch[1][1], best_source_patch[0][1]), (0, 255, 0), 1)
        disp_img[self.front > 0] = np.array([0, 0, 255], dtype=np.uint8)
        best_roi = get_roi(self.working_image, best_source_patch)
        if self.out_progress_dir:
            cv2.imwrite(os.path.join(self.out_progress_dir, "inpainting_step_{:08d}.png".format(self.iter)), disp_img)
        
        if self.plot_progress:
            cv2.namedWindow("Inpainting process", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Inpainting process", disp_img)
            # cv2.namedWindow("Fill front", cv2.WINDOW_AUTOSIZE)
            # cv2.imshow("Fill front", disp_front)
            # cv2.namedWindow("best ROI", cv2.WINDOW_AUTOSIZE)
            # cv2.imshow("best ROI", best_roi)
            # cv2.namedWindow("Working mask", cv2.WINDOW_AUTOSIZE)
            # cv2.imshow("Working mask", self.working_mask.astype(np.uint8)*255)
            cv2.waitKey(0)
        # print("hi")
        # plt.imshow(self.last_template, cmap = 'gray')
        # plt.title('template'), plt.xticks([]), plt.yticks([]) 
        # plt.show()
        # plt.imshow(self.last_template_mask, cmap = 'gray')
        # plt.title('template_mask'), plt.xticks([]), plt.yticks([])
        # plt.show()
        # plt.imshow(self.last_tm,cmap = 'gray')
        # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        # plt.show(block=True)

        # disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)        
        # plt.imshow(disp_img)
        # plt.show(block=False)

    def _update_priority(self):
        self._update_confidence()
        self._update_data()
        self.priority = self.confidence * self.data * self.front.astype(np.float64)

        # plt.imshow(self.confidence)
        # plt.title('confidence'), plt.xticks([]), plt.yticks([])
        # plt.show(block=True)
        # plt.imshow(self.data)
        # plt.title('data'), plt.xticks([]), plt.yticks([])
        # plt.show(block=True)
        # plt.imshow(self.priority)
        # plt.title('priority'), plt.xticks([]), plt.yticks([])
        # plt.show(block=True)

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
        # start_ts = timeit.default_timer()
        # front_isophotes = self._compute_front_isophotes()
        # end_ts = timeit.default_timer()
        # print("_compute_front_isophotes time: ", end_ts-start_ts)

        # start_ts = timeit.default_timer()
        # front_isophotes = self._calc_gradient_matrix()
        # end_ts = timeit.default_timer()
        # print("_calc_gradient_matrix time: ", end_ts-start_ts)

        front_isophotes = self._compute_gradients_ignoring_mask()

        #front_normals = self._compute_front_normals()
        # front_normals = self._calc_normal_matrix()
        front_normals = self._compute_front_normals_2()

        # plt.imshow(front_normals[:, :, 0], cmap = 'gray')
        # plt.title('normal[:, :, 0]'), plt.xticks([]), plt.yticks([])
        # plt.show(block=True)
        # plt.imshow(front_normals[:, :, 1], cmap = 'gray')
        # plt.title('normal[:, :, 1]'), plt.xticks([]), plt.yticks([])
        # plt.show(block=True)
        # plt.imshow(front_isophotes[:, :, 0])
        # plt.title('gradient[:, :, 0]'), plt.xticks([]), plt.yticks([])
        # plt.show(block=True)
        # plt.imshow(front_isophotes[:, :, 1])
        # plt.title('gradient[:, :, 1]'), plt.xticks([]), plt.yticks([])
        # plt.show(block=True)

        # height, width = self.working_image.shape[:2]
        # y, x = np.meshgrid(np.linspace(0, 1, height),  
        #                    np.linspace(0, 1, width))
        # plt.quiver(x, y, front_normals[:, :, 0], front_normals[:, :, 1], color='g')
        # plt.gca().invert_yaxis()
        # plt.show(block=True)

        # plt.quiver(x, y, front_isophotes[:, :, 0], front_isophotes[:, :, 1], color='b')
        # plt.gca().invert_yaxis()
        # plt.show(block=True)

        front = front_isophotes*front_normals
        self.data = np.sqrt(
            front[:, :, 0]**2 + front[:, :, 1]**2
        ) + 0.001 

        # plt.imshow(self.data)
        # plt.title('self.data'), plt.xticks([]), plt.yticks([])
        # plt.show(block=True)

    def _compute_front_isophotes(self):
        # grad, gx, gy = image_gradients(self.working_image)
        grey_image_float = cv2.cvtColor(self.working_image, cv2.COLOR_BGR2GRAY).astype(np.float64)/255.0

        grey_image_float[self.working_mask == 1] = None # Excluding the masked area in the gradient computations
        gradient = np.nan_to_num(np.array(np.gradient(grey_image_float)))
        gy = gradient[0]
        gx = gradient[1]
        grad = np.sqrt(gx**2 + gy**2)

        height, width = self.working_image.shape[:2]
        isophotes = np.zeros([height, width, 2])

        front_points = np.argwhere(self.front == 1)
        for p in front_points:
            patch = self._get_patch(p)
            patch_grad_x = get_roi(gx, patch)
            patch_grad_y = get_roi(gy, patch)            
            patch_gradient_val = get_roi(grad, patch)
            patch_max_pos = np.unravel_index(
                patch_gradient_val.argmax(),
                patch_gradient_val.shape
            )

            isophotes[p[0], p[1], 0] = patch_grad_y[patch_max_pos]
            isophotes[p[0], p[1], 1] = patch_grad_x[patch_max_pos]
        
        # for p in front_points:
        #     isophotes[p[0], p[1], 0] = -gy[p[0], p[1]]
        #     isophotes[p[0], p[1], 1] = gx[p[0], p[1]]
                    
        # plt.imshow(isophotes[:, :, 0], cmap = 'gray')
        # plt.title('isophotes[:, :, 0]'), plt.xticks([]), plt.yticks([])
        # plt.show()
        # plt.imshow(isophotes[:, :, 1], cmap = 'gray')
        # plt.title('isophotes[:, :, 1]'), plt.xticks([]), plt.yticks([])
        # plt.show()

        return isophotes
    
    def _calc_normal_matrix(self):
        x_kernel = np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])
        y_kernel = np.array([[-.25, -.5, -.25], [0, 0, 0], [.25, .5, .25]])

        x_normal = convolve(self.working_mask.astype(float), x_kernel)
        y_normal = convolve(self.working_mask.astype(float), y_kernel)
        normal = np.dstack((x_normal, y_normal))

        height, width = normal.shape[:2]
        norm = np.sqrt(y_normal**2 + x_normal**2) \
                 .reshape(height, width, 1) \
                 .repeat(2, axis=2)
        norm[norm == 0] = 1

        unit_normal = normal/norm
        return unit_normal

    def _calc_gradient_matrix(self):
        # TODO: find a better method to calc the gradient
        height, width = self.working_image.shape[:2]

        grey_image = rgb2gray(self.working_image)
        grey_image[self.working_mask == 1] = None

        gradient = np.nan_to_num(np.array(np.gradient(grey_image)))
        gradient_val = np.sqrt(gradient[0]**2 + gradient[1]**2)
        max_gradient = np.zeros([height, width, 2])

        front_positions = np.argwhere(self.front == 1)
        for point in front_positions:
            patch = self._get_patch(point)
            patch_y_gradient = get_roi(gradient[0], patch)
            patch_x_gradient = get_roi(gradient[1], patch)
            patch_gradient_val = get_roi(gradient_val, patch)

            patch_max_pos = np.unravel_index(
                patch_gradient_val.argmax(),
                patch_gradient_val.shape
            )

            max_gradient[point[0], point[1], 0] = \
                patch_y_gradient[patch_max_pos]
            max_gradient[point[0], point[1], 1] = \
                patch_x_gradient[patch_max_pos]

        return max_gradient

    def _compute_front_normals_2(self):
        [ny, nx] = np.gradient(self.working_mask.astype(float))
        height, width = nx.shape[:2]
        norm = np.sqrt(ny**2 + nx**2)
        norm[norm == 0] = 1
        nx = nx/norm
        ny = ny/norm
        return np.dstack((-ny, nx))
    
    def _compute_front_normals(self):
        x_kernel = np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])
        y_kernel = np.array([[-.25, -.5, -.25], [0, 0, 0], [.25, .5, .25]])

        working_mask_float = self.working_mask.astype(float)
        x_normal = cv2.filter2D(src=working_mask_float, ddepth=-1, kernel=x_kernel)
        y_normal = cv2.filter2D(src=working_mask_float, ddepth=-1, kernel=y_kernel)
        normal = np.dstack((x_normal, y_normal))

        # plt.imshow(normal[:, :, 0], cmap = 'gray')
        # plt.title('normal[:, :, 0]'), plt.xticks([]), plt.yticks([])
        # plt.show()
        # plt.imshow(normal[:, :, 1], cmap = 'gray')
        # plt.title('normal[:, :, 1]'), plt.xticks([]), plt.yticks([])
        # plt.show()

        height, width = normal.shape[:2]
        norm = np.sqrt(y_normal**2 + x_normal**2) \
                 .reshape(height, width, 1) \
                 .repeat(2, axis=2)
        norm[norm == 0] = 1

        unit_normal = normal/norm

        # plt.imshow(unit_normal[:, :, 0], cmap = 'gray')
        # plt.title('unit_normal[:, :, 0]'), plt.xticks([]), plt.yticks([])
        # plt.show()
        # plt.imshow(unit_normal[:, :, 1], cmap = 'gray')
        # plt.title('unit_normal[:, :, 1]'), plt.xticks([]), plt.yticks([])
        # plt.show()

        return unit_normal

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

        target_patch = self._get_patch(target_pixel)
        template = get_roi(self.working_image, target_patch)

        # Convert to another color space?
        if self.search_color_space == SearchColorSpace.BGR:
            working_image = self.working_image
        elif self.search_color_space == SearchColorSpace.HSV:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
            working_image = cv2.cvtColor(self.working_image, cv2.COLOR_BGR2HSV)
        elif self.search_color_space == SearchColorSpace.LAB:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2LAB)
            working_image = cv2.cvtColor(self.working_image, cv2.COLOR_BGR2LAB)        
        
        template_mask = 1-get_roi(self.working_mask, target_patch)
        tm = cv2.matchTemplate(working_image, template, cv2.TM_SQDIFF, None, template_mask.astype(np.uint8)*255)
        struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, (self.patch_size, self.patch_size))
        # dilated_mask = cv2.dilate(self.working_mask.astype(np.uint8)*255, struct_element)
        valid_mask = cv2.filter2D((1-self.working_mask), -1, struct_element, anchor=(0,0)) == ((self.patch_size)*(self.patch_size))
        
        if self.search_original_source_only:
            # Do NOT take texture from the inpainted area
            valid_mask_ext = cv2.filter2D((1-self.mask), -1, struct_element, anchor=(0,0)) == ((self.patch_size)*(self.patch_size))
            valid_mask = valid_mask & valid_mask_ext

        # plt.imshow(~self.mask.astype(np.bool), cmap = 'gray')
        # plt.title('mask'), plt.xticks([]), plt.yticks([])
        # plt.show(block=True)

        # plt.imshow(valid_mask, cmap = 'gray')
        # plt.title('valid_mask'), plt.xticks([]), plt.yticks([])
        # plt.show(block=True)
        
        
        valid_mask = valid_mask[0:tm.shape[0], 0:tm.shape[1]]
        tm[~valid_mask] = np.max(tm)+1

        # mask_non_valid_tm = (dilated_mask[0:tm.shape[0], 0:tm.shape[1]] > 0)
        # plt.imshow(mask_non_valid_tm, cmap = 'gray')
        # plt.title('mask_non_valid_tm'), plt.xticks([]), plt.yticks([])
        # plt.show()
        # tm[mask_non_valid_tm] = np.max(tm)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(tm)

        min_vals_coords = cv2.findNonZero((tm == min_val).astype(np.uint8))
        if len(min_vals_coords) != 1:
            # In case there are more than a single instance of the minimum, we need to select one
            if self.patch_preference == PatchPreference.ANY:
                # Choose any (the first one in the list)
                min_loc = min_vals_coords[0][0]
            elif self.patch_preference == PatchPreference.CLOSEST:
                # Choose the one closer to the current pixel
                distances = np.sqrt((min_vals_coords[:,:,0] - target_pixel[1]) ** 2 + (min_vals_coords[:,:,1] - target_pixel[0]) ** 2)
                nearest_index = np.argmin(distances)
                min_loc = min_vals_coords[nearest_index][0]
            elif self.patch_preference == PatchPreference.RANDOM:
                # Random choice
                ind = random.randrange(len(min_vals_coords))
                min_loc = min_vals_coords[ind][0]
            else:
                raise ValueError("Unknown patch preference")

        # plt.imshow(template, cmap = 'gray')
        # plt.title('template'), plt.xticks([]), plt.yticks([])
        # plt.show()
        # plt.imshow(template_mask, cmap = 'gray')
        # plt.title('template_mask'), plt.xticks([]), plt.yticks([])
        # plt.show()
        # plt.imshow(tm,cmap = 'gray')
        # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        # plt.show(block=True)
        # self.last_tm = tm
        # self.last_template = template
        # self.last_template_mask = template_mask
        
        best_patch = [[min_loc[1], min_loc[1]+self.patch_size-1], [min_loc[0], min_loc[0]+self.patch_size-1]]
        return best_patch
    
    def _update_image(self, target_pixel, source_patch):
        target_patch = self._get_patch(target_pixel)
        pixels_positions = np.argwhere(
            get_roi(
                self.working_mask,
                target_patch
            ) == 1
        ) + [target_patch[0][0], target_patch[1][0]]
        patch_confidence = self.confidence[target_pixel[0], target_pixel[1]]
        for point in pixels_positions:
            self.confidence[point[0], point[1]] = patch_confidence

        mask = get_roi(self.working_mask, target_patch)
        rgb_mask = to_three_channels(mask)
        source_data = get_roi(self.working_image, source_patch)
        target_data = get_roi(self.working_image, target_patch)

        if target_data.shape != source_data.shape:
            # In some cases, the target patch may get over the borders. In such cases, the target_data will be smaller than the source_data (source_data is enforced to be within the image in _find_source_patch)
            source_data = source_data[0:target_data.shape[0], 0:target_data.shape[1]]
        
        new_data = source_data*rgb_mask + target_data*(1-rgb_mask)

        # plt.imshow(new_data, cmap = 'gray')
        # plt.title('new_data'), plt.xticks([]), plt.yticks([])
        # plt.show()

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
   
    def _find_source_patch_2(self, target_pixel):
        target_patch = self._get_patch(target_pixel)
        height, width = self.working_image.shape[:2]
        patch_height, patch_width = roi_shape(target_patch)

        best_match = None
        best_match_difference = 0

        # lab_image = rgb2lab(self.working_image)
        lab_image = self.working_image

        for y in range(height - patch_height + 1):
            for x in range(width - patch_width + 1):
                source_patch = [
                    [y, y + patch_height-1],
                    [x, x + patch_width-1]
                ]
                if get_roi(self.working_mask, source_patch) \
                   .sum() != 0:
                    continue

                difference = self._calc_patch_difference(
                    lab_image,
                    target_patch,
                    source_patch
                )

                if best_match is None or difference < best_match_difference:
                    best_match = source_patch
                    best_match_difference = difference
        return best_match

    def _calc_patch_difference(self, image, target_patch, source_patch):
        mask = 1 - get_roi(self.working_mask, target_patch)
        rgb_mask = to_three_channels(mask)
        target_data = get_roi(
            image,
            target_patch
        ) * rgb_mask
        source_data = get_roi(
            image,
            source_patch
        ) * rgb_mask
        squared_distance = ((target_data - source_data)**2).sum()
        euclidean_distance = np.sqrt(
            (target_patch[0][0] - source_patch[0][0])**2 +
            (target_patch[1][0] - source_patch[1][0])**2
        )  # tie-breaker factor
        return squared_distance + euclidean_distance


    def _finished(self):        
        remaining = np.count_nonzero(self.working_mask)
        # if self.show_progress_bar:
            # self.progress.update(self.task, completed=remaining/self.num_pixels_to_fill*100)
            # print('%d of %d completed' % (self.num_pixels-remaining, self.num_pixels))
        return remaining == 0
    
    def _compute_gradients_ignoring_mask(self):
        working_image_float = self.working_image.astype(np.float64)/255.0
        height, width = working_image_float.shape[:2]
        inds = [*range(1, height)]
        inds.append(height-1)
        gy = working_image_float[inds, :, :].astype(np.float64)
        gy = gy - working_image_float
        inds = [*range(1, width)]
        inds.append(width-1)
        gx = working_image_float[:, inds, :].astype(np.float64)
        gx = gx - working_image_float
        front_positions = np.argwhere(self.front == 1)        
        for point in front_positions:
            r = point[0] # Row 
            c = point[1] # Column

            # Ys
            if r+1 < height and self.working_mask[r+1, c] == 0:
                pass # Already computed above (gy = gy - self.working_image)
            elif r-1 >= 0 and self.working_mask[r-1,c] == 0:
                gy[r, c] = working_image_float[r-1, c, :] - working_image_float[r, c, :]
            else:
                gy[r, c] = 0

            # Xs
            if c+1 < width and self.working_mask[r, c+1] == 0:
                pass # Already computed above (gx = gx - self.working_image)
            elif c-1 >= 0 and self.working_mask[r, c-1] == 0:
                gx[r, c] = working_image_float[r, c-1, :] - working_image_float[r, c, :]
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
        # return [gx, gy]
        # return np.dstack((gx, gy))
        return np.dstack((gx, gy))