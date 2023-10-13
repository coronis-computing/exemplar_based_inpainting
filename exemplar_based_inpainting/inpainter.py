import cv2
import numpy as np


class Inpainter():
    def __init__(self, patch_size=9, plot_progress=False):
        """
        Inpainter Constructor. 
        
        Just sets the desired inpainting patch size and wether to plot progress (but they can be overriden before calling 'inpaint') 
        """
        self.patch_size = patch_size
        self.half_patch_size = (self.patch_size-1)//2
        self.plot_progress = plot_progress

    @property
    def patch_size(self):
        return self.patch_size
    
    @patch_size.setter
    def patch_size(self, var):
        self.patch_size = var
        self.half_patch_size = (self.patch_size-1)//2
        
    def inpaint(self, image, mask):        
        # Initialization
        if image.shape[:2] != mask.shape:
            raise ValueError("The input image and mask must be of the same size.")        
        self.initialize(image, mask)

        # Inpainting
        while True:
            self._find_front()
            if self.plot_progress:
                self._plot_current_state()
            self._update_priority()

    def initialize(self, image, mask):
        """
        Initializes the inpainting problem and its required intermediate variables

        The confidence is initially the inverse of the mask, that is, the
        target region is 0 and source region is 1.

        The data starts with zero for all pixels.

        The working image and working mask start as copies of the original
        image and mask.
        """
        self.image = image.astype('uint8')
        self.mask = mask.astype('uint8')

        # Non initialized attributes
        self.working_image = np.copy(self.image)
        self.working_mask = np.copy(self.mask)
        self.front = None
        self.confidence = (1 - self.mask).astype(float)
        self.data = np.zeros(self.image.shape[:2])
        self.priority = None

    def _find_front(self):
        self.front = cv2.Laplacian(self.mask, -1)

    def _plot_current_state(self):
        cv2.namedWindow("Inpainting process", cv2.WINDOW_GUI_NORMAL)
        cv2.imshow("Inpainting process", self.image)
        cv2.waitKey()

    def _update_priority(self):
        self._update_confidence()
        self._update_data()
        self.priority = self.confidence * self.data * self.front

    def _update_confidence(self):
        new_confidence = np.copy(self.confidence)
        front_positions = np.argwhere(self.front == 1)
        for point in front_positions:
            patch = self._get_patch(point)
            new_confidence[point[0], point[1]] = sum(sum(
                self._patch_data(self.confidence, patch)
            ))/self._patch_area(patch)

        self.confidence = new_confidence

    def _update_data(self):
        # https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/
        # compute gradients along the x and y axis, respectively
        gray = cv2.cvtColor(self.working_image, cv2.COLOR_BGR2GRAY)

        gX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        gY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        
        # compute the gradient magnitude and orientation
        magnitude = np.sqrt((gX ** 2) + (gY ** 2))
        orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180

        normal_gradient = magnitude*orientation
        self.data = np.sqrt(
            normal_gradient[:, :, 0]**2 + normal_gradient[:, :, 1]**2
        ) + 0.00001  

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

    @staticmethod
    def _get_patch_data(image, patch):
        return image[
            patch[0][0]:patch[0][1]+1,
            patch[1][0]:patch[1][1]+1
        ]