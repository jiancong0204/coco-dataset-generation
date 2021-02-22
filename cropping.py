import cv2
import numpy as np
import json
import math
import random


class Cropping:

    def __init__(self):
        self.__json_file = None
        self.__mask = None
        self.__masked_image = None
        self.__roi_corners = None
        self.__ann_file = None
        self.__json_data = None

    def set_ann_file(self, ann_file):
        self.__ann_file = ann_file
        self.__json_file = open(ann_file)
        self.__json_data = json.load(self.__json_file)

    def generate_mask(self, roi_corners, img):
        channels = img.shape[2]
        mask = np.zeros(img.shape, dtype=np.uint8)

        channel_count = channels
        ignore_mask_color = (255,) * channel_count

        cv2.fillPoly(mask, roi_corners, ignore_mask_color)
        return mask

    def image_cropping(self):

        # read json file
        json_data = self.__json_data

        image_name = self.__ann_file
        image_name = image_name.replace("json", "jpg")

        # read image file
        img = cv2.imread(image_name)

        # determine region of interest
        self.__roi_corners = np.array(json_data["shapes"][0]["points"], dtype=np.int32)
        size = int(self.__roi_corners.size / 2)
        self.__roi_corners = self.__roi_corners.reshape((1, size, 2))

        self.__mask = self.generate_mask(self.__roi_corners, img)

        self.__masked_image = cv2.bitwise_and(img, self.__mask)

    def rotate(self):
        rotation_angle = random.uniform(0, 360)  # Degree
        height = self.__masked_image.shape[0]
        width = self.__masked_image.shape[1]
        rotation_mat = cv2.getRotationMatrix2D((width // 2, height // 2), rotation_angle, 1)
        new_height = int(width * math.fabs(math.sin(math.radians(rotation_angle))) + height * math.fabs(
            math.cos(math.radians(rotation_angle))))
        new_width = int(height * math.fabs(math.sin(math.radians(rotation_angle))) + width * math.fabs(
            math.cos(math.radians(rotation_angle))))
        rotation_mat[0, 2] += (new_width - width) // 2
        rotation_mat[1, 2] += (new_height - height) // 2
        self.__masked_image = cv2.warpAffine(self.__masked_image, rotation_mat, (new_width, new_height),
                                             borderValue=(0, 0, 0))
        self.__roi_corners = self.__roi_corners[0]
        self.__roi_corners = np.column_stack((self.__roi_corners, np.ones(self.__roi_corners.shape[0])))
        self.__roi_corners = np.dot(rotation_mat, self.__roi_corners.T).T

    def display_masked_image(self):
        # show the mask
        cv2.imshow("Masked Image", self.__masked_image)
        cv2.waitKey(0)

    def get_masked_image(self):
        return self.__masked_image

    def get_roi_corners(self):
        return self.__roi_corners

    def get_mask(self):
        return self.__mask

    def get_rot_corners(self):
        return self.__roi_corners

    def get_json_data(self):
        return self.__json_data


if __name__ == "__main__":
    c = Cropping()
    c.set_ann_file('label/0001.json')
    c.image_cropping()
    c.rotate()
    c.display_masked_image()
