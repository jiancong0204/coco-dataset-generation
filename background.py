import cropping
import cv2
import os
import random
import json
import numpy as np


class Background:

    def __init__(self):
        self.__bg = None
        self.__res_img = None
        self.c = cropping.Cropping()
        self.roi_corners = None
        self.json_label = None
        self.__rotation_angle = None
        self.__zoom_coff = None
        self.__translation_x = None
        self.__translation_y = None
        self.__mask_num = None
        self.__mask_img = None
        self.__mask_height = None
        self.__mask_width = None
        self.__mask_resize_coff = None
        self.__json_len = None
        self.__pre_roi_polygons = []
        self.res_height = 900
        self.res_width = 1200

    def prepare_cropping(self, ann_file):
        self.__pre_roi_polygons = []
        self.c.set_ann_file(ann_file)
        self.json_label = self.c.get_json_data()
        self.__json_len = len(self.json_label["shapes"])

    def cropping(self, mask_num):
        self.__mask_num = mask_num

        self.c.image_cropping(mask_num)
        self.c.rotate(self.__rotation_angle)
        self.__mask_img = self.c.get_masked_image()
        self.__mask_height = self.__mask_img.shape[0]
        self.__mask_width = self.__mask_img.shape[1]
        zoom_coff_x = self.__mask_width / self.res_width
        zoom_coff_y = self.__mask_height / self.res_height
        self.__mask_resize_coff = max(zoom_coff_x, zoom_coff_y)
        self.__mask_height = int(self.__mask_img.shape[0] / self.__mask_resize_coff)
        self.__mask_width = int(self.__mask_img.shape[1] / self.__mask_resize_coff)
        self.__mask_img = cv2.resize(self.__mask_img, (self.__mask_width, self.__mask_height), interpolation=cv2.INTER_CUBIC)

    def read_bg(self, bg_file):
        self.__bg = cv2.imread(bg_file)
        self.__res_img = self.__bg
        self.__res_img = cv2.resize(self.__res_img, (self.res_width, self.res_height), interpolation=cv2.INTER_CUBIC)

    def generate_rnd_coffs(self):
        self.__rotation_angle = random.uniform(0, 360)
        self.__zoom_coff = random.uniform(2, 6)
        self.__translation_x = random.randint(0, 100)
        self.__translation_y = random.randint(0, 100)

    @staticmethod
    def is_ray_intersects_segment(ray_point, start_point, end_point) -> bool:
        if start_point[1] == end_point[1]:
            return False
        if start_point[1] > ray_point[1] and end_point[1] > ray_point[1]:
            return False
        if start_point[1] < ray_point[1] and end_point[1] < ray_point[1]:
            return False
        if start_point[1] == ray_point[1] and end_point[1] > ray_point[1]:
            return True
        if end_point[1] == ray_point[1] and start_point[1] > ray_point[1]:
            return True
        if start_point[0] < ray_point[0] and end_point[1] < ray_point[1]:
            return False

        intersection_x = end_point[0] - (end_point[0] - start_point[0]) * (end_point[1] - ray_point[1]) / (end_point[1] - start_point[1])  # 求交
        if intersection_x < ray_point[0]:
            return False
        return True

    @property
    def generate_res(self) -> bool:
        masked_img = self.__mask_img

        # random zoom
        zoom_coff = self.__zoom_coff * self.__mask_resize_coff

        height = int(masked_img.shape[0] / self.__zoom_coff)
        width = int(masked_img.shape[1] / self.__zoom_coff)
        masked_img = cv2.resize(masked_img, (width, height), interpolation=cv2.INTER_CUBIC)

        new_height = self.__res_img.shape[0]
        new_width = self.__res_img.shape[1]

        translation_y = (new_height - height) * self.__translation_y / 100
        translation_x = (new_width - width) * self.__translation_x / 100
        translation_y = int(translation_y)
        translation_x = int(translation_x)

        roi_corners = self.c.get_roi_corners()

        for i in range(len(roi_corners)):
            real_x = roi_corners[i][0] // zoom_coff + translation_x
            real_y = roi_corners[i][1] // zoom_coff + translation_y

            if real_x <= new_width and real_y <= new_height:
                roi_corners[i][0] = real_x
                roi_corners[i][1] = real_y
            else:
                roi_corners[i][0] = new_width
                roi_corners[i][1] = new_height

        if len(self.__pre_roi_polygons) != 0:

            for curr_point in roi_corners:
                for pre_roi_corners in self.__pre_roi_polygons:
                    cnt_intersection = 0
                    for i in range(len(pre_roi_corners)):
                        start = (float(pre_roi_corners[i][0]), float(pre_roi_corners[i][1]))
                        next_i = (i + 1) % len(pre_roi_corners)
                        end = (float(pre_roi_corners[next_i][0]), float(pre_roi_corners[next_i][1]))
                        if self.is_ray_intersects_segment(curr_point, start, end):
                            cnt_intersection += 1
                    if cnt_intersection % 2 == 1:
                        return False

            for pre_roi_corners in self.__pre_roi_polygons:
                for curr_point in pre_roi_corners:
                    cnt_intersection = 0
                    for i in range(len(roi_corners)):
                        start = (float(roi_corners[i][0]), float(roi_corners[i][1]))
                        next_i = (i + 1) % len(roi_corners)
                        end = (float(roi_corners[next_i][0]), float(roi_corners[next_i][1]))
                        if self.is_ray_intersects_segment(curr_point, start, end):
                            cnt_intersection += 1

                    if cnt_intersection % 2 == 1:
                        return False

        self.__pre_roi_polygons.append(roi_corners)

        for y in range(height):
            for x in range(width):
                if y + translation_y >= new_height or x + translation_x >= new_width:
                    continue
                pixel = masked_img[y, x]
                if pixel.all() != 0:
                    self.__res_img[y + translation_y, x + translation_x] = pixel

        self.json_label["shapes"][self.__mask_num]["points"] = roi_corners.tolist()
        return True

    def get_res_img(self):
        return self.__res_img

    def display_res(self):
        cv2.imshow("Image", self.__res_img)
        cv2.waitKey(0)


if __name__ == "__main__":
    bg = Background()
    path = 'background/'
    f = os.listdir(path)
    n = 0
    print('Running...')
    for i in f:
        name = path + f[n]
        bg.read_bg(name)
        bg.prepare_cropping('label/0004.json')
        length = len(bg.json_label["shapes"])
        j = 0
        cnt = 0
        while j < length:
            rnd = random.uniform(0, 1)
            if rnd > 0.7:
                bg.json_label["shapes"].pop(j)
                length -= 1
                continue
            bg.generate_rnd_coffs()
            bg.cropping(j)
            if not bg.generate_res:
                bg.json_label["shapes"].pop(j)
                length -= 1
                continue
            j += 1
            cnt += 1

        # bg.display_res()
        res = bg.get_res_img()
        save_path = 'train/' + f[n]
        cv2.imwrite(save_path, res)
        save_path = save_path.replace('jpg', 'json')
        bg.json_label["imagePath"] = f[n]
        bg.json_label["imageHeight"] = bg.res_height
        bg.json_label["imageWidth"] = bg.res_width
        b = json.dumps(bg.json_label)
        f2 = open(save_path, 'w')
        f2.write(b)
        f2.close()
        n += 1

    print('Done.')
