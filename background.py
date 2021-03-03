import cropping
import cv2
import os
import random
import json


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
        self.res_height = 900
        self.res_width = 1200

    def prepare_cropping(self, ann_file):
        self.c.set_ann_file(ann_file)
        self.json_label = self.c.get_json_data()

    def cropping(self, mask_num):
        self.__mask_num = mask_num

        self.c.image_cropping(mask_num)
        self.c.rotate(self.__rotation_angle)

    def read_bg(self, bg_file):
        self.__bg = cv2.imread(bg_file)
        self.__res_img = self.__bg
        self.__res_img = cv2.resize(self.__res_img, (self.res_width, self.res_height), interpolation=cv2.INTER_CUBIC)

    def generate_rnd_coffs(self):
        self.__rotation_angle = random.uniform(0, 360)
        self.__zoom_coff = random.uniform(1, 1.1)
        self.__translation_x = random.randint(0, 100)
        self.__translation_y = random.randint(0, 100)

    def generate_res(self):
        masked_img = self.c.get_masked_image()
        height = masked_img.shape[0]
        width = masked_img.shape[1]
        zoom_coff_x = width / self.res_width
        zoom_coff_y = height / self.res_height
        # random zoom
        zoom_coff = self.__zoom_coff * max(zoom_coff_x, zoom_coff_y)

        height = int(masked_img.shape[0] / zoom_coff)
        width = int(masked_img.shape[1] / zoom_coff)
        masked_img = cv2.resize(masked_img, (width, height), interpolation=cv2.INTER_CUBIC)

        new_height = self.__res_img.shape[0]
        new_width = self.__res_img.shape[1]
        translation_y = (new_height - height) * self.__translation_y / 100
        translation_x = (new_width - width) * self.__translation_x / 100
        translation_y = int(translation_y)
        translation_x = int(translation_x)

        for y in range(height):
            for x in range(width):
                if y + translation_y >= new_height or x + translation_x >= new_width:
                    continue
                pixel = masked_img[y, x]
                if pixel.all() != 0:
                    self.__res_img[y + translation_y, x + translation_x] = pixel

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

        self.json_label["shapes"][self.__mask_num]["points"] = roi_corners.tolist()

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
        bg.generate_rnd_coffs()
        name = path + f[n]
        bg.read_bg(name)
        bg.prepare_cropping('label/0004.json')
        for j in range(len(bg.json_label["shapes"])):
            bg.cropping(j)
            bg.generate_res()
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
        break
    print('Done.')
