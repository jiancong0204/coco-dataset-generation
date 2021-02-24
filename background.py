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

    def cropping(self, ann_file):
        self.c.set_ann_file(ann_file)
        self.json_label = self.c.get_json_data()
        self.c.image_cropping()
        self.c.rotate()

    def read_bg(self, bg_file):
        self.__bg = cv2.imread(bg_file)

    def generate_res(self):
        masked_img = self.c.get_masked_image()

        # random zoom
        zoom_coff = random.uniform(2, 6)

        height = int(masked_img.shape[0] / zoom_coff)
        width = int(masked_img.shape[1] / zoom_coff)
        masked_img = cv2.resize(masked_img, (width, height), interpolation=cv2.INTER_CUBIC)

        self.__res_img = self.__bg
        self.__res_img = cv2.resize(self.__res_img, (1200, 900), interpolation=cv2.INTER_CUBIC)
        new_height = self.__res_img.shape[0]
        new_width = self.__res_img.shape[1]
        translation_x = random.randint(0, new_height - height)
        translation_y = random.randint(0, new_width - width)
        for x in range(height):
            for y in range(width):
                if x + translation_x >= new_height or y + translation_y >= new_width:
                    continue
                pixel = masked_img[x, y]
                if pixel.all() != 0:
                    self.__res_img[x + translation_x, y + translation_y] = pixel

        roi_corners = self.c.get_roi_corners()
        for i in range(len(roi_corners)):
            real_y = roi_corners[i][0] // zoom_coff + translation_y
            real_x = roi_corners[i][1] // zoom_coff + translation_x
            if real_x <= new_height and real_y <= new_width:
                roi_corners[i][0] = real_y
                roi_corners[i][1] = real_x
            else:
                roi_corners[i][0] = new_width
                roi_corners[i][1] = new_height

        self.json_label["shapes"][0]["points"] = roi_corners.tolist()
        self.json_label["imageHeight"] = new_height
        self.json_label["imageWidth"] = new_width


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
        bg.cropping('label/0001.json')
        name = path + f[n]
        bg.read_bg(name)
        bg.generate_res()
        # bg.display_res()
        res = bg.get_res_img()
        save_path = 'train/' + f[n]
        cv2.imwrite(save_path, res)

        save_path = save_path.replace('jpg', 'json')
        bg.json_label["imagePath"] = f[n]
        b = json.dumps(bg.json_label)
        f2 = open(save_path, 'w')
        f2.write(b)
        f2.close()
        n += 1
    print('Done.')
