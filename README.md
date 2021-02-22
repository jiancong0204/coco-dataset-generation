# coco-dataset-generation

## Installation

### Environment

- labelme
- numpy

### Instruction

- Put your background images in the directory ```/background```
- Put your labeled data (both .json file and .jpg file) in ```/label```. Note that the format follows that of lebelme.
- Run ```background.py``` to generate new data
- Run ```labelme2coco.py``` to convert lebelme into coco

### Example
```shell
$ python background.py
# Dataset in labelme format will be saved in the directory '''/train'''

$ python labelme2coco.py train/ coco/ --labels labels.txt --ann train
# Dataset in coco format can be found in the directory '''coco'''
```
