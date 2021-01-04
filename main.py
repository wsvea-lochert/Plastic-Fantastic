# Code inspired by: https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
# Written by William Svea-Lochert
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from numpy import expand_dims
from tqdm import tqdm


class ImageAugment:
    #  Constructor
    def __init__(self, data_dir, output_path):
        self.data_dir = data_dir
        self.output_path = output_path

    #  Open image from directory
    def __open_img(self, file):
        img = load_img(self.data_dir + '/' + file)
        data = img_to_array(img)
        return expand_dims(data, 0)

    #  Saving some processing and saving the current image
    def __save_img(self, datagen, samples, count):
        it = datagen.flow(samples, batch_size=1)
        batch = it.next()
        image = batch[0].astype('uint8')
        im = Image.fromarray(image)
        im.save(self.output_path + '/img' + str(count) + '.jpg')

    #  Horizontal image shift <->
    def horizontal_img_shift(self, shift_amount):  # shift_amount example: [-250, 250]
        counter = 0
        for file in tqdm(os.listdir(self.data_dir)):
            samples = self.__open_img(file)
            datagen = ImageDataGenerator(width_shift_range=shift_amount)
            self.__save_img(datagen=datagen, samples=samples, count=counter)
            counter = counter + 1

    #  Vertically image shift ^
    def vertical_img_shift(self, shift_amount):  # Shift amount example: 0.5
        counter = 0
        for file in tqdm(os.listdir(self.data_dir)):
            samples = self.__open_img(file)
            datagen = ImageDataGenerator(height_shift_range=shift_amount)
            self.__save_img(datagen=datagen, samples=samples, count=counter)
            counter = counter + 1

    #  Horizontal flip image
    def horizontal_img_flip(self):
        counter = 0
        for file in tqdm(os.listdir(self.data_dir)):
            samples = self.__open_img(file)
            datagen = ImageDataGenerator(horizontal_flip=True)
            self.__save_img(datagen=datagen, samples=samples, count=counter)
            counter = counter + 1

    #  vertical flip image
    def vertical_img_flip(self):
        counter = 0
        for file in tqdm(os.listdir(self.data_dir)):
            samples = self.__open_img(file)
            datagen = ImageDataGenerator(vertical_flip=True)
            self.__save_img(datagen=datagen, samples=samples, count=counter)
            counter = counter + 1

    #  random image rotation
    def random_img_rotation(self, rot_range):  # Range example: 90
        counter = 0
        for file in tqdm(os.listdir(self.data_dir)):
            samples = self.__open_img(file)
            datagen = ImageDataGenerator(rotation_range=rot_range)
            self.__save_img(datagen=datagen, samples=samples, count=counter)
            counter = counter + 1

    #  Random image brightness
    def random_img_brightness(self, b_range):  # b_range example: [0.2, 1.0]
        counter = 0
        for file in tqdm(os.listdir(self.data_dir)):
            samples = self.__open_img(file)
            datagen = ImageDataGenerator(brightness_range=b_range)
            self.__save_img(datagen=datagen, samples=samples, count=counter)
            counter = counter + 1

    #  random zoom image
    def random_img_zoom(self, z_range):  # z_range example: [0.5, 1.0]
        counter = 0
        for file in tqdm(os.listdir(self.data_dir)):
            samples = self.__open_img(file)
            datagen = ImageDataGenerator(zoom_range=z_range)
            self.__save_img(datagen=datagen, samples=samples, count=counter)
            counter = counter + 1
