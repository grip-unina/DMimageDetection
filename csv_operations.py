#
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#

# This script applies random operations to all images in an input directory and saves the results in an output directory.
# The images are cropped and resized to 200x200 pixels and then compressed using JPEG at a random quality level.
#


import os
from PIL import Image
import tqdm
import shutil
import glob
from random import Random
import csv
import argparse

output_size = 200
cropsize_min = 160
cropsize_max = 2048
cropsize_ratio = (5, 8)
qf_range = (65, 100)


def check_img(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'))


def csv_operations(input_dir, output_dir, csv_file):
    print('CSV Operations from ', input_dir, 'to', output_dir, flush=True)
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)  # remove existing output directory
    os.makedirs(output_dir)  # create output directory

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(csv_file, 'r') as f:
        print("Reading parameters from: ", csv_file)
        data = list(csv.reader(f))
        # Start from one to remove the header
        for i in tqdm.tqdm(range(1, len(data))):
            row = data[i]
            # open image
            img = Image.open(os.path.join(input_dir, row[0])).convert('RGB')

            # read the size of the crop

            cropsize = int(row[1])

            # select the type of interpolation
            interp = Image.ANTIALIAS if cropsize > output_size else Image.CUBIC

            # read the position of the crop
            x1 = int(row[2])
            y1 = int(row[3])

            # read the jpeg quality factor
            qf = int(row[4])

            # make cropping
            img = img.crop((x1, y1, x1+cropsize, y1+cropsize))
            assert img.size[0] == cropsize
            assert img.size[1] == cropsize

            # make resizing
            img = img.resize((output_size, output_size), interp)
            assert img.size[0] == output_size
            assert img.size[1] == output_size

            path_split = row[0].split('/')
            folder = path_split[0]
            if not os.path.exists(os.path.join(output_dir, folder)):
                os.mkdir(os.path.join(output_dir, folder))
            dst = os.path.join(os.path.join(output_dir, folder), path_split[1])

            # Destination of the file
            # make jpeg compression
            img.save(dst, "JPEG", quality=qf)
    shutil.copy(csv_file, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="The path to the images of the testset on which to apply the operations specified in the csv", default="./TestSet/")
    parser.add_argument("--out_dir", type=str, help="The Path where the modified should be saved", default="./TestSetCSV")
    parser.add_argument("--csv_file", type=str, help="The path to the csv file containing the operations", default="./TestSet/operations.csv")
    args = vars(parser.parse_args())
    csv_operations(args['data_dir'], args['out_dir'], args['csv_file'])
