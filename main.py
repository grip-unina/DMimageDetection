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

import torch  # >=1.6.0
import os
import pandas
import numpy as np
import tqdm
import glob
import sys
from PIL import Image
import torchvision.transforms as transforms
import argparse

from normalization import CenterCropNoPad, get_list_norm
from normalization2 import PaddingWarp
from get_method_here import get_method_here, def_model


def runnig_tests(data_path, output_dir, weights_dir, csv_file):
    DATA_PATH = data_path

    print("CURRENT OUT FOLDER")
    print(output_dir)
    datasets = {os.path.basename(os.path.dirname(_)): _ for _ in glob.glob(DATA_PATH+"*/")}
    csvfilename = csv_file
    outroot = output_dir

    if not os.path.exists(outroot):
        os.makedirs(outroot)

    print(len(datasets), datasets.keys())

    # NOTE: Substitute the device with 'cpu' if gpu acceleration is not required

    device = 'cuda:0'  # in ['cpu', 'cuda:0']
    batch_size = 1

    ### list of models

    models_list = {
        'Grag2021_progan': 'Grag2021_progan',
        'Grag2021_latent': 'Grag2021_latent'
    }

    models_dict = dict()
    transform_dict = dict()
    for model_name in models_list:

        _, model_path, arch, norm_type, patch_size = get_method_here(models_list[model_name], weights_path=weights_dir)

        model = def_model(arch, model_path, localize=False)
        model = model.to(device).eval()

        transform = list()

        if patch_size is not None:
            if isinstance(patch_size, tuple):
                print('input resize:', patch_size)
                transform.append(transforms.Resize(*patch_size))
                transform_key = 'res%d_%s' % (patch_size[0], norm_type)
            else:
                if patch_size > 0:
                    print('input crop:', patch_size)
                    transform.append(CenterCropNoPad(patch_size))
                    transform_key = 'crop%d_%s' % (patch_size, norm_type)
                else:
                    print('input crop pad:', patch_size)
                    transform.append(CenterCropNoPad(-patch_size))
                    transform.append(PaddingWarp(-patch_size))
                    transform_key = 'cropp%d_%s' % (-patch_size, norm_type)
        else:
            transform_key = 'none_%s' % norm_type

        transform = transform + get_list_norm(norm_type)
        transform = transforms.Compose(transform)
        transform_dict[transform_key] = transform
        models_dict[model_name] = (transform_key, model)

    print(list(transform_dict.keys()))
    print(list(models_dict.keys()))

    ### test

    with torch.no_grad():
        table = pandas.read_csv(csvfilename)[['src', ]]
        for dataset in datasets:
            outdir = os.path.join(outroot, dataset)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            print(outdir)
            output_csv = outdir + '/'+dataset+'.csv'
            rootdataset = DATA_PATH
            if os.path.isfile(output_csv):
                table_old = table['src']
                table = pandas.read_csv(output_csv)
                assert len(table_old) == len(table['src'])
                assert all([a == b for a, b in zip(table_old, table['src'])])
                del table_old
                do_models = [_ for _ in models_dict.keys() if _ not in table]
                print('ok')
            else:
                do_models = list(models_dict.keys())
            do_transforms = set([models_dict[_][0] for _ in do_models])
            print(do_models)
            print(do_transforms)

            if len(do_models) == 0:
                continue

            batch_img = {k: list() for k in transform_dict}
            batch_id = list()
            table_to_save = table[table['src'].str.contains(dataset+"/")].copy()
            print(dataset, csvfilename, flush=True)
            print(dataset, 'Number of images:', len(table_to_save))
            for index, dat in tqdm.tqdm(table_to_save.iterrows(), total=len(table_to_save)):
                if dataset in dat['src'].split('/')[0]:

                    filename = os.path.join(rootdataset, dat['src'])
                    if not os.path.isfile(filename):
                        filename = filename[:-4] + '.png'
                    if not os.path.isfile(filename):
                        filename = filename[:-4] + '.tif'

                    for k in transform_dict:
                        batch_img[k].append(transform_dict[k](Image.open(filename).convert('RGB')))
                    batch_id.append(index)

                    if len(batch_id) >= batch_size:
                        for k in do_transforms:
                            batch_img[k] = torch.stack(batch_img[k], 0)
                        for model_name in do_models:
                            out_tens = models_dict[model_name][1](batch_img[models_dict[model_name][0]].clone().to(device)).cpu().numpy()

                            if out_tens.shape[1] == 1:
                                out_tens = out_tens[:, 0]
                            elif out_tens.shape[1] == 2:
                                out_tens = out_tens[:, 1] - out_tens[:, 0]
                            else:
                                assert False
                            if len(out_tens.shape) > 1:
                                logit1 = np.mean(out_tens, (1, 2))
                            else:
                                logit1 = out_tens

                            for ii, logit in zip(batch_id, logit1):
                                table_to_save.loc[ii, model_name] = logit

                        batch_img = {k: list() for k in transform_dict}
                        batch_id = list()

                if len(batch_id) > 0:
                    for k in transform_dict:
                        batch_img[k] = torch.stack(batch_img[k], 0)
                    for model_name in models_dict:
                        out_tens = models_dict[model_name][1](batch_img[models_dict[model_name][0]].clone().to(device)).cpu().numpy()

                        if out_tens.shape[1] == 1:
                            out_tens = out_tens[:, 0]
                        elif out_tens.shape[1] == 2:
                            out_tens = out_tens[:, 1] - out_tens[:, 0]
                        else:
                            assert False
                        if len(out_tens.shape) > 1:
                            logit1 = np.mean(out_tens, (1, 2))
                        else:
                            logit1 = out_tens
                        for ii, logit in zip(batch_id, logit1):
                            table_to_save.loc[ii, model_name] = logit
                    batch_img = {k: list() for k in transform_dict}
                    batch_id = list()
            if "real" in dataset:
                table_to_save.insert(1, 'label', False)
            else:
                table_to_save.insert(1, 'label', True)
            table_to_save.to_csv(output_csv, index=False)  # save the results as csv file


def main():
    print("Running the Tests")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="The path to the images of the testset on which the operations have been pre applied with the provided code", default="./TestSetCSV/")
    parser.add_argument("--out_dir", type=str, help="The Path where the csv containing the outputs of the networks should be saved", default="./results_tst")
    parser.add_argument("--weights_dir", type=str, help="The path to the weights of the networks", default="./weights")
    parser.add_argument("--csv_file", type=str, help="The path to the csv file", default="./TestSetCSV/operations.csv")
    args = vars(parser.parse_args())
    runnig_tests(args['data_dir'], args['out_dir'], args['weights_dir'], args['csv_file'])


main()
