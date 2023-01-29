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

import os
import numpy as np
import pandas
from sklearn import metrics
import dmetrics
import argparse


def calculate_metrics(csv_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    dist_metrics = {
        'acc': lambda y_label, y_pred: dmetrics.balanced_accuracy_score(y_label, y_pred > 0),
        'auc': lambda y_label, y_pred: dmetrics.roc_auc_score(y_label, y_pred),
        'pd10': lambda y_label, y_pred: dmetrics.pd_at_far(y_label, y_pred, 0.10),
        'eer': lambda y_label, y_pred: dmetrics.calculate_eer(y_label, y_pred),
        'pd05': lambda y_label, y_pred: dmetrics.pd_at_far(y_label, y_pred, 0.05),
        'pd01': lambda y_label, y_pred: dmetrics.pd_at_far(y_label, y_pred, 0.01),
        'macc': lambda y_label, y_pred: dmetrics.macc(y_label, y_pred),
        'count1': lambda y_label, y_pred: np.sum(y_label == 1),
        'count0': lambda y_label, y_pred: np.sum(y_label == 0),
    }

    db0s = ['real_coco_valid', 'real_imagenet_val', 'real_ucid', ]
    db1d = {
        'ProGAN': ['progan_lsun'],
        'StyleGAN2': ['stylegan2_ffhq_256x256', 'stylegan2_lsundog_256x256', 'stylegan2_afhqv2_512x512', 'stylegan2_ffhq_1024x1024', ],
        'StyleGAN3': ['stylegan3_r_ffhqu_256x256', 'stylegan3_t_ffhqu_256x256', 'stylegan3_r_afhqv2_512x512', 'stylegan3_r_ffhqu_1024x1024', 'stylegan3_t_afhqv2_512x512', 'stylegan3_t_ffhqu_1024x1024', ],
        'BigGAN': ['biggan_256', 'biggan_512', ],
        'EG3d': ['eg3d', ],
        'Taming Tran.': ['taming-transformers_class2image_ImageNet', 'taming-transformers_noise2image_FFHQ', 'taming-transformers_segm2image_valid'],
        'DALL·E Mini': ['dalle-mini_valid', ],
        'DALL·E 2': ['dalle_2', ],
        'GLIDE': ['glide_text2img_valid', ],
        'Latent Diff.': ['latent-diffusion_class2image_ImageNet', 'latent-diffusion_noise2image_FFHQ',
                         'latent-diffusion_noise2image_LSUNbedrooms', 'latent-diffusion_noise2image_LSUNchurches', 'latent-diffusion_text2img_valid'],
        'Stable Diff.': ['stable_diffusion_256', ],
        'ADM': ['guided-diffusion_class2image_ImageNet',  'guided-diffusion_noise2image_LSUNbedrooms',
                'guided-diffusion_noise2image_LSUNcats', 'guided-diffusion_noise2image_LSUNhorses'],
    }

    # NOTE: all the methodologies to use to evaluate the metrics for
    mm = ['Grag2021_progan', 'Grag2021_latent']
    tab_metrics_p1 = pandas.DataFrame(index=db1d.keys(), columns=mm)
    tab_metrics_p2 = pandas.DataFrame(index=db1d.keys(), columns=mm)

    tab_rs = []
    for db0 in db0s:
        tab_rs.append(pandas.read_csv(os.path.join(os.path.join(csv_path, db0), db0 + ".csv"), index_col='src'))
    tab_r = pandas.concat(tab_rs)
    for db1 in db1d:
        tab_f = []
        for folder in db1d[db1]:
            tab_f.append(pandas.read_csv(os.path.join(os.path.join(csv_path, folder), folder + ".csv")))
        if len(tab_f) > 1:
            tab_f = pandas.concat(tab_f)
        else:
            tab_f = tab_f[0]
        tab_all = []
        tab_all.append(tab_f)
        tab_all.append(tab_r)
        tab_both = pandas.concat(tab_all)
        label = tab_both['label']
        for method in mm:
            predict = tab_both[method]
            v = predict[np.isfinite(predict)]
            predict = predict.clip(np.min(v), np.max(v))
            predict[np.isnan(predict)] = 0.0
            tab_metrics_p1.loc[db1, method] = dist_metrics['acc'](label, predict)
            tab_metrics_p2.loc[db1, method] = dist_metrics['auc'](label, predict)

    tab_metrics_p1.loc['AVR'] = tab_metrics_p1.mean(0)
    tab_metrics_p2.loc['AVR'] = tab_metrics_p2.mean(0)

    accuracy_path = os.path.join(output_path, "acc.csv")
    auc_path = os.path.join(output_path, "auc.csv")

    tab_metrics_p1.to_csv(accuracy_path)
    tab_metrics_p2.to_csv(auc_path)
    print(tab_metrics_p1)
    print(tab_metrics_p2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, help="The path to of the csv files of the output of the networks", default="./results_tst")
    parser.add_argument("--out_dir", type=str, help="The Path where the csv containing the calculated metrics of the networks should be saved", default="./")
    args = vars(parser.parse_args())
    calculate_metrics(args['csv_dir'], args['out_dir'])


main()
