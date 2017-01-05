#!/usr/bin/env python
#--------------------------------
# Name:         monte_carlo_func.py
# Purpose:      METRIC Automated Hot/Cold Pixel Selection
# Author:       Charles Morton
# Created       2016-09-02
# Python:       2.7
#--------------------------------

import argparse
import ConfigParser
from datetime import datetime
import logging
import pandas as pd
import os
import random
import re
import sys
import shutil
import glob
from time import sleep
import numpy as np
import matplotlib.pyplot as plt

import pixel_points_func as pixel_points
import metric_model2_func as metric_model2
import auto_calibration_func as auto_calibration
from scipy import interpolate, ndimage, optimize, stats

import gdal_common as gdc
import et_image
from python_common import read_param

def monte_carlo(image_ws, mc_ini):

    crop_type_name = ['1_Corn',
                      '2_Cotton',
                      '21_Barley',
                      '24_Wheat',
                      '36_Alfalfa',
                      '37_Non Alfalfa',
                      '54_Tomatoes',
                      '69_Grapes',
                      '75_Almond',
                      '76_Walnut',
                      '204_Pistachios',
                      '212_Oranges',
                      '225_WinWht_Corn']

    env = gdc.env
    image = et_image.Image(image_ws, env)

    # Folder names
    etrf_ws = os.path.join(image_ws, 'ETRF')
    ts_ws = os.path.join(image_ws, 'ts')
    albedo_ws=os.path.join(image_ws, 'albedo_at_sur')

    print (albedo_ws)
    region_ws = os.path.join(image_ws, 'PIXEL_REGIONS')
    plots_ws = os.path.join(image_ws, 'PLOTS')

    for root, dirs, files in os.walk(plots_ws):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

    for crop_type in crop_type_name:

        print (image_ws)

        # File names
        r_fmt = '.img'
        mc_fmt = '.img'
        etrf_path = os.path.join(etrf_ws, 'et_rf' + mc_fmt)
        ts_path = os.path.join(ts_ws + mc_fmt)
        albedo_path = os.path.join(albedo_ws + mc_fmt)

        print (albedo_path)

        # Change here
        region_path = os.path.join(region_ws, crop_type + r_fmt)
        region_mask = gdc.raster_to_array(region_path, return_nodata=False)
        region_mask = region_mask.astype(np.bool)

        ndvi_array = gdc.raster_to_array(image.ndvi_toa_raster, return_nodata=False)

        ts_array = gdc.raster_to_array(image.ts_raster, return_nodata=False)
        # Only process ag. ETrF pixels
        ts_array[~region_mask] = np.nan
        ts_sub_array = ts_array[region_mask]
        auto_calibration.save_ts_histograms(ts_sub_array, plots_ws)

        albedo_array = gdc.raster_to_array(image.albedo_sur_raster, return_nodata=False)
        # Only process ag. ETrF pixels
        albedo_array[~region_mask] = np.nan
        albedo_sub_array = albedo_array[region_mask]
        auto_calibration.save_albedo_histograms(albedo_sub_array, plots_ws)

        # Disable this part when not using iterative ETrF plot
        etrf_array = gdc.raster_to_array(etrf_path, return_nodata=False)
        etrf_array[~region_mask] = np.nan
        etrf_sub_array = etrf_array[region_mask]

        # Only process ag. ETrF pixels
        ndvi_array[~region_mask] = np.nan
        ndvi_sub_array = ndvi_array[region_mask]
        auto_calibration.save_ndvi_histograms(ndvi_sub_array, plots_ws)

        # Generate ndvi ts density plot
        auto_calibration.save_ndvi_ts_density_plot(ndvi_sub_array, ts_sub_array, plots_ws)
        auto_calibration.save_etrf_histograms(etrf_sub_array, plots_ws)
        auto_calibration.save_ndvi_etrf_density_plot(ndvi_sub_array, etrf_sub_array, ts_sub_array,albedo_sub_array,plots_ws)
        auto_calibration.save_etrf_ts_density_plot(etrf_sub_array, ts_sub_array, plots_ws)

    ComDir=r'M:/Central_Valley_Documentation/Results_individual_crop/Mean_Standard_deviation'

    plots_ws = os.path.join(image_ws, 'PLOTS')
    csv_fmt = '.csv'

    df = pd.read_csv((os.path.join(plots_ws, 'statistics' + csv_fmt)),index_col=False)

    plt.figure(figsize=(20, 10))

    plt.rcParams.update({'font.size': 18})

    df['Index']=range(1, len(df) + 1)
    plt.xticks(df['Index'], df['crop'],rotation=30,fontsize=23)
    # plt.setp(labels, rotation=30, fontsize=10)

    width = 0.2       # the width of the bars

    ax = plt.subplot(111)

    plt.bar(df['Index'],df['mean_etrf'], width,  color='g', yerr=df['standard_deviation_etrf'],label="mean_etrf",align='center', ecolor="black")
    plt.bar(df['Index']+width,df['mean_ndvi'], width, color='y', yerr=df['standard_deviation_ndvi'], label='mean_ndvi',align='center',ecolor="black")
    plt.bar(df['Index'] + width+width, df['mean_albedo'], width, color='b', yerr=df['standard_deviation_albedo'],
            label='mean_albedo', align='center',ecolor="black")

    plt.ylabel('ETrF/NDVI/albedo', fontsize=25)
    plt.ylim(0, 1.3)

    plt.xlabel('Crops', fontsize=25)
    plt.legend(loc='upper left')

    ax2 = ax.twinx()

    plt.bar(df['Index'] + width+width+width, df['mean_ts'], width,color='r', yerr=df['standard_deviation_ts'],
            label='mean_ts', align='center',ecolor="black")

    plt.ylabel('Ts (K)',fontsize=25)
    plt.ylim(280, 330)
    plt.legend(loc='upper right' )

    plt.grid()

    plt.title(str(df['date'].iloc[0]), fontsize=18)

    plt.tight_layout()

    plt.savefig(os.path.join(
        plots_ws,  str(df['date'].iloc[0])+'.png'))

    plt.savefig(os.path.join(
        ComDir,  str(df['date'].iloc[0])+'.png'))
    # plt.show()
    plt.close()

def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='METRIC Monte Carlo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'workspace', nargs='?', default=os.getcwd(),
        help='Landsat scene folder', metavar='FOLDER')
    parser.add_argument(
        '-gs', '--groupsize', default=64, type=int,
        help='Minimum group size for placing calibration points')
    parser.add_argument(
        '--metric_ini', required=True,
        help='METRIC input file', metavar='PATH')
    parser.add_argument(
        '--mc_ini', required=True,
        help='Monte Carlo input file', metavar='FILE')
    parser.add_argument(
        '-o', '--overwrite',
        default=None, action='store_true',
        help='Force overwrite of existing files')
    parser.add_argument(
        '--stats', default=False, action="store_true",
        help='Compute raster statistics')

    args = parser.parse_args()

    # Convert input file to an absolute path
    if args.workspace and os.path.isdir(os.path.abspath(args.workspace)):
        args.workspace = os.path.abspath(args.workspace)
    if args.metric_ini and os.path.isfile(os.path.abspath(args.metric_ini)):
        args.metric_ini = os.path.abspath(args.metric_ini)
    if args.mc_ini and os.path.isfile(os.path.abspath(args.mc_ini)):
        args.mc_ini = os.path.abspath(args.mc_ini)
    return args

if __name__ == '__main__':
    args = arg_parse()

    # METRIC Monte Carlo
    monte_carlo(image_ws=args.workspace, mc_ini=args.mc_ini)
