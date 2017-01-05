#!/usr/bin/env python
#--------------------------------
# Name:         auto_calibration_func.py
# Purpose:      METRIC Automated Calibration based on ETRF distribution
# Author:       Ramesh Dhungel/Charles Morton
# Created       2017-01-04
# Python:       2.7
#--------------------------------

import argparse
# import ConfigParser
from datetime import datetime
import logging
import os
import pandas as pd
from random import choice
import re
import string
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy import interpolate, ndimage, optimize, stats

import gdal_common as gdc
import et_common
from python_common import remove_file

break_line = '\n{0}'.format('#' * 80)
pixel_str_fmt = '    {0:<12s}{1:>14s}  {2:>14s}'
pixel_flt_fmt = '    {0:<12s}{1:>14.2f}  {2:>14.2f}'

def calc_histogram_bins(value_array, bin_size=0.01):
    """Calculate histogram bins

    Args:
        etrf_array (): NumPy array
        bin_size (float):

    Returns:
        NumPy array
    """
    bin_min = np.floor(np.nanmin(value_array) / bin_size) * bin_size
    bin_max = np.ceil(np.nanmax(value_array) / bin_size) * bin_size
    bin_count = round((bin_max - bin_min) / bin_size) + 1
    bin_array = np.linspace(bin_min, bin_max, bin_count)
    logging.debug('  Histogram bins')
    logging.debug('    min: {}'.format(bin_min))
    logging.debug('    max: {}'.format(bin_max))
    logging.debug('    count: {}'.format(bin_count))
    # logging.debug('    bins: {}'.format(bin_array))
    return bin_array

counter = 1

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

def save_ndvi_histograms(ndvi_array, plots_ws, mc_iter=None):

    global counter
    if counter == 1:
        Crop = 'Corn'
    elif counter == 2:
        Crop = 'Cotton'
    elif counter == 3:
        Crop = 'Barley'
    elif counter == 4:
        Crop = 'Wheat'
    elif counter == 5:
        Crop = 'Alfalfa'
    elif counter == 6:
        Crop = 'Non Alfalfa'
    elif counter == 7:
        Crop = 'Tomatoes'
    elif counter == 8:
        Crop = 'Grapes'
    elif counter == 9:
        Crop = 'Almond'
    elif counter == 10:
        Crop = 'Walnut'
    elif counter == 11:
        Crop = 'Pistachios'
    elif counter == 12:
        Crop = 'Oranges'
    elif counter == 13:
        Crop = 'WinWht_Corn'

    """Plot NDVI histogram

    Args:
        ndvi_array (str): the NDVI array (assuming non-ag pixels were masked)
        plots_ws (str): the folder for saving the ETrF histogram plots
        mc_iter (int): the current monte carlo iteration (default None)

    Returns:
        None
    """
# for crop_type in crop_type_name:
#     print crop_type

    logging.debug('Save NDVI Histograms')
    scene_id = os.path.basename(os.path.dirname(plots_ws))

# mc_iter is None when run in stand alone mode
# mc_iter index is 1 based when run in Monte Carlo mode
    if mc_iter is None:
        mc_title = ''
        mc_image = ''
    else:
        mc_title = ' - Iteration {0:02d}'.format(int(mc_iter))
        mc_image = '_{0:02d}'.format(int(mc_iter))

# Save historgram on check run
    plt.figure()
    ndvi_bins = calc_histogram_bins(ndvi_array, 0.01)
    n, bins, patches = plt.hist(ndvi_array, bins=ndvi_bins,range=(-0.2, 1.2))
    plt.title(str(Crop)+'_NDVI - {0}{1}'.format( scene_id, mc_title))
    plt.xlabel('NDVI')
    plt.ylabel('# of agricultural pixels')

    ndvi_std = np.nanstd(ndvi_array)
    ndvi_mean = np.nanmean(ndvi_array)
    ndvi_n_f = len(ndvi_array)

    mean_ndvi= "Mean-" + str(round(ndvi_mean, 2)) + " mm"
    std_ndvi="Standard deviation-" + str(round(ndvi_std, 2)) + " mm"
    n_ndvi_f = "n-" + str(int(ndvi_n_f))

    text = mean_ndvi + '\n' + std_ndvi + '\n' + n_ndvi_f

    plt.annotate(text, xy=(1, 1), xytext=(-15, -15), fontsize=10,
                xycoords='axes fraction', textcoords='offset points',
                bbox=dict(facecolor='white', alpha=0.8),
                horizontalalignment='right', verticalalignment='top')

    plt.xlim(-0.2, 1.2)
    plt.savefig(os.path.join(
        plots_ws, str(Crop)+'_NDVI{0}.png'.format(mc_image)))

    plt.close()

def save_etrf_histograms(etrf_array, plots_ws, mc_iter=None):

    global counter
    if counter == 1:
        Crop = 'Corn'
    elif counter == 2:
        Crop = 'Cotton'
    elif counter == 3:
        Crop = 'Barley'
    elif counter == 4:
        Crop = 'Wheat'
    elif counter == 5:
        Crop = 'Alfalfa'
    elif counter == 6:
        Crop = 'Non Alfalfa'
    elif counter == 7:
        Crop = 'Tomatoes'
    elif counter == 8:
        Crop = 'Grapes'
    elif counter == 9:
        Crop = 'Almond'
    elif counter == 10:
        Crop = 'Walnut'
    elif counter == 11:
        Crop = 'Pistachios'
    elif counter == 12:
        Crop = 'Oranges'
    elif counter == 13:
        Crop = 'WinWht_Corn'
    """Plot ETrF histogram

    Args:
        ndvi_array (str): the NDVI array (assuming non-ag pixels were masked)
        plots_ws (str): the folder for saving the ETrF histogram plots
        mc_iter (int): the current monte carlo iteration (default None)

    Returns:
        None
    """
    logging.debug('Save ETrF Histograms')
    scene_id = os.path.basename(os.path.dirname(plots_ws))

    # mc_iter is None when run in stand alone mode
    # mc_iter index is 1 based when run in Monte Carlo mode
    if mc_iter is None:
        mc_title = ''
        mc_image = ''
    else:
        mc_title = ' - Iteration {0:02d}'.format(int(mc_iter))
        mc_image = '_{0:02d}'.format(int(mc_iter))

    # Save historgram on check run
    plt.figure()
    etrf_bins = calc_histogram_bins(etrf_array, 0.01)
    n, bins, patches = plt.hist(etrf_array, bins=etrf_bins,range=(-3, 3))

    plt.title(str(Crop)+'_ETrF - {0}{1}'.format(scene_id, mc_title))
    plt.xlabel('ETrF')
    plt.ylabel('# of agricultural pixels')
    plt.xlim(-0.2, 1.2)

    etrf_std = np.nanstd(etrf_array)
    etrf_mean = np.nanmean(etrf_array)
    etrf_n = len(etrf_array)

    mean_etrf= "Mean-" + str(round(etrf_mean, 2)) + " mm"
    std_etrf="Standard deviation-" + str(round(etrf_std, 2)) + " mm"
    n_etrf = "n-" + str(int(etrf_n))

    text = mean_etrf + '\n' + std_etrf + '\n' + n_etrf

    plt.annotate(text, xy=(1, 1), xytext=(-15, -15), fontsize=10,
                xycoords='axes fraction', textcoords='offset points',
                bbox=dict(facecolor='white', alpha=0.8),
                horizontalalignment='right', verticalalignment='top')

    # plt.ylim(0,6000)
    plt.savefig(os.path.join(
        plots_ws, str(Crop)+'_ETrF{0}.png'.format(mc_image)))
    # plt.show()
    plt.close()

def save_ts_histograms(ts_array, plots_ws, mc_iter=None):

    global counter
    if counter == 1:
        Crop = 'Corn'
    elif counter == 2:
        Crop = 'Cotton'
    elif counter == 3:
        Crop = 'Barley'
    elif counter == 4:
        Crop = 'Wheat'
    elif counter == 5:
        Crop = 'Alfalfa'
    elif counter == 6:
        Crop = 'Non Alfalfa'
    elif counter == 7:
        Crop = 'Tomatoes'
    elif counter == 8:
        Crop = 'Grapes'
    elif counter == 9:
        Crop = 'Almond'
    elif counter == 10:
        Crop = 'Walnut'
    elif counter == 11:
        Crop = 'Pistachios'
    elif counter == 12:
        Crop = 'Oranges'
    elif counter == 13:
        Crop = 'WinWht_Corn'

    logging.debug('Save Ts Histograms')
    scene_id = os.path.basename(os.path.dirname(plots_ws))

    if mc_iter is None:
        mc_title = ''
        mc_image = ''
    else:
        mc_title = ' - Iteration {0:02d}'.format(int(mc_iter))
        mc_image = '_{0:02d}'.format(int(mc_iter))

    # Save historgram on check run
    plt.figure()
    ts_bins = calc_histogram_bins(ts_array, 0.4)
    n, bins, patches = plt.hist(ts_array, bins=ts_bins,range=(-3, 3))
    plt.title(str(Crop)+'_Ts - {0}{1}'.format(scene_id, mc_title))
    plt.xlabel('Ts(K)')
    plt.ylabel('# of agricultural pixels')

    ts_std = np.nanstd(ts_array)
    ts_mean = np.nanmean(ts_array)
    ts_n = len(ts_array)

    mean_ts= "Mean-" + str(round(ts_mean, 2)) + " mm"
    std_ts="Standard deviation-" + str(round(ts_std, 2)) + " mm"
    n_ts = "n-" + str(int(ts_n))

    text = mean_ts + '\n' + std_ts + '\n' + n_ts

    plt.annotate(text, xy=(1, 1), xytext=(-15, -15), fontsize=10,
                xycoords='axes fraction', textcoords='offset points',
                bbox=dict(facecolor='white', alpha=0.8),
                horizontalalignment='right', verticalalignment='top')

    plt.savefig(os.path.join(
        plots_ws, str(Crop)+'_Ts{0}.png'.format(mc_image)))
    # plt.show()
    plt.close()

def save_albedo_histograms(albedo_array, plots_ws, mc_iter=None):

    global counter
    if counter == 1:
        Crop = 'Corn'
    elif counter == 2:
        Crop = 'Cotton'
    elif counter == 3:
        Crop = 'Barley'
    elif counter == 4:
        Crop = 'Wheat'
    elif counter == 5:
        Crop = 'Alfalfa'
    elif counter == 6:
        Crop = 'Non Alfalfa'
    elif counter == 7:
        Crop = 'Tomatoes'
    elif counter == 8:
        Crop = 'Grapes'
    elif counter == 9:
        Crop = 'Almond'
    elif counter == 10:
        Crop = 'Walnut'
    elif counter == 11:
        Crop = 'Pistachios'
    elif counter == 12:
        Crop = 'Oranges'
    elif counter == 13:
        Crop = 'WinWht_Corn'

    logging.debug('Save albedo Histograms')
    scene_id = os.path.basename(os.path.dirname(plots_ws))

    if mc_iter is None:
        mc_title = ''
        mc_image = ''
    else:
        mc_title = ' - Iteration {0:02d}'.format(int(mc_iter))
        mc_image = '_{0:02d}'.format(int(mc_iter))

    # Save historgram on check run
    plt.figure()
    albedo_bins = calc_histogram_bins(albedo_array, 0.01)
    n, bins, patches = plt.hist(albedo_array, bins=albedo_bins,range=(-3, 3))
    plt.title(str(Crop)+'_albedo - {0}{1}'.format(scene_id, mc_title))
    plt.xlabel('Albedo')
    plt.ylabel('# of agricultural pixels')

    albedo_std = np.nanstd(albedo_array)
    albedo_mean = np.nanmean(albedo_array)
    albedo_n = len(albedo_array)

    mean_albedo= "Mean-" + str(round(albedo_mean, 2)) + " mm"
    std_albedo="Standard deviation-" + str(round(albedo_std, 2)) + " mm"
    n_albedo = "n-" + str(int(albedo_n))

    text = mean_albedo + '\n' + std_albedo + '\n' + n_albedo

    plt.annotate(text, xy=(1, 1), xytext=(-15, -15), fontsize=10,
                xycoords='axes fraction', textcoords='offset points',
                bbox=dict(facecolor='white', alpha=0.8),
                horizontalalignment='right', verticalalignment='top')

    plt.savefig(os.path.join(
        plots_ws, str(Crop)+'_Albedo{0}.png'.format(mc_image)))
    # plt.show()
    plt.close()

def save_ndvi_ts_density_plot(ndvi_array,ts_array, plots_ws, mc_iter=None):

    global counter
    if counter == 1:
        Crop = 'Corn'
    elif counter == 2:
        Crop = 'Cotton'
    elif counter == 3:
        Crop = 'Barley'
    elif counter == 4:
        Crop = 'Wheat'
    elif counter == 5:
        Crop = 'Alfalfa'
    elif counter == 6:
        Crop = 'Non Alfalfa'
    elif counter == 7:
        Crop = 'Tomatoes'
    elif counter == 8:
        Crop = 'Grapes'
    elif counter == 9:
        Crop = 'Almond'
    elif counter == 10:
        Crop = 'Walnut'
    elif counter == 11:
        Crop = 'Pistachios'
    elif counter == 12:
        Crop = 'Oranges'
    elif counter == 13:
        Crop = 'WinWht_Corn'

    logging.debug('Save NDVI Ts density plot')
    scene_id = os.path.basename(os.path.dirname(plots_ws))

    if mc_iter is None:
        mc_title = ''
        mc_image = ''
    else:
        mc_title = ' - Iteration {0:02d}'.format(int(mc_iter))
        mc_image = '_{0:02d}'.format(int(mc_iter))

    # Save historgram on check run
    plt.figure()
    plt.hexbin(ndvi_array, ts_array, mincnt=5,gridsize = 200)
    plt.title(str(Crop)+'_NDVI_Ts - {0}{1}'.format(scene_id, mc_title))
    plt.xlabel('NDVI')
    plt.ylabel('Ts (K)')
    plt.savefig(os.path.join(
        plots_ws, str(Crop)+'_NDVI_Ts{0}.png'.format(mc_image)))
    # plt.show()
    plt.close()

def save_ndvi_etrf_density_plot(ndvi_array,etrf_array, ts_array, albedo_array, plots_ws, mc_iter=None):

    global counter
    if counter == 1:
        Crop = 'Corn'
    elif counter == 2:
        Crop = 'Cotton'
    elif counter == 3:
        Crop = 'Barley'
    elif counter == 4:
        Crop = 'Wheat'
    elif counter == 5:
        Crop = 'Alfalfa'
    elif counter == 6:
        Crop = 'Non Alfalfa'
    elif counter == 7:
        Crop = 'Tomatoes'
    elif counter == 8:
        Crop = 'Grapes'
    elif counter == 9:
        Crop = 'Almond'
    elif counter == 10:
        Crop = 'Walnut'
    elif counter == 11:
        Crop = 'Pistachios'
    elif counter == 12:
        Crop = 'Oranges'
    elif counter == 13:
        Crop = 'WinWht_Corn'

    logging.debug('Save NDVI ETrF density plot')
    scene_id = os.path.basename(os.path.dirname(plots_ws))

    if mc_iter is None:
        mc_title = ''
        mc_image = ''
    else:
        mc_title = ' - Iteration {0:02d}'.format(int(mc_iter))
        mc_image = '_{0:02d}'.format(int(mc_iter))

    # Save historgram on check run
    plt.figure()
    plt.hexbin(ndvi_array, etrf_array, mincnt=5,gridsize = 200)
    plt.plot([0, 1], color="black", linewidth=5)
    plt.title(str(Crop)+'_ETrF_NDVI - {0}{1}'.format(scene_id, mc_title))
    plt.xlabel('NDVI')
    plt.ylabel('ETrF')

    ####################################################################
    # with r2

    mask = ~np.isnan(ndvi_array) & ~np.isnan(etrf_array)

    ndvi_array=ndvi_array[mask]
    etrf_array=etrf_array[mask]

    x_limits = np.array([min(ndvi_array), max(ndvi_array)])
    x_data = np.arange(
        x_limits[0], x_limits[1], 0.01 * (x_limits[1] - x_limits[0]))

    def rto_func(x, a):
        return a * x

    def linear_func(x, a, b):
        return (a * x) + b

    def log_func(x, a, b, c):
        return a * np.log(b * x) + c

    def see_func(y, y_est):
        return float(np.sqrt(np.sum((y - y_est) ** 2) / y.shape[0]))

    fit_type = 'linear'
    if fit_type == 'linear':
        p = optimize.curve_fit(
            linear_func, ndvi_array, etrf_array,
            [1., 0.], maxfev=10000)[0]
        fit_array = linear_func(ndvi_array, *p)
        y_data = linear_func(x_data, *p)
    elif fit_type == 'rto':
        p = optimize.curve_fit(
            rto_func, ndvi_array, etrf_array, 1., maxfev=10000)[0]
        fit_array = rto_func(ndvi_array, *p)
        y_data = rto_func(x_data, *p)
    elif fit_type == 'log':
        p = optimize.curve_fit(
            log_func, ndvi_array, etrf_array,
            [0.6, 1., 1.], maxfev=10000)[0]
        fit_array = log_func(ndvi_array, *p)
        y_data = log_func(x_data, *p)

    plt.plot(x_data, y_data, '-k', lw=1.5)
    plt.plot(x_limits, x_limits, '--k', lw=1.5)
    n = len(ndvi_array)
    per_negative = 100 * float(np.sum(etrf_array <= 0)) / n

    r = stats.pearsonr(etrf_array, fit_array)[0]
    see = see_func(etrf_array, fit_array)

    if fit_type == 'rto':
        text_x = 0.70
        plt.figtext(text_x, 0.25, ('$y = {0:5.3f}x$'.format(p[0])))
    elif fit_type == 'linear':
        text_x = 0.65
        plt.figtext(
            text_x, 0.25,
            ('$y = {0:5.2f}x{1:+5.2f}$'.format(*p)))
    elif fit_type == 'log':
        text_x = 0.60
        plt.figtext(
            text_x, 0.25,
            ('$y = {0:5.2f}log({1:5.2f}x){2:+5.2f}$'.format(*p)))
    plt.figtext(text_x, 0.21, ('$R^2\!= {0:6.4f}$'.format(r ** 2)))
    plt.figtext(text_x, 0.17, ('$S_E\! = {0:6.4f}$'.format(see)))
    plt.figtext(text_x, 0.13, ('$n = {0}$'.format(n)))
    #
    #################################

    ndvi_std = np.nanstd(ndvi_array)
    ndvi_mean = np.nanmean(ndvi_array)
    ndvi_n = len(ndvi_array)
    mask_n = len(mask)

    mean_ndvi= "Mean-" + str(round(ndvi_mean, 2)) + " mm"
    std_ndvi="Standard deviation-" + str(round(ndvi_std, 2)) + " mm"
    n_ndvi = "n-" + str(int(ndvi_n))
    n_mask = "n-" + str(int(mask_n))


    #########################################
    etrf_std = np.nanstd(etrf_array)
    etrf_mean = np.nanmean(etrf_array)
    etrf_n = len(etrf_array)

    mean_etrf= "Mean-" + str(round(etrf_mean, 2)) + " mm"
    std_etrf="Standard deviation-" + str(round(etrf_std, 2)) + " mm"
    n_etrf = "n-" + str(int(etrf_n))

    ###########################################

    ts_std = np.nanstd(ts_array)
    ts_mean = np.nanmean(ts_array)
    ts_n = len(ts_array)

    mean_ts = "Mean-" + str(round(ts_mean, 2)) + " mm"
    std_ts = "Standard deviation-" + str(round(ts_std, 2)) + " mm"
    n_ts = "n-" + str(int(ts_n))

    #############################################

    albedo_std = np.nanstd(albedo_array)
    albedo_mean = np.nanmean(albedo_array)
    albedo_n = len(albedo_array)

    mean_albedo = "Mean-" + str(round(albedo_mean, 2)) + " mm"
    std_albedo = "Standard deviation-" + str(round(albedo_std, 2)) + " mm"
    n_albedo = "n-" + str(int(albedo_n))

    ############################################

    input = open((os.path.join(
        plots_ws, 'statistics.csv')), 'a')

    header = ','.join(('date', 'crop', 'number_b_m','mean_ndvi','standard_deviation_ndvi',
                       'mean_etrf','standard_deviation_etrf','mean_albedo', 'standard_deviation_albedo',
                       'mean_ts', 'standard_deviation_ts','number_a_m','S_ndvi_etrf','r2_ndvi_etrf'
                       '\n'))

    if counter == 1:
        input.write(header)

    result = ','.join(('{0}{1}'.format( scene_id, mc_title), Crop,
                       str(int(mask_n)),str(round(ndvi_mean, 2)), str(round(ndvi_std, 2)),
                       str(round(etrf_mean, 2)), str(round(etrf_std, 2)),str(round(albedo_mean, 2)), str(round(albedo_std, 2)),
                       str(round(ts_mean, 2)), str(round(ts_std, 2)),str(int(ndvi_n)),str(round(see,3)),str(round(r ** 2,3)),
                       '\n'))
    input.write(result)

    ####################################

    plt.savefig(os.path.join(
        plots_ws, str(Crop)+'_NDVI_ETrF{0}.png'.format(mc_image)))
    # plt.show()
    plt.close()

def save_etrf_ts_density_plot(etrf_array, ts_array,plots_ws, mc_iter=None):

    global counter
    global Crop

    if counter == 1:
        Crop = 'Corn'
    elif counter == 2:
        Crop = 'Cotton'
    elif counter == 3:
        Crop = 'Barley'
    elif counter == 4:
        Crop = 'Wheat'
    elif counter == 5:
        Crop = 'Alfalfa'
    elif counter == 6:
        Crop = 'Non Alfalfa'
    elif counter == 7:
        Crop = 'Tomatoes'
    elif counter == 8:
        Crop = 'Grapes'
    elif counter == 9:
        Crop = 'Almond'
    elif counter == 10:
        Crop = 'Walnut'
    elif counter == 11:
        Crop = 'Pistachios'
    elif counter == 12:
        Crop = 'Oranges'
    elif counter == 13:
        Crop = 'WinWht_Corn'

    logging.debug('Save ETrF Ts density plot')
    scene_id = os.path.basename(os.path.dirname(plots_ws))

    counter += 1
    logging.debug('Save ETrF Ts density plot')
    scene_id = os.path.basename(os.path.dirname(plots_ws))

    if mc_iter is None:
        mc_title = ''
        mc_image = ''
    else:
        mc_title = ' - Iteration {0:02d}'.format(int(mc_iter))
        mc_image = '_{0:02d}'.format(int(mc_iter))

    # Save historgram on check run
    plt.figure()
    plt.hexbin(ts_array, etrf_array, mincnt=5,gridsize = 200)
    plt.title(str(Crop)+'_ETrF_Ts - {0}{1}'.format(scene_id, mc_title))
    plt.xlabel('Ts (K)')
    plt.ylabel('ETrF')

    plt.savefig(os.path.join(
        plots_ws, str(Crop)+'_ETrF_Ts{0}.png'.format(mc_image)))
    # plt.show()
    plt.close()


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='METRIC Automated Calibration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'workspace', nargs='?', default=os.getcwd(),
        help='Landsat scene folder', metavar='FOLDER')
    parser.add_argument(
        '-i', '--ini', required=True, metavar='PATH',
        help='Monte Carlo input file')
    args = parser.parse_args()

    # Convert input file to an absolute path
    if args.workspace and os.path.isdir(os.path.abspath(args.workspace)):
        args.workspace = os.path.abspath(args.workspace)
    if args.ini and os.path.isfile(os.path.abspath(args.ini)):
        args.ini = os.path.abspath(args.ini)
    return args


if __name__ == '__main__':
    args = arg_parse()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    log_console = logging.StreamHandler()
    log_console.setLevel(args.loglevel)
    formatter = logging.Formatter('%(message)s')
    log_console.setFormatter(formatter)
    logger.addHandler(log_console)

    if not args.no_file_logging:
        log_file_name = 'auto_calibration_log.txt'
        log_file = logging.FileHandler(
            os.path.join(args.workspace, log_file_name), "w")
        log_file.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        log_file.setFormatter(formatter)
        logger.addHandler(log_file)

    logging.info(break_line)
    log_fmt = '{0:<20s} {1}'
    logging.info(log_fmt.format(
        'Run Time Stamp:', datetime.now().isoformat(' ')))
    logging.info(log_fmt.format('Current Directory:', args.workspace))
    logging.info(log_fmt.format('Script:', os.path.basename(sys.argv[0])))
