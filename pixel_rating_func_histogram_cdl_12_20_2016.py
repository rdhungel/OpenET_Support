#!/usr/bin/env python
# --------------------------------
# Name:         pixel_rating_func.py
# Purpose:      METRIC Hot/Cold Pixel Rating Tool
# Author:       Charles Morton/Ramesh Dhungel
# Created       2016-01-25
# Python:       2.7
# --------------------------------

import argparse
import ConfigParser
from datetime import datetime
import logging
import math
import os
import random
import sys
from time import sleep

import numpy as np
from osgeo import gdal, ogr, osr
from scipy import stats, ndimage

import gdal_common as gdc
import et_image
from python_common import read_param, remove_file


def pixel_rating(image_ws, config_file, stats_flag=False, overwrite_flag=None):
    """Calculate pixel rating

    Args:
        image_ws (str): Image folder path
        config_file (str): Pixel regions config file path
        stats_flag (bool): if True, compute raster statistics
        ovewrite_flag (bool): if True, overwrite existing files

    Returns:
        None
    """
    logging.info('Generating suggested hot/cold pixel regions')
    log_fmt = '  {0:<18s} {1}'

    env = gdc.env
    image = et_image.Image(image_ws, env)
    np.seterr(invalid='ignore')

    # Check  that image_ws is valid
    # image_re = re.compile('^(LT4|LT5|LE7)(\d{3})(\d{3})(\d{4})(\d{3})')
    # if not os.path.isdir(image_ws) or not image_re.match(scene_id):
    #     logging.error('\nERROR: Image folder is invalid or does not exist\n')
    #     return False

    # Folder Paths
    region_ws = os.path.join(image_ws, 'PIXEL_REGIONS')

    # Open pixel regions config file
    logging.debug(log_fmt.format(
        'Pixel Regions INI:', os.path.basename(args.ini)))
    config = ConfigParser.ConfigParser()
    try:
        config.readfp(open(config_file))
    except:
        logging.error(('\nERROR: Config file could not be read, ' +
                       'is not an input file, or does not exist\n' +
                       'ERROR: config_file = {0}\n').format(config_file))
        sys.exit()
    logging.debug('Reading Input File')

    # Arrays are processed by block
    bs = read_param('block_size', 1024, config)
    logging.info('  {0:<18s} {1}'.format('Block Size:', bs))

    # Raster pyramids/statistics
    pyramids_flag = read_param('pyramids_flag', False, config)
    if pyramids_flag:
        gdal.SetConfigOption('HFA_USE_RRD', 'YES')
    if stats_flag is None:
        stats_flag = read_param('statistics_flag', False, config)

    # Overwrite
    if overwrite_flag is None:
        overwrite_flag = read_param('overwrite_flag', True, config)

    # Check that common_area raster exists
    if not os.path.isfile(image.common_area_raster):
        logging.error(
            '\nERROR: A common area raster was not found.' +
            '\nERROR: Please rerun prep tool to build these files.\n' +
            '    {}\n'.format(image.common_area_raster))
        sys.exit()

    # Use common_area to set mask parameters
    common_ds = gdal.Open(image.common_area_raster)
    # env.mask_proj = raster_ds_proj(common_ds)
    env.mask_geo = gdc.raster_ds_geo(common_ds)
    env.mask_rows, env.mask_cols = gdc.raster_ds_shape(common_ds)
    env.mask_extent = gdc.geo_extent(
        env.mask_geo, env.mask_rows, env.mask_cols)
    env.mask_array = gdc.raster_ds_to_array(common_ds)[0]
    env.mask_path = image.common_area_raster
    env.snap_osr = gdc.raster_path_osr(image.common_area_raster)
    env.snap_proj = env.snap_osr.ExportToWkt()
    env.cellsize = gdc.raster_path_cellsize(image.common_area_raster)[0]
    common_ds = None
    logging.debug('  {0:<18s} {1}'.format('Mask Extent:', env.mask_extent))

    # Read Pixel Regions config file
    # Currently there is no code to support applying an NLCD mask
    apply_nlcd_mask = False
    # apply_nlcd_mask = read_param('apply_nlcd_mask', False, config)
    apply_cdl_ag_mask = read_param('apply_cdl_ag_mask', False, config)
    apply_field_mask = read_param('apply_field_mask', False, config)
    apply_ndwi_mask = read_param('apply_ndwi_mask', True, config)
    # Currently the code to apply a study area mask is commented out
    # apply_study_area_mask = read_param(
    #     'apply_study_area_mask', False, config)

    # Set the flags that control which rating algorithms are used based
    #  on the image type, either Landsat or MODIS
    if 'MODIS' in image.type:
        albedo_rating_flag = read_param('albedo_rating_flag', False, config)
        nlcd_rating_flag = read_param('nlcd_rating_flag', False, config)
        ndvi_rating_flag = read_param('ndvi_rating_flag', True, config)
        ts_rating_flag = read_param('ts_rating_flag', True, config)
        ke_rating_flag = read_param('ke_rating_flag', False, config)
    if 'Landsat' in image.type:
        albedo_rating_flag = read_param('albedo_rating_flag', True, config)
        nlcd_rating_flag = read_param('nlcd_rating_flag', True, config)
        ndvi_rating_flag = read_param('ndvi_rating_flag', True, config)
        ts_rating_flag = read_param('ts_rating_flag', True, config)
        ke_rating_flag = read_param('ke_rating_flag', False, config)

    # if apply_study_area_mask:
    #     study_area_path = config.get('INPUTS', 'study_area_path')
    if apply_nlcd_mask or nlcd_rating_flag:
        nlcd_raster = config.get('INPUTS', 'landuse_raster')

    if apply_cdl_ag_mask:
        cdl_ag_raster = config.get('INPUTS', 'cdl_ag_raster')
        cdl_buffer_cells = read_param('cdl_buffer_cells', 0, config)
        cdl_ag_eroded_name = read_param(
            'cdl_ag_eroded_name', 'cdl_ag_eroded_{0}.img', config)

    if apply_cdl_ag_mask:
        cdl_raster = config.get('INPUTS', 'cdl_raster')
        cdl_buffer_cells = read_param('cdl_buffer_cells', 0, config)
        cdl_ag_eroded_name = read_param(
            'cdl_ag_eroded_name', 'cdl_ag_eroded_{0}.img', config)

        print (cdl_raster)

    if apply_field_mask:
        field_raster = config.get('INPUTS', 'fields_raster')

    cold_rating_pct = read_param('cold_percentile', 99, config)
    hot_rating_pct = read_param('hot_percentile', 99, config)
    # min_cold_rating_score = read_param('min_cold_rating_score', 0.3, config)
    # min_hot_rating_score = read_param('min_hot_rating_score', 0.3, config)

    ts_bin_count = int(read_param('ts_bin_count', 10, config))
    if 100 % ts_bin_count != 0:
        logging.warning(('WARNING: ts_bins_count of {0} is not a divisor ' +
                         'of 100. Using default ts_bins_count = 4').format(
            ts_bin_count))
        ts_bin_count = 10
    bin_size = 1. / (ts_bin_count - 1)
    hot_rating_values = np.arange(0., 1. + bin_size, step=bin_size)
    cold_rating_values = hot_rating_values[::-1]

    # Input raster paths
    r_fmt = '.img'
    if 'Landsat' in image.type:
        albedo_raster = image.albedo_sur_raster
        ndvi_raster = image.ndvi_toa_raster
        ndwi_raster = image.ndwi_toa_raster
        ts_raster = image.ts_raster
        ke_raster = image.ke_raster
    elif 'MODIS' in image.type:
        albedo_raster = image.albedo_sur_raster
        ndvi_raster = image.ndvi_sur_raster
        ndwi_raster = image.ndwi_sur_raster
        ts_raster = image.ts_raster
        ke_raster = image.ke_raster

    # Check config file input paths
    # if apply_study_area_mask and not os.path.isfile(study_area_path):
    #     logging.error(
    #         ('\nERROR: The study area shapefile {0} does ' +
    #             'not exist\n').format(study_area_path))
    #     sys.exit()
    if ((apply_nlcd_mask or nlcd_rating_flag) and
            not os.path.isfile(nlcd_raster)):
        logging.error(
            ('\nERROR: The NLCD raster {0} does ' +
             'not exist\n').format(nlcd_raster))
        sys.exit()
    if apply_cdl_ag_mask and not os.path.isfile(cdl_ag_raster):
        logging.error(
            ('\nERROR: The CDL Ag raster {0} does ' +
             'not exist\n').format(cdl_ag_raster))
        sys.exit()
    if apply_field_mask and not os.path.isfile(field_raster):
        logging.error(
            ('\nERROR: The field raster {0} does ' +
             'not exist\n').format(field_raster))
        sys.exit()
    if (not (isinstance(cold_rating_pct, (int, long, float)) and
                 (0 <= cold_rating_pct <= 100))):
        logging.error(
            '\nERROR: cold_percentile must be a value between 0 and 100\n')
        sys.exit()
    if (not (isinstance(hot_rating_pct, (int, long, float)) and
                 (0 <= hot_rating_pct <= 100))):
        logging.error(
            '\nERROR: hot_percentile must be a value between 0 and 100\n')
        sys.exit()

    # Set raster names
    raster_dict = dict()

    # Output Rasters
    raster_dict['region_mask'] = os.path.join(
        region_ws, 'region_mask' + r_fmt)
    raster_dict['cold_rating'] = os.path.join(
        region_ws, 'cold_pixel_rating' + r_fmt)
    raster_dict['hot_rating'] = os.path.join(
        region_ws, 'hot_pixel_rating' + r_fmt)
    raster_dict['cold_sugg'] = os.path.join(
        region_ws, 'cold_pixel_suggestion' + r_fmt)
    raster_dict['hot_sugg'] = os.path.join(
        region_ws, 'hot_pixel_suggestion' + r_fmt)

    # Read pixel region raster flags
    save_dict = dict()
    save_dict['region_mask'] = read_param(
        'save_region_mask_flag', False, config)
    save_dict['cold_rating'] = read_param(
        'save_rating_rasters_flag', False, config)
    save_dict['hot_rating'] = read_param(
        'save_rating_rasters_flag', False, config)
    save_dict['cold_sugg'] = read_param(
        'save_suggestion_rasters_flag', True, config)
    save_dict['hot_sugg'] = read_param(
        'save_suggestion_rasters_flag', True, config)

    # Output folder
    if not os.path.isdir(region_ws):
        os.mkdir(region_ws)

    # Remove existing files if necessary
    region_ws_file_list = [
        os.path.join(region_ws, item) for item in os.listdir(region_ws)]
    if overwrite_flag and region_ws_file_list:
        for raster_path in raster_dict.values():
            if raster_path in region_ws_file_list:
                remove_file(raster_path)

    # Check scene specific input paths
    if apply_ndwi_mask and not os.path.isfile(ndwi_raster):
        logging.error(
            'ERROR: NDWI raster does not exist\n {}'.format(ndwi_raster))
        sys.exit()
    elif ke_rating_flag and not os.path.isfile(ke_raster):
        logging.error(
            ('ERROR: The Ke raster does not exist\n {}').format(ke_raster))
        sys.exit()

    # Remove existing and build new empty rasters if necessary
    # If processing by block, rating rasters must be built
    logging.debug('\nBuilding empty rasters')
    for name, save_flag in sorted(save_dict.items()):
        if save_flag and 'rating' in name:
            gdc.build_empty_raster(raster_dict[name], 1, np.float32)
        elif save_flag:
            gdc.build_empty_raster(
                raster_dict[name], 1, np.uint8, output_nodata=0)

    if apply_cdl_ag_mask:
        logging.info('Building CDL ag mask')
        cdl_array = gdc.raster_to_array(
            cdl_ag_raster, mask_extent=env.mask_extent,
            return_nodata=False)
        if cdl_buffer_cells > 0:
            logging.info('  Eroding CDL by {} cells'.format(
                cdl_buffer_cells))
            structure_array = np.ones(
                (cdl_buffer_cells, cdl_buffer_cells), dtype=np.int)
            # Deadbeef - This could blow up in memory on bigger rasters
            cdl_array = ndimage.binary_erosion(
                cdl_array, structure_array).astype(structure_array.dtype)
        cdl_ag_eroded_raster = os.path.join(
            image.support_ws, cdl_ag_eroded_name.format(cdl_buffer_cells))
        gdc.array_to_raster(
            cdl_array, cdl_ag_eroded_raster, output_geo=env.mask_geo,
            output_proj=env.snap_proj, mask_array=env.mask_array,
            output_nodata=0, stats_flag=False)
        cdl_array = None
        del cdl_array

    # Build region mask
    logging.debug('Building region mask')
    region_mask = np.copy(env.mask_array).astype(np.bool)
    if apply_field_mask:
        field_mask, field_nodata = gdc.raster_to_array(
            field_raster, mask_extent=env.mask_extent,
            return_nodata=True)
        region_mask &= field_mask != field_nodata
        del field_mask, field_nodata
    if apply_ndwi_mask:
        ndwi_array = gdc.raster_to_array(
            ndwi_raster, 1, mask_extent=env.mask_extent,
            return_nodata=False)
        region_mask &= ndwi_array > 0.0
        del ndwi_array
    if apply_cdl_ag_mask:
        cdl_array, cdl_nodata = gdc.raster_to_array(
            cdl_ag_eroded_raster, mask_extent=env.mask_extent,
            return_nodata=True)
        region_mask &= cdl_array != cdl_nodata
        del cdl_array, cdl_nodata
    if save_dict['region_mask']:
        gdc.array_to_raster(
            region_mask, raster_dict['region_mask'],
            stats_flag=False)

    # Initialize rating arrays
    # This needs to be done before the ts_rating if block
    cold_rating_array = np.ones(env.mask_array.shape, dtype=np.float32)
    hot_rating_array = np.ones(env.mask_array.shape, dtype=np.float32)
    cold_rating_array[~region_mask] = np.nan
    hot_rating_array[~region_mask] = np.nan

    # Temperature pixel rating - grab the max and min value for the entire
    #  Ts image in a memory safe way by using gdal_common blocks
    # The following is a percentile based approach
    if ts_rating_flag:
        logging.debug('Computing Ts percentile rating')
        ts_array = gdc.raster_to_array(
            ts_raster, mask_extent=env.mask_extent,
            return_nodata=False)
        ts_array[~region_mask] = np.nan

        percentiles = xrange(
            0, (100 + ts_bin_count), (100 / ts_bin_count))
        ts_score_value = 1. / (ts_bin_count - 1)
        hot_rating_values = np.arange(
            0, (1. + ts_score_value), step=ts_score_value)[:ts_bin_count]
        cold_rating_values = hot_rating_values[::-1]
        ts_percentile_array = stats.scoreatpercentile(
            ts_array[np.isfinite(ts_array)], percentiles)

        for bins_i in range(len(ts_percentile_array))[:-1]:
            bool_array = (
                (ts_array > ts_percentile_array[bins_i]) &
                (ts_array <= ts_percentile_array[bins_i + 1]))
            cold_rating_array[bool_array] = cold_rating_values[bins_i]
            hot_rating_array[bool_array] = hot_rating_values[bins_i]
        # gdc.array_to_raster(cold_rating_array, raster_dict['cold_rating'])
        # gdc.array_to_raster(hot_rating_array, raster_dict['hot_rating'])

        # Cleanup
        del ts_array, ts_percentile_array
        del cold_rating_values, hot_rating_values
        del ts_score_value, percentiles

    # Process by block
    logging.info('\nProcessing by block')
    logging.debug('  Mask  cols/rows: {0}/{1}'.format(
        env.mask_cols, env.mask_rows))
    for b_i, b_j in gdc.block_gen(env.mask_rows, env.mask_cols, bs):
        logging.debug('  Block  y: {0:5d}  x: {1:5d}'.format(b_i, b_j))
        block_data_mask = gdc.array_to_block(
            env.mask_array, b_i, b_j, bs).astype(np.bool)
        # block_nodata_mask = ~block_data_mask
        block_rows, block_cols = block_data_mask.shape
        block_geo = gdc.array_offset_geo(env.mask_geo, b_j, b_i)
        block_extent = gdc.geo_extent(block_geo, block_rows, block_cols)
        logging.debug('    Block rows: {0}  cols: {1}'.format(
            block_rows, block_cols))
        # logging.debug('    Block extent: {0}'.format(block_extent))
        # logging.debug('    Block geo: {0}'.format(block_geo))

        # Don't skip empty blocks since block rating needs to be written
        #  back to the array at the end of the block loop
        block_region_mask = gdc.array_to_block(region_mask, b_i, b_j, bs)
        if not np.any(block_region_mask):
            logging.debug('    Empty block')
            block_empty_flag = True
        else:
            block_empty_flag = False

        # New style continuous pixel weighting
        cold_rating_block = gdc.array_to_block(
            cold_rating_array, b_i, b_j, bs)
        hot_rating_block = gdc.array_to_block(
            hot_rating_array, b_i, b_j, bs)

        # Rating arrays already have region_mask set
        # cold_rating_block = np.ones(block_region_mask.shape, dtype=np.float32)
        # hot_rating_block = np.ones(block_region_mask.shape, dtype=np.float32)
        # cold_rating_block[~block_region_mask] = np.nan
        # hot_rating_block[~block_region_mask] = np.nan
        # del block_region_mask

        if ndvi_rating_flag and not block_empty_flag:
            # NDVI based rating
            ndvi_array = gdc.raster_to_array(
                ndvi_raster, 1, mask_extent=block_extent,
                return_nodata=False)
            # Don't let NDVI be negative
            ndvi_array.clip(0., 0.833, out=ndvi_array)
            # ndvi_array.clip(0.001, 0.833, out=ndvi_array)
            cold_rating_block *= ndvi_array
            cold_rating_block *= 1.20
            hot_rating_block *= stats.norm.pdf(
                np.log(ndvi_array), math.log(0.15), 0.5)
            hot_rating_block *= 1.25
            # cold_rating_block.clip(0., 1., out=cold_rating_block)
            # hot_rating_block.clip(0., 1., out=hot_rating_block)
            del ndvi_array

        if albedo_rating_flag and not block_empty_flag:
            # Albdo based rating
            albedo_array = gdc.raster_to_array(
                albedo_raster, 1, mask_extent=block_extent,
                return_nodata=False)
            if image.type.lower() == 'modis':
                albedo_cold_pdf = stats.norm.pdf(albedo_array, 0.14, 0.03)
                albedo_hot_pdf = stats.norm.pdf(albedo_array, 0.14, 0.06)
            else:
                albedo_cold_pdf = stats.norm.pdf(albedo_array, 0.21, 0.03)
                albedo_hot_pdf = stats.norm.pdf(albedo_array, 0.21, 0.06)
            del albedo_array
            cold_rating_block *= albedo_cold_pdf
            cold_rating_block *= 0.07
            hot_rating_block *= albedo_hot_pdf
            hot_rating_block *= 0.15
            # cold_rating_block.clip(0., 1., out=cold_rating_block)
            # hot_rating_block.clip(0., 1., out=hot_rating_block)
            del albedo_cold_pdf, albedo_hot_pdf

        if nlcd_rating_flag and not block_empty_flag:
            # NLCD based weighting, this could be CDL instead?
            nlcd_array = nlcd_rating(gdc.raster_to_array(
                nlcd_raster, 1, mask_extent=block_extent,
                return_nodata=False))
            cold_rating_block *= nlcd_array
            hot_rating_block *= nlcd_array
            del nlcd_array

        if ke_rating_flag and not block_empty_flag:
            # SWB Ke based rating
            ke_array = gdc.raster_to_array(
                ke_raster, 1, mask_extent=block_extent,
                return_nodata=False)
            # Don't let NDVI be negative
            ke_array.clip(0., 1., out=ke_array)
            # Assumption, lower Ke is better for selecting the hot pixel
            # As the power (2) decreases and approaches 1,
            #   the relationship gets more linear
            # cold_rating_block *= (1 - ke_array ** 2)
            # hot_rating_block *= (1 - ke_array ** 1.5)
            # Linear inverse
            # cold_rating_block *= (1. - ke_array)
            hot_rating_block *= (1. - ke_array)
            # cold_rating_block.clip(0., 1., out=cold_rating_block)
            # hot_rating_block.clip(0., 1., out=hot_rating_block)
            del ke_array

        # Clearness
        # clearness = 1.0
        # cold_rating *= clearness
        # hot_rating *= clearness

        # Reset nan values
        # cold_rating_block[~region_mask] = np.nan
        # hot_rating_block[~region_mask] = np.nan

        # Save rating values
        cold_rating_array = gdc.block_to_array(
            cold_rating_block, cold_rating_array, b_i, b_j, bs)
        hot_rating_array = gdc.block_to_array(
            hot_rating_block, hot_rating_array, b_i, b_j, bs)

        # Save rating rasters
        if save_dict['cold_rating']:
            gdc.block_to_raster(
                cold_rating_block, raster_dict['cold_rating'], b_i, b_j, bs)
        if save_dict['hot_rating']:
            gdc.block_to_raster(
                hot_rating_block, raster_dict['hot_rating'], b_i, b_j, bs)
        # Save rating values
        cold_rating_array = gdc.block_to_array(
            cold_rating_block, cold_rating_array, b_i, b_j, bs)
        hot_rating_array = gdc.block_to_array(
            hot_rating_block, hot_rating_array, b_i, b_j, bs)

        del cold_rating_block, hot_rating_block

    # Select pixels above target percentile
    # Only build suggestion arrays if saving
    logging.debug('Building suggested pixel rasters')
    if save_dict['cold_sugg']:
        cold_rating_score = float(stats.scoreatpercentile(
            cold_rating_array[np.isfinite(cold_rating_array)],
            cold_rating_pct))
        # cold_rating_array, cold_rating_nodata = gdc.raster_to_array(
        #     raster_dict['cold_rating'], 1, mask_extent=env.mask_extent)
        # if cold_rating_score < float(min_cold_rating_score):
        #     logging.error(('ERROR: The cold_rating_score ({0}) is less ' +
        #                    'than the min_cold_rating_score ({1})').format(
        #                     cold_rating_score, min_cold_rating_score))
        #     sys.exit()
        cold_sugg_mask = cold_rating_array >= cold_rating_score
        gdc.array_to_raster(
            cold_sugg_mask, raster_dict['cold_sugg'], stats_flag=stats_flag)
        logging.debug('  Cold Percentile: {}'.format(cold_rating_pct))
        logging.debug('  Cold Score:  {0:.6f}'.format(cold_rating_score))
        logging.debug('  Cold Pixels: {}'.format(np.sum(cold_sugg_mask)))
        del cold_sugg_mask, cold_rating_array
    if save_dict['hot_sugg']:
        hot_rating_score = float(stats.scoreatpercentile(
            hot_rating_array[np.isfinite(hot_rating_array)],
            hot_rating_pct))
        # hot_rating_array, hot_rating_nodata = gdc.raster_to_array(
        #     raster_dict['hot_rating'], 1, mask_extent=env.mask_extent)
        # if hot_rating_score < float(min_hot_rating_score):
        #     logging.error(('ERROR: The hot_rating_array ({0}) is less ' +
        #                    'than the min_hot_rating_score ({1})').format(
        #                     hot_rating_array, min_hot_rating_score))
        #     sys.exit()
        hot_sugg_mask = hot_rating_array >= hot_rating_score
        gdc.array_to_raster(
            hot_sugg_mask, raster_dict['hot_sugg'], stats_flag=stats_flag)
        logging.debug('  Hot Percentile: {}'.format(hot_rating_pct))
        logging.debug('  Hot Score:  {0:.6f}'.format(hot_rating_score))
        logging.debug('  Hot Pixels: {}'.format(np.sum(hot_sugg_mask)))
        del hot_sugg_mask, hot_rating_array

    # Raster Statistics
    if stats_flag:
        logging.info('Calculating Statistics')
        for name, save_flag in save_dict.items():
            if save_flag:
                gdc.raster_statistics(raster_dict[name])
    # Raster Pyramids
    if pyramids_flag:
        logging.info('Building Pyramids')
        for name, save_flag in save_dict.items():
            if save_flag:
                gdc.raster_pyramids(raster_dict[name])

    cdl_names = {
        1: 'Corn',
        2: 'Cotton',
        21: 'Barley',
        24: 'Wheat',
        36: 'Alfalfa',
        37: 'Non Alfalfa',
        54: 'Tomatoes',
        69: 'Grapes',
        75: 'Almond',
        76: 'Walnut',
        204: 'Pistachios',
        212: 'Oranges',
        225: 'WinWht_Corn'
    }

    for cdl_number, val in cdl_names.items():
        cdl_array = gdc.raster_to_array(
            cdl_raster, mask_extent=env.mask_extent,
            return_nodata=False)

        # Set all not target valeus to NaN
        cdl_crop_mask = (cdl_array == cdl_number)

        # Build output path based on crop number and name
        output_path = os.path.join(
            region_ws, "{}_{}.img".format(cdl_number, val))

        # gdc.array_to_raster(
        #     cdl_crop_mask, output_path, output_geo=input_geo,
        #     output_proj=input_proj)

        gdc.array_to_raster(
            cdl_crop_mask, output_path)


def nlcd_rating(landcover_array):
    """Function to rate pixels based on landcover_array

    The landcover values must be mapped to match the NLCD
    landcover sheme set up by the USGS NLCD 2011.
    http://www.mrlc.gov/nlcd2011.php

    Inputs:
        landcover_array (:class:`np.array`): A NumPy array
            of landcover values to be rated

    Returns:
        :NumPy Array
        A NumPy array of the same shape as the input with the landcover
        class transformed to a rating

    """
    lc_rating_array = np.zeros(
        landcover_array.shape, dtype=np.float32)

    lc_rating_array[landcover_array == 11] = 0.500
    lc_rating_array[landcover_array == 12] = 0.001
    lc_rating_array[landcover_array == 21] = 0.200
    lc_rating_array[landcover_array == 22] = 0.100
    lc_rating_array[landcover_array == 23] = 0.050
    lc_rating_array[landcover_array == 24] = 0.020
    lc_rating_array[landcover_array == 31] = 0.010
    lc_rating_array[landcover_array == 32] = 0.010
    lc_rating_array[landcover_array == 41] = 0.200
    lc_rating_array[landcover_array == 42] = 0.250
    lc_rating_array[landcover_array == 43] = 0.250
    lc_rating_array[landcover_array == 51] = 0.150
    lc_rating_array[landcover_array == 52] = 0.150
    lc_rating_array[landcover_array == 71] = 0.400
    lc_rating_array[landcover_array == 72] = 0.700
    lc_rating_array[landcover_array == 73] = 0.300
    lc_rating_array[landcover_array == 74] = 0.300
    lc_rating_array[landcover_array == 81] = 1.000
    lc_rating_array[landcover_array == 82] = 1.000
    lc_rating_array[landcover_array == 90] = 0.700
    lc_rating_array[landcover_array == 94] = 0.200
    lc_rating_array[landcover_array == 95] = 0.650
    return lc_rating_array


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='METRIC Pixel Rating',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'workspace', nargs='?', default=os.getcwd(),
        help='Landsat scene folder', metavar='FOLDER')
    parser.add_argument(
        '-i', '--ini', required=True,
        help='Pixel regions input file', metavar='PATH')
    parser.add_argument(
        '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    parser.add_argument(
        '--delay', default=0, type=int, metavar='N',
        help='Max random delay starting job in seconds')
    parser.add_argument(
        '-o', '--overwrite', default=None, action="store_true",
        help='Force overwrite of existing files')
    parser.add_argument(
        '--no_file_logging', default=False, action="store_true",
        help='Turn off file logging')
    parser.add_argument(
        '--stats', default=False, action="store_true",
        help='Compute raster statistics')
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
        log_file_name = 'pixel_rating_log.txt'
        log_file = logging.FileHandler(
            os.path.join(args.workspace, log_file_name), "w")
        log_file.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        log_file.setFormatter(formatter)
        logger.addHandler(log_file)

    logging.info('\n{0}'.format('#' * 80))
    log_fmt = '{0:<20s} {1}'
    logging.info(log_fmt.format(
        'Run Time Stamp:', datetime.now().isoformat(' ')))
    logging.info(log_fmt.format('Current Directory:', args.workspace))
    logging.info(log_fmt.format('Script:', os.path.basename(sys.argv[0])))
    logging.info('')

    # Delay
    sleep(random.uniform(0, max([0, args.delay])))

    # Pixel Rating
    pixel_rating(
        image_ws=args.workspace, config_file=args.ini,
        stats_flag=args.stats, overwrite_flag=args.overwrite)
