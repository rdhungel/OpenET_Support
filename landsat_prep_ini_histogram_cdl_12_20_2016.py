#!/usr/bin/env python
#--------------------------------
# Name:         landsat_prep_ini.py
# Purpose:      Prepare Landsat path/row data
# Author:       Charles Morton/Ramesh Dhungel
# Created       2016-09-01
# Python:       2.7
#--------------------------------

import argparse
from collections import defaultdict
import ConfigParser
from datetime import datetime
import logging
import os
# import re
import shutil
import sys

# import numpy as np
from osgeo import gdal, ogr, osr

import gdal_common as gdc
from python_common import read_param, parse_int_set


def main(ini_path, tile_list=None, overwrite_flag=False):
    """Prep Landsat path/row specific data

    Args:
        ini_path (str): file path of the input parameters file
        tile_list (list): list of Landsat path/row (i.e. [p45r43, p45r33])
            This will override the tile list in the INI file
        overwrite_flag (bool): boolean, overwrite existing files
        mp_procs (int): number of cores to use

    Returns:
        None
    """
    logging.info('\nPrepare path/row INI files')

    # Open config file
    logging.info('  {0:<18s} {1}'.format(
        'INI File:', os.path.basename(ini_path)))
    config = ConfigParser.ConfigParser()
    try:
        config.readfp(open(ini_path))
    except IOError:
        logging.error(('\nERROR: Config file does not exist\n' +
                       '  {0}\n').format(ini_path))
        sys.exit()
    except ConfigParser.MissingSectionHeaderError:
        logging.error('\nERROR: Input file is missing a section header\n' +
                      '    Please make sure the following line is at the ' +
                      'beginning of the file\n[INPUTS]\n')
        sys.exit()
    except:
        logging.error(('\nERROR: Unknown error reading input file\n' +
                       '  {0}\n').format(ini_path))
        sys.exit()

    # Get input parameters
    year = config.getint('INPUTS', 'year')
    if tile_list is None:
        tile_list = read_param('tile_list', [], config, 'INPUTS')
    project_ws = config.get('INPUTS', 'project_folder')
    logging.debug('  Year: {}'.format(year))
    logging.debug('  Path/rows: {}'.format(', '.join(tile_list)))
    logging.debug('  Project: {}'.format(project_ws))

    ini_file_flag = read_param('ini_file_flag', True, config, 'INPUTS')
    landsat_flag = read_param('landsat_flag', True, config, 'INPUTS')
    ledaps_flag = read_param('ledaps_flag', False, config, 'INPUTS')
    dem_flag = read_param('dem_flag', True, config, 'INPUTS')
    nlcd_flag = read_param('nlcd_flag', True, config, 'INPUTS')
    modis_lc_flag = read_param('modis_lc_flag', False, config, 'INPUTS')
    cdl_flag = read_param('cdl_flag', True, config, 'INPUTS')
    landfire_flag = read_param('landfire_flag', False, config, 'INPUTS')
    field_flag = read_param('field_flag', False, config, 'INPUTS')
    metric_flag = read_param('metric_flag', True, config, 'INPUTS')
    ssebop_flag = read_param('ssebop_flag', False, config, 'INPUTS')
    monte_carlo_flag = read_param('monte_carlo_flag', False, config, 'INPUTS')
    interp_rasters_flag = read_param(
        'interpolate_rasters_flag', False, config, 'INPUTS')
    interp_tables_flag = read_param(
        'interpolate_tables_flag', False, config, 'INPUTS')

    metric_hourly_weather = read_param(
        'metric_hourly_weather', 'NLDAS', config, 'INPUTS')

    project_ws = config.get('INPUTS', 'project_folder')
    footprint_path = config.get('INPUTS', 'footprint_path')
    skip_list_path = read_param('skip_list_path', '', config, 'INPUTS')

    # tile_gcs_buffer = read_param('tile_buffer', 0.1, config)

    # Template input files for scripts
    if metric_flag:
        metric_ini = config.get('INPUTS', 'metric_ini')
        pixel_rating_ini = config.get('INPUTS', 'pixel_rating_ini')
    if ssebop_flag:
        ssebop_ini = config.get('INPUTS', 'ssebop_ini')
    if monte_carlo_flag:
        monte_carlo_ini = config.get('INPUTS', 'monte_carlo_ini')

    if interp_rasters_flag or interp_tables_flag:
        interpolate_folder = read_param('interpolate_folder', 'ET', config)
        interpolate_ini = config.get('INPUTS', 'interpolate_ini')
    if interp_rasters_flag:
        study_area_path = config.get('INPUTS', 'study_area_path')
        study_area_mask_flag = read_param('study_area_mask_flag', True, config)
        study_area_snap = read_param('study_area_snap', (0, 0), config)
        study_area_cellsize = read_param('study_area_cellsize', 30, config)
        study_area_buffer = read_param('study_area_buffer', 0, config)
        study_area_proj = read_param('study_area_proj', '', config)
    if interp_tables_flag:
        zones_path = config.get('INPUTS', 'zones_path')
        zones_name_field = read_param('zones_name_field', 'FID', config)
        # zones_buffer = read_param('zones_buffer', 0, config)
        zones_snap = read_param('zones_snap', (0, 0), config)
        zones_cellsize = read_param('zones_cellsize', 30, config)
        # zones_proj = read_param('zones_proj', '', config)
        zones_mask = read_param('zones_mask', None, config)
        zones_buffer = None
        zones_proj = None

    # Input/output folder and file paths
    if landsat_flag:
        landsat_input_ws = config.get('INPUTS', 'landsat_input_folder')
    else:
        landsat_input_ws = None
    if ledaps_flag:
        ledaps_input_ws = config.get('INPUTS', 'ledaps_input_folder')
    else:
        ledaps_input_ws = None

    if dem_flag:
        dem_input_ws = config.get('INPUTS', 'dem_input_folder')
        dem_tile_fmt = config.get('INPUTS', 'dem_tile_fmt')
        dem_output_ws = config.get('INPUTS', 'dem_output_folder')
        dem_output_name = read_param(
            'dem_output_name', 'dem.img', config)
        # dem_output_name = config.get('INPUTS', 'dem_output_name')
    else:
        dem_input_ws, dem_tile_fmt = None, None
        dem_output_ws, dem_output_name = None, None

    if nlcd_flag:
        nlcd_input_path = config.get('INPUTS', 'nlcd_input_path')
        nlcd_output_ws = config.get('INPUTS', 'nlcd_output_folder')
        nlcd_output_fmt = read_param(
            'nlcd_output_fmt', 'nlcd_{0:04d}.img', config)
    else:
        nlcd_input_path, nlcd_output_ws, nlcd_output_fmt = None, None, None

    if modis_lc_flag:
        modis_lc_input_path = config.get('INPUTS', 'modis_lc_input_folder')
        modis_lc_output_ws = config.get('INPUTS', 'modis_lc_output_folder')
        modis_lc_output_name = read_param(
            'modis_lc_output_name', 'modis_lc.img', config)
    else:
        modis_lc_input_path = None
        modis_lc_output_ws, modis_lc_output_name = None, None

    if cdl_flag:
        cdl_input_path = config.get('INPUTS', 'cdl_input_path')
        cdl_ag_list = config.get('INPUTS', 'cdl_ag_list')
        cdl_ag_list = list(parse_int_set(cdl_ag_list))
        # default_cdl_ag_list = range(1,62) + range(66,78) + range(204,255)
        # cdl_ag_list = read_param(
        #    'cdl_ag_list', default_cdl_ag_list, config)
        # cdl_ag_list = map(int, cdl_ag_list)
        # cdl_non_ag_list = read_param(
        #    'cdl_non_ag_list', [], config)
        cdl_output_ws = config.get('INPUTS', 'cdl_output_folder')
        cdl_output_fmt = read_param(
            'cdl_output_fmt', 'cdl_{0:04d}.img', config)
        cdl_ag_output_fmt = read_param(
            'cdl_ag_output_fmt', 'cdl_ag_{0:04d}.img', config)
    else:
        cdl_input_path, cdl_ag_list = None, None
        cdl_output_ws, cdl_output_fmt, cdl_ag_output_fmt = None, None, None

    if landfire_flag:
        landfire_input_path = config.get('INPUTS', 'landfire_input_path')
        landfire_ag_list = config.get('INPUTS', 'landfire_ag_list')
        landfire_ag_list = list(parse_int_set(landfire_ag_list))
        # default_landfire_ag_list = range(3960,4000)
        # landfire_ag_list = read_param(
        #    'landfire_ag_list', default_landfire_ag_list, config)
        # landfire_ag_list = map(int, landfire_ag_list)
        landfire_output_ws = config.get('INPUTS', 'landfire_output_folder')
        landfire_output_fmt = read_param(
            'landfire_output_fmt', 'landfire_{0:04d}.img', config)
        landfire_ag_output_fmt = read_param(
            'landfire_ag_output_fmt', 'landfire_ag_{0:04d}.img', config)
    else:
        landfire_input_path, landfire_ag_list = None, None
        landfire_output_ws = None
        landfire_output_fmt, landfire_ag_output_fmt = None, None

    if field_flag:
        field_input_path = config.get('INPUTS', 'field_input_path')
        field_output_ws = config.get('INPUTS', 'field_output_folder')
        field_output_fmt = read_param(
            'field_output_fmt', 'fields_{0:04d}.img', config)
    else:
        field_input_path = None
        field_output_ws, field_output_fmt = None, None

    if monte_carlo_flag:
        etrf_training_path = config.get('INPUTS', 'etrf_training_path')
        # mc_iter_list = config.get('INPUTS', 'mc_iter_list')
        # mc_iter_list = list(parse_int_set(mc_iter_list))
    if monte_carlo_flag or interp_rasters_flag or interp_tables_flag:
        etrf_input_ws = read_param('etrf_input_folder', None, config)
        # if etrf_input_ws is None:
        #     etrf_input_ws = os.path.join(project_ws, year)
        etr_input_ws = config.get('INPUTS', 'etr_input_folder')
        ppt_input_ws = config.get('INPUTS', 'ppt_input_folder')
        etr_input_re = config.get('INPUTS', 'etr_input_re')
        ppt_input_re = config.get('INPUTS', 'ppt_input_re')
    if monte_carlo_flag or interp_rasters_flag or interp_tables_flag:
        awc_input_path = config.get('INPUTS', 'awc_input_path')
        spinup_days = read_param('swb_spinup_days', 30, config, 'INPUTS')
        min_spinup_days = read_param(
            'swb_min_spinup_days', 5, config, 'INPUTS')

    # Weather data parameters
    if metric_flag:
        metric_hourly_weather_list = ['NLDAS', 'REFET']
        metric_hourly_weather = config.get(
            'INPUTS', 'metric_hourly_weather').upper()
        if metric_hourly_weather not in metric_hourly_weather_list:
            logging.error(
                ('\nERROR: The METRIC hourly weather type {0} is invalid.' +
                 '\nERROR: Set metric_hourly_weather to {1}').format(
                    metric_hourly_weather,
                    ','.join(metric_hourly_weather_list)))
            sys.exit()
        elif metric_hourly_weather == 'REFET':
            refet_params_path = os.path.normpath(
                config.get('INPUTS', 'refet_params_path'))
        elif metric_hourly_weather == 'NLDAS':
            # metric_hourly_re = config.get('INPUTS', 'metric_hourly_re')
            # metric_daily_re = config.get('INPUTS', 'metric_daily_re')
            metric_ea_input_ws = config.get('INPUTS', 'metric_ea_input_folder')
            metric_wind_input_ws = config.get(
                'INPUTS', 'metric_wind_input_folder')
            metric_etr_input_ws = config.get(
                'INPUTS', 'metric_etr_input_folder')
            try:
                calc_metric_tair_flag = config.getboolean(
                    'INPUTS', 'calc_metric_tair_flag')
                metric_tair_input_ws = config.get(
                    'INPUTS', 'metric_tair_input_folder')
            except:
                calc_metric_tair_flag = False
                metric_tair_input_ws = ''
    if ssebop_flag:
        # ssebop_daily_re = config.get('INPUTS', 'ssebop_daily_re')
        ssebop_ea_input_ws = config.get('INPUTS', 'ssebop_ea_input_folder')
        ssebop_tmin_input_ws = config.get('INPUTS', 'ssebop_tmin_input_folder')
        ssebop_tmax_input_ws = config.get('INPUTS', 'ssebop_tmax_input_folder')
        ssebop_etr_input_ws = config.get('INPUTS', 'ssebop_etr_input_folder')
        # ssebop_wind_input_ws = config.get('INPUTS', 'ssebop_wind_input_folder')

    # Check inputs folders/paths
    logging.info('\nChecking input folders/files')
    file_check(footprint_path)
    if landsat_flag:
        folder_check(landsat_input_ws)
    if ledaps_flag:
        folder_check(ledaps_input_ws)
    if dem_flag:
        folder_check(dem_input_ws)
    if nlcd_flag:
        file_check(nlcd_input_path)
    if cdl_flag:
        file_check(cdl_input_path)
    if landfire_flag:
        # Landfire will likely be an ESRI grid (set as a folder)
        if not (os.path.isdir(landfire_input_path) or
                os.path.isfile(landfire_input_path)):
            logging.error('  {0} does not exist.'.format(
                landfire_input_path))
            sys.exit()
    if field_flag:
        file_check(field_input_path)
    if metric_flag:
        file_check(metric_ini)
        file_check(pixel_rating_ini)
    if ssebop_flag:
        file_check(ssebop_ini)
    if interp_rasters_flag or interp_tables_flag or monte_carlo_flag:
        if etrf_input_ws is not None:
            folder_check(etrf_input_ws)
        folder_check(etr_input_ws)
        folder_check(ppt_input_ws)
        file_check(awc_input_path)
    if monte_carlo_flag:
        file_check(monte_carlo_ini)
        file_check(etrf_training_path)
    if metric_flag:
        if metric_hourly_weather == 'REFET':
            file_check(refet_params_path)
        elif metric_hourly_weather == 'NLDAS':
            folder_check(metric_ea_input_ws)
            folder_check(metric_wind_input_ws)
            folder_check(metric_etr_input_ws)
            if calc_metric_tair_flag:
                folder_check(metric_tair_input_ws)
    if ssebop_flag:
        folder_check(ssebop_ea_input_ws)
        folder_check(ssebop_tmin_input_ws)
        folder_check(ssebop_tmax_input_ws)
        folder_check(ssebop_etr_input_ws)
        # folder_check(ssebop_wind_input_ws)
    if skip_list_path:
        file_check(skip_list_path)

    # Build output folders
    if not os.path.isdir(project_ws):
        os.makedirs(project_ws)

    # For now assume path/row are two digit numbers
    tile_fmt = 'p{0:02d}r{1:02d}'

    # Set snap environment parameters
    snap_cs = 30
    snap_xmin, snap_ymin = (15, 15)
    env = gdc.env
    env.cellsize = snap_cs
    env.snap_xmin, env.snap_ymin = snap_xmin, snap_ymin


    # Use WGSS84 (EPSG 4326) for GCS spatial reference
    # Could also use NAD83 (EPSG 4269)
    # gcs_epsg = 4326
    # gcs_osr = epsg_osr(4326)
    # gcs_proj = osr_proj(gcs_osr)

    # Landsat Footprints (WRS2 Descending Polygons)
    logging.debug('\nFootprint (WRS2 descending should be GCS84):')
    tile_gcs_osr = gdc.feature_path_osr(footprint_path)
    logging.debug('  OSR: {0}'.format(tile_gcs_osr))

    # Doublecheck that WRS2 descending shapefile is GCS84
    # if tile_gcs_osr != epsg_osr(4326):
    #     logging.error('  WRS2 is not GCS84')
    #     sys.exit()

    # Get geometry for each path/row
    tile_gcs_wkt_dict = path_row_wkt_func(
        footprint_path, path_field='PATH', row_field='ROW')
    # Get UTM zone for each path/row
    tile_utm_zone_dict = path_row_utm_zone_func(
        footprint_path, zone_field='UTM_ZONE',
        path_field='PATH', row_field='ROW')

    # Check that each path/row extent and UTM zone exist
    logging.info('\nChecking path/row list against footprint shapefile')
    for tile_name in sorted(tile_list):
        if tile_name not in tile_gcs_wkt_dict.keys():
            logging.error(
                '  {0} feature not in footprint shapefile'.format(tile_name))
            continue
        elif tile_name not in tile_utm_zone_dict.keys():
            logging.error(
                '  {0} UTM zone not in footprint shapefile'.format(tile_name))
            continue
        elif tile_utm_zone_dict[tile_name] == 0:
            logging.error((
                '  UTM zone is not set for {0} in ' +
                'footprint shapefile').format(tile_name))
            continue

    # Read RefET parameters
    if metric_hourly_weather == 'REFET':
        refet_ws = os.path.dirname(refet_params_path)
        with open(refet_params_path, 'r') as input_f:
            lines = input_f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line.split(',') for line in lines if line]
        columns = lines.pop(0)
        refet_params_dict = defaultdict(dict)
        for line in lines:
            tile_name = tile_fmt.format(
                int(line[columns.index('PATH')]),
                int(line[columns.index('ROW')]))
            yr_tile_name = '{0}_{1}'.format(
                line[columns.index('YEAR')], tile_name)
            for i, column in enumerate(columns):
                if column not in ['YEAR', 'PATH', 'ROW']:
                    refet_params_dict[yr_tile_name][column.lower()] = line[i]

    # Process input files for each year and path/row
    logging.info('\nBuilding path/row specific input files')
    for tile_name in tile_list:
        tile_output_ws = os.path.join(project_ws, str(year), tile_name)
        logging.info('{0} {1}'.format(year, tile_name))
        yr_tile_name = '{0}_{1}'.format(year, tile_name)
        if not os.path.isdir(tile_output_ws):
            os.makedirs(tile_output_ws)

        # File paths
        if metric_flag:
            tile_metric_ini = os.path.join(
                tile_output_ws, os.path.basename(metric_ini).replace(
                    '.ini', '_{0}_{1}.ini'.format(year, tile_name)))
            tile_pixel_rating_ini = os.path.join(
                tile_output_ws, os.path.basename(pixel_rating_ini).replace(
                    '.ini', '_{0}_{1}.ini'.format(year, tile_name)))
            if overwrite_flag and os.path.isfile(tile_metric_ini):
                os.remove(tile_metric_ini)
            if overwrite_flag and os.path.isfile(tile_pixel_rating_ini):
                os.remove(tile_pixel_rating_ini)
        if ssebop_flag:
            tile_ssebop_ini = os.path.join(
                tile_output_ws, os.path.basename(ssebop_ini).replace(
                    '.ini', '_{0}_{1}.ini'.format(year, tile_name)))
            if overwrite_flag and os.path.isfile(tile_ssebop_ini):
                os.remove(tile_ssebop_ini)

        # Monte Carlo is independent of tile and year, but process
        #   with METRIC input file
        if monte_carlo_flag:
            tile_monte_carlo_ini = os.path.join(
                tile_output_ws, os.path.basename(monte_carlo_ini).replace(
                    '.ini', '_{0}_{1}.ini'.format(year, tile_name)))
            if overwrite_flag and os.path.isfile(tile_monte_carlo_ini):
                os.remove(tile_monte_carlo_ini)

        if dem_flag:
            dem_output_path = os.path.join(
                dem_output_ws, tile_name, dem_output_name)
        if nlcd_flag:
            nlcd_output_path = os.path.join(
                nlcd_output_ws, tile_name, nlcd_output_fmt.format(year))
            if modis_lc_flag:
                modis_lc_output_path = os.path.join(
                    modis_lc_output_ws, tile_name,
                    '{1}_{0}{2}'.format(year, *os.path.splitext(modis_lc_output_name)))
        if cdl_flag:
            cdl_ag_output_path = os.path.join(
                cdl_output_ws, tile_name, cdl_ag_output_fmt.format(year))
            print (cdl_ag_output_path)

        if cdl_flag:
            cdl_output_path = os.path.join(
                cdl_output_ws, tile_name, cdl_output_fmt.format(year))
            print (cdl_output_path)

        if landfire_flag:
            landfire_ag_output_path = os.path.join(
                landfire_output_ws, tile_name, landfire_output_fmt.format(year))
        if field_flag:
            field_output_path = os.path.join(
                field_output_ws, tile_name, field_output_fmt.format(year))

        # Check that the path/row was in the RefET parameters file
        if (metric_flag and
                metric_hourly_weather == 'REFET' and
                yr_tile_name not in refet_params_dict.keys()):
            logging.error(
                ('    The year {0} & path/row {1} is not in the ' +
                 'RefET parameters csv, skipping').format(year, tile_name))
            continue

        if metric_flag and not os.path.isfile(tile_metric_ini):
            # DEADBEEF - This approach removes all formatting and comments
            config = ConfigParser.RawConfigParser()
            config.read(metric_ini)
            # shutil.copy(metric_ini, tile_metric_ini)
            # config.read(tile_metric_ini)

            if metric_hourly_weather == 'REFET':
                # Add RefET options
                config.set('INPUTS', 'weather_data_source', 'REFET')
                config.set(
                    'INPUTS', 'refet_file',
                    os.path.join(
                        refet_ws, os.path.normpath(
                            refet_params_dict[yr_tile_name]['refet_file'])))
                config.set(
                    'INPUTS', 'gmt_offset',
                    refet_params_dict[yr_tile_name]['gmt_offset'])
                config.set(
                    'INPUTS', 'datum',
                    refet_params_dict[yr_tile_name]['datum'])
            elif metric_hourly_weather == 'NLDAS':
                # Add NLDAS options
                config.set('INPUTS', 'weather_data_source', 'NLDAS')
                # Remove RefET options
                try:
                    config.remove_option('INPUTS', 'refet_file')
                except:
                    pass
                try:
                    config.remove_option('INPUTS', 'gmt_offset')
                except:
                    pass
                #try: config.remove_option('INPUTS', 'datum')
                #except: pass

            if dem_flag:
                config.set('INPUTS', 'dem_raster', dem_output_path)
            else:
                try:
                    config.remove_option('INPUTS', 'dem_raster')
                except:
                    pass
                # config.set('INPUTS', 'dem_raster', 'None')

            if nlcd_flag:
                config.set('INPUTS', 'landuse_raster', nlcd_output_path)
            else:
                try:
                    config.remove_option('INPUTS', 'landuse_raster')
                except:
                    pass
                # config.set('INPUTS', 'landuse_raster', 'None')

            if modis_lc_flag:
                config.set('INPUTS', 'landuse_raster', modis_lc_output_path)
            else:
                try:
                    config.remote_option('INPUTS', 'landuse_raster')
                except:
                    pass

            logging.debug('  {}'.format(tile_metric_ini))
            with open(tile_metric_ini, 'wb') as config_f:
                config.write(config_f)

        if metric_flag and not os.path.isfile(tile_pixel_rating_ini):
            config = ConfigParser.RawConfigParser()
            config.read(pixel_rating_ini)
            if nlcd_flag:
                config.set('INPUTS', 'landuse_raster', nlcd_output_path)
            elif modis_lc_flag:
                config.set('INPUTS', 'landuse_raster', modis_lc_output_path)
            else:
                try:
                    config.remove_option('INPUTS', 'landuse_raster')
                except:
                    pass
                # config.set('INPUTS', 'landuse_raster', 'None')
            if cdl_flag:
                config.set('INPUTS', 'apply_cdl_ag_mask', True)
                config.set('INPUTS', 'cdl_ag_raster', cdl_ag_output_path)
            else:
                config.set('INPUTS', 'apply_cdl_ag_mask', False)
                try:
                    config.remove_option('INPUTS', 'cdl_ag_raster')
                except:
                    pass

            if cdl_flag:
                config.set('INPUTS', 'apply_cdl_ag_mask', True)
                config.set('INPUTS', 'cdl_raster', cdl_output_path)
            else:
                config.set('INPUTS', 'apply_cdl_ag_mask', False)
                try:
                    config.remove_option('INPUTS', 'cdl_raster')
                except:
                    pass

            if field_flag:
                config.set('INPUTS', 'apply_field_mask', True)
                config.set('INPUTS', 'fields_raster', field_output_path)
            else:
                config.set('INPUTS', 'apply_field_mask', False)
                try:
                    config.remove_option('INPUTS', 'fields_raster')
                except:
                    pass
                # config.set('INPUTS', 'fields_raster', 'None')
            # if landfire_flag:
            #     config.set('INPUTS', 'apply_landfire_ag_mask', True)
            #     config.set('INPUTS', 'landfire_ag_raster', cdl_ag_output_path)
            # else:
            #     config.set('INPUTS', 'apply_landfire_ag_mask', False)
            #     try: config.remove_option('INPUTS', 'landfire_ag_raster')
            #     except: pass
            #     # config.set('INPUTS', 'landfire_ag_raster', 'None')

            logging.debug('  {}'.format(tile_pixel_rating_ini))
            with open(tile_pixel_rating_ini, 'wb') as config_f:
                config.write(config_f)

        if monte_carlo_flag and not os.path.isfile(tile_monte_carlo_ini):
            config = ConfigParser.RawConfigParser()
            config.read(monte_carlo_ini)
            config.set('INPUTS', 'etrf_training_path', etrf_training_path)
            config.set('INPUTS', 'etr_ws', etr_input_ws)
            config.set('INPUTS', 'ppt_ws', ppt_input_ws)
            config.set('INPUTS', 'etr_re', etr_input_re)
            config.set('INPUTS', 'ppt_re', ppt_input_re)
            config.set('INPUTS', 'awc_path', awc_input_path)
            config.set('INPUTS', 'swb_spinup_days', spinup_days)
            config.set('INPUTS', 'swb_min_spinup_days', min_spinup_days)

            logging.debug('  {}'.format(tile_monte_carlo_ini))
            with open(tile_monte_carlo_ini, 'wb') as config_f:
                config.write(config_f)

        if ssebop_flag and not os.path.isfile(tile_ssebop_ini):
            # DEADBEEF - This approach removes all formatting and comments
            config = ConfigParser.RawConfigParser()
            config.read(ssebop_ini)
            # shutil.copy(metric_ini, tile_ssebop_ini)
            # config.read(tile_ssebop_ini)

            # Add GRIDMET options
            # config.set('INPUTS', 'daily_re', ssebop_daily_re)
            # config.set('INPUTS', 'ea_folder', ssebop_ea_output_path)
            # config.set('INPUTS', 'tmin_folder', ssebop_tmin_output_path)
            # config.set('INPUTS', 'tmax_folder', ssebop_tmax_output_path)
            # config.set('INPUTS', 'wind_folder', ssebop_wind_output_path)
            # config.set('INPUTS', 'etr_folder', ssebop_etr_output_path)

            if dem_flag:
                config.set('INPUTS', 'dem_raster', dem_output_path)
            else:
                try:
                    config.remove_option('INPUTS', 'dem_raster')
                except:
                    pass
                # config.set('INPUTS', 'dem_raster', 'None')

            logging.debug('  {}'.format(tile_ssebop_ini))
            with open(tile_ssebop_ini, 'wb') as config_f:
                config.write(config_f)

        # Cleanup
        del tile_output_ws, yr_tile_name

    # Interpolator input file
    if interp_rasters_flag or interp_tables_flag:
        logging.info('\nBuilding interpolator input files')
        year_interpolator_name = os.path.basename(interpolate_ini).replace(
            '.ini', '_{0}_{1}.ini'.format(year, interpolate_folder.lower()))
        year_interpolator_ini = os.path.join(
            project_ws, str(year), year_interpolator_name)
        if overwrite_flag and os.path.isfile(year_interpolator_ini):
            os.remove(year_interpolator_ini)
        if not os.path.isfile(year_interpolator_ini):
            # First copy the template config file to the year folder
            shutil.copy(interpolate_ini, year_interpolator_ini)

            # Open the existing config file and update the values
            # DEADBEEF - This approach removes all formatting and comments
            config = ConfigParser.RawConfigParser()
            config.read(year_interpolator_ini)
            config.set('INPUTS', 'folder_name', interpolate_folder)
            config.set('INPUTS', 'tile_list', ', '.join(tile_list))
            if interp_rasters_flag:
                config.set('INPUTS', 'study_area_path', study_area_path)
                config.set('INPUTS', 'study_area_mask_flag', study_area_mask_flag)
                config.set('INPUTS', 'study_area_snap', ', '.join(map(str, study_area_snap)))
                config.set('INPUTS', 'study_area_cellsize', study_area_cellsize)
                config.set('INPUTS', 'study_area_buffer', study_area_buffer)
                if study_area_proj:
                    config.set('INPUTS', 'study_area_proj', study_area_proj)
                else:
                    try:
                        config.remove_option('INPUTS', 'study_area_proj', study_area_proj)
                    except:
                        pass
            if interp_tables_flag:
                config.set('INPUTS', 'zones_path', zones_path)
                config.set('INPUTS', 'zones_snap', ', '.join(map(str, zones_snap)))
                config.set('INPUTS', 'zones_cellsize', zones_cellsize)
                config.set('INPUTS', 'zones_name_field', zones_name_field)
                # zones_buffer is not currently implemented
                if zones_buffer:
                    config.set('INPUTS', 'zones_buffer', zones_buffer)
                else:
                    try:
                        config.remove_option('INPUTS', 'zones_buffer', zones_buffer)
                    except:
                        pass
                # zones proj., cellsize, and snap are not needed or
                #   read in if zones_mask is set
                # zones_proj is not currently implemented
                if zones_mask:
                    config.set('INPUTS', 'zones_mask', zones_mask)
                    try:
                        config.remove_option('INPUTS', 'zones_proj')
                    except:
                        pass
                    try:
                        config.remove_option('INPUTS', 'zones_cellsize')
                    except:
                        pass
                    try:
                        config.remove_option('INPUTS', 'zones_snap')
                    except:
                        pass
                # elif zones_proj:
                #     config.set('INPUTS', 'zones_proj', zones_proj)
                #     try:
                #         config.remove_option('INPUTS', 'zones_mask')
                #     except:
                #         pass
                else:
                    try:
                        config.remove_option('INPUTS', 'zones_proj')
                    except:
                        pass
                    try:
                        config.remove_option('INPUTS', 'zones_mask')
                    except:
                        pass
            config.set('INPUTS', 'year', year)
            config.set('INPUTS', 'footprint_path', footprint_path)
            if etrf_input_ws is not None:
                config.set('INPUTS', 'etrf_input_folder', etrf_input_ws)
            config.set('INPUTS', 'etr_input_folder', etr_input_ws)
            config.set('INPUTS', 'etr_input_re', etr_input_re)
            config.set('INPUTS', 'ppt_input_folder', ppt_input_ws)
            config.set('INPUTS', 'ppt_input_re', ppt_input_re)
            # DEADBEEF - add check for SWB flag
            config.set('INPUTS', 'awc_input_path', awc_input_path)
            config.set('INPUTS', 'swb_spinup_days', spinup_days)
            config.set('INPUTS', 'swb_min_spinup_days', min_spinup_days)

            logging.debug('  {}'.format(year_interpolator_ini))
            with open(year_interpolator_ini, 'wb') as config_f:
                config.write(config_f)


def path_row_wkt_func(input_path, path_field='PATH', row_field='ROW',
                      tile_fmt='p{0:02d}r{1:02d}'):
    """Return a dictionary of Landsat path/rows and their geometries"""
    output_dict = dict()
    input_ds = ogr.Open(input_path, 0)
    input_lyr = input_ds.GetLayer()
    input_ftr = input_lyr.GetNextFeature()
    while input_ftr:
        path = input_ftr.GetFieldAsInteger(
            input_ftr.GetFieldIndex(path_field))
        row = input_ftr.GetFieldAsInteger(
            input_ftr.GetFieldIndex(row_field))
        input_wkt = input_ftr.GetGeometryRef().ExportToWkt()
        output_dict[tile_fmt.format(path, row)] = input_wkt
        input_ftr = input_lyr.GetNextFeature()
    input_ds = None
    return output_dict

def path_row_utm_zone_func(input_path, zone_field='UTM_ZONE',
                           path_field='PATH', row_field='ROW',
                           tile_fmt='p{0:02d}r{1:02d}'):
    """Return a dictionary of path/rows and their UTM zones"""
    output_dict = dict()
    input_ds = ogr.Open(input_path, 0)
    input_lyr = input_ds.GetLayer()
    input_ftr = input_lyr.GetNextFeature()
    while input_ftr:
        path = input_ftr.GetFieldAsInteger(
            input_ftr.GetFieldIndex(path_field))
        row = input_ftr.GetFieldAsInteger(
            input_ftr.GetFieldIndex(row_field))
        utm_zone = input_ftr.GetFieldAsInteger(
            input_ftr.GetFieldIndex(zone_field))
        output_dict[tile_fmt.format(path, row)] = utm_zone
        input_ftr = input_lyr.GetNextFeature()
    input_ds = None
    return output_dict

def folder_check(folder_path):
    """"""
    if not os.path.isdir(folder_path):
        logging.info('  {0} does not exist.'.format(folder_path))
        return False
    else:
        return True

def file_check(file_path):
    """"""
    if not os.path.isfile(file_path):
        logging.info('  {0} does not exist.'.format(file_path))
        return False
    else:
        return True

def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Batch Landsat path/row prep',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', required=True,
        help='Landsat project input file', metavar='FILE')
    parser.add_argument(
        '-o', '--overwrite', default=False, action="store_true",
        help='Force overwrite of existing files')
    parser.add_argument(
        '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    args = parser.parse_args()

    # Convert relative paths to absolute paths
    if os.path.isfile(os.path.abspath(args.ini)):
        args.ini = os.path.abspath(args.ini)
    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{0}'.format('#'*80))
    logging.info('{0:<20s} {1}'.format(
        'Run Time Stamp:', datetime.now().isoformat(' ')))
    logging.info('{0:<20s} {1}'.format('Current Directory:', os.getcwd()))
    logging.info('{0:<20s} {1}'.format('Script:', os.path.basename(sys.argv[0])))

    main(ini_path=args.ini, overwrite_flag=args.overwrite)
