# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 10:09:31 2022

@author: r41331jc
"""
import logging
import glob
import os
import warnings
from astropy.io import fits
import argparse
import pandas as pd
import numpy as np

import creating_image_functions

def iauname_to_filename(iauname, base_dir):
    return os.path.join(base_dir, iauname[:4], iauname + '.fits')

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--fits-dir', dest='fits_dir', type=str)
    parser.add_argument('--save-dir', dest='save_dir', type=str)
    parser.add_argument('--max-redshift', dest='max_redshift', default = 0.2, type=float)
    parser.add_argument('--step-size', dest='step_size', default = 0.004, type=float)
    
    args = parser.parse_args()
    
    
    if os.path.isdir('/share/nas2'):
        catalog_loc = '/share/nas2/walml/repos/gz-decals-classifiers/data/catalogs/nsa_v1_0_1_mag_cols.parquet'
        ml_safe_loc = '/share/nas2/walml/repos/gz-decals-classifiers/data/catalogs/training_catalogs/dr5_ortho_v2_labelled_catalog.parquet'
        # note that not every galaxy in this catalog has a good image downloaded
    else:
        catalog_loc = 'nsa_v1_0_1_mag_cols.parquet'
        ml_safe_loc = 'dr5_ortho_v2_labelled_catalog.parquet'

    df = pd.read_parquet(catalog_loc, columns= ['iauname', 'redshift'])
    ml_safe = pd.read_parquet(ml_safe_loc, columns=['id_str'])
    # ml_safe['iauname'] = ml_safe['id_str'].apply(lambda x: os.path.basename(x).replace('.jpeg', '').replace('.png', ''))
    logging.info(ml_safe['id_str'])
    df = df[df['iauname'].isin(ml_safe['id_str'])].reset_index(drop=True)  # filter to only galaxies with good images
    assert len(df) > 0

    df = df.sort_values('iauname')

    fits_dir =  args.fits_dir
    #fits_dir = 'samples'
    logging.info('Loading images from {}'.format(fits_dir))
    assert os.path.isdir(fits_dir)

    save_dir = args.save_dir
    #save_dir = 'samples_saved'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    # '/**/*.fits', recursive=True):
    # imgs = {} # Opens dictionary for storing images, like (filename: file contents)
    # filenames = glob.glob(f'{fits_dir}' + '/*.fits')[:10] # operates over all FIT's within the desired directory
    # logging.info(filenames)
    # filenames = list(filenames)[:5]

    logging.info('Galaxies with good images: {}'.format(len(df)))
    df = df.query('redshift < 0.055')

    # TODO refactor out max galaxies
    filenames = list(df['iauname'].apply(lambda x: iauname_to_filename(x, base_dir=fits_dir)))
    logging.info('Filenames: {}'.format(len(filenames)))
    logging.info('Example filename: {}'.format(filenames[0]))

    for original_loc in filenames:  # [50000:]

        try:
            img, hdr = fits.getdata(original_loc, 0, header=True) #Extract FITs data
            valid_data = True
        except Exception:
            # most images will not exist as only a subset of NSA catalog was downloaded
            logging.debug('Invalid fits at {}'.format(original_loc))
            valid_data = False

        if valid_data:

            iauname = os.path.basename(original_loc).replace('.fits','')
            logging.debug(iauname)

            galaxy = df.query(f'iauname == "{iauname}"').squeeze()
            logging.debug(galaxy)
            
            galaxy_redshift = galaxy['redshift']
            logging.debug(galaxy_redshift)
            
            for redshift in np.arange(galaxy_redshift, args.max_redshift, args.step_size):
                scale_factor = redshift/galaxy_redshift
                filename_scale = iauname + '_{0}.jpeg'.format(scale_factor)  # save output image with scale_factor appended
                # file_loc = os.path.join('/share/nas/walml/repos/understanding_galaxies', output_dir_name[1], filename)
                scaled_file_loc = os.path.join(save_dir, filename_scale)
                if not os.path.isfile(scaled_file_loc):
                    _, _, img_scaled = creating_image_functions.photon_counts_from_FITS(img, scale_factor) # Second input is scale factor, changed in parser
                    creating_image_functions.make_jpg_from_corrected_fits(
                        img=img_scaled,
                        loc=scaled_file_loc,
                        size=424)
                else:
                    logging.info('Skipping {}, already exists'.format(scaled_file_loc))
                
            logging.info('Made images for {}'.format(iauname))

    logging.info('Successfully made images - exiting')
