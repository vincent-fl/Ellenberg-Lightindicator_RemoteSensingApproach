# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:34:24 2025

@author: vince
"""

import pandas as pd
import geopandas as gpd

def load_point_data_and_add_light_values(path):
    points = gpd.read_file(path)
    points = points.to_crs("EPSG:2154")

    moisture = pd.read_csv(
        'Data/lightValues.csv',
        sep=';',
        decimal='.',
        header=[0,1]
    )
    moisture.columns = ['_'.join([str(i) for i in col]).strip() for col in moisture.columns]

    ellenberg_clean = moisture[['LIGHT_Taxon', 'France_Average']].copy()
    ellenberg_clean.rename(columns={
        'LIGHT_Taxon': 'species',
        'France_Average': 'L'
    }, inplace=True)

    points = points.merge(ellenberg_clean, on='species', how='left')
    
    return points

if __name__ == "__main__":
    gpkg_path="Data/Somme_All_EPSG2154.gpkg"
    test = load_point_data_and_add_light_values(gpkg_path)
    