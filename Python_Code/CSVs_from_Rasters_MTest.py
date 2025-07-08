# -*- coding: utf-8 -*-
"""
Created on Tue May 20 08:32:09 2025

@author: vince
"""

import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.mask import mask

raster_path = "Data"
points_path_Loz = "Data/Lozere_All_EPSG2154.gpkg"
points_path_Som = "Data/Somme_All_EPSG2154.gpkg"
save_path = "Data"

# ----------- Funktion zum Erstellen von DataFrames aus den Rasterdateien und den Punktdaten -----------
def process_raster_files(raster_path, file_name, points_path, sample_indices=None, n_samples=500000):
    """
    Verarbeitet ein Rasterfile und extrahiert:
    - Eine Stichprobe zufälliger gültiger Pixelwerte
    - Alle Werte, die mit den Punktgeometrien überlappen

    Falls keine sample_indices übergeben werden, werden sie beim ersten Aufruf erzeugt und zurückgegeben.
    """

    # Laden der Punktdaten
    points = gpd.read_file(points_path)

    # Pfad zur TIFF-Datei
    tiff_file_path = f"{raster_path}/{file_name}"

    with rasterio.open(tiff_file_path) as src:
        # Rasterdaten laden und NaN maskieren
        raster_data = src.read(1, masked=True)
        valid_mask = ~np.isnan(raster_data)
        all_values = raster_data[valid_mask].flatten()

        # Indizes erzeugen, falls nicht übergeben
        if sample_indices is None:
            rng = np.random.default_rng(seed=42)
            sample_indices = rng.choice(
                len(all_values), size=min(n_samples, len(all_values)), replace=False
            )
        
        # Stichprobe ziehen
        sampled_all_values = all_values[sample_indices[:min(len(all_values), len(sample_indices))]]
        sampled_all_values_df = pd.DataFrame(sampled_all_values, columns=['values'])

        # Punktüberlappung extrahieren
        out_image, out_transform = mask(src, points.geometry, crop=True, nodata=np.nan)
        overlap_data = out_image[0]
        overlap_valid_mask = ~np.isnan(overlap_data)
        point_values = overlap_data[overlap_valid_mask].flatten()
        point_values_df = pd.DataFrame(point_values, columns=['values'])

    return sampled_all_values_df, point_values_df, sample_indices


# ----------- Aufruf der Funktion -----------
# die Dataframes mit All sind auf 500000 Zufallsstichproben begrenzt, da andernfalls mehrere hundertmillionen Datenpunkte vorhanden wären

# ----- Lozere -----
Loz_EnMean_All, Loz_EnMean_Br, sample_indices_Loz  = process_raster_files(raster_path, "Lozere_Energie_Mean.tif", points_path_Loz)
Loz_EnMax_All, Loz_EnMax_Br, _ = process_raster_files(raster_path, "Lozere_Energie_Max.tif", points_path_Loz, sample_indices=sample_indices_Loz)
Loz_EnMin_All, Loz_EnMin_Br, _ = process_raster_files(raster_path, "Lozere_Energie_Min.tif", points_path_Loz, sample_indices=sample_indices_Loz)
Loz_EnRange_All, Loz_EnRange_Br, _ = process_raster_files(raster_path, "Lozere_Energie_Range.tif", points_path_Loz, sample_indices=sample_indices_Loz)

Loz_SunMean_All, Loz_SunMean_Br, _ = process_raster_files(raster_path, "Lozere_Sonnenstunden_Mean.tif", points_path_Loz, sample_indices=sample_indices_Loz)
Loz_SunMax_All, Loz_SunMax_Br, _ = process_raster_files(raster_path, "Lozere_Sonnenstunden_Max.tif", points_path_Loz, sample_indices=sample_indices_Loz)
Loz_SunMin_All, Loz_SunMin_Br, _ = process_raster_files(raster_path, "Lozere_Sonnenstunden_Min.tif", points_path_Loz, sample_indices=sample_indices_Loz)
Loz_SunRange_All, Loz_SunRange_Br, _ = process_raster_files(raster_path, "Lozere_Sonnenstunden_Range.tif", points_path_Loz, sample_indices=sample_indices_Loz)

Loz_Aspect_All, Loz_Aspect_Br, _ = process_raster_files(raster_path, "Lozere_Hangausrichtung.tif", points_path_Loz, sample_indices=sample_indices_Loz)
Loz_Slope_All, Loz_Slope_Br, _ = process_raster_files(raster_path, "Lozere_Hangneigung.tif", points_path_Loz, sample_indices=sample_indices_Loz)

Loz_DiffH_All, Loz_DiffH_Br, _ = process_raster_files(raster_path, "Lozere_DiffH.tif", points_path_Loz, sample_indices=sample_indices_Loz)

# with different names
Loz_All = pd.DataFrame({
    'Loz_Sun_Mean': Loz_SunMean_All['values'],
    'Loz_Sun_Max': Loz_SunMax_All['values'],
    'Loz_Sun_Min': Loz_SunMin_All['values'],
    'Loz_Sun_Range': Loz_SunRange_All['values'],
    
    'Loz_Energy_Mean': Loz_EnMean_All['values'],
    'Loz_Energy_Max': Loz_EnMax_All['values'],
    'Loz_Energy_Min': Loz_EnMin_All['values'],
    'Loz_Energy_Range': Loz_EnRange_All['values'],
    
    'Loz_Aspect': Loz_Aspect_All['values'],
    'Loz_Slope': Loz_Slope_All['values'],
    
    'Loz_DiffH': Loz_DiffH_All['values'],
})

Loz_Bromus = pd.DataFrame({
    'LozS_Sun_Mean': Loz_SunMean_Br['values'],
    'LozS_Sun_Max': Loz_SunMax_Br['values'],
    'LozS_Sun_Min': Loz_SunMin_Br['values'],
    'LozS_Sun_Range': Loz_SunRange_Br['values'],
    
    'LozS_Energy_Mean': Loz_EnMean_Br['values'],
    'LozS_Energy_Max': Loz_EnMax_Br['values'],
    'LozS_Energy_Min': Loz_EnMin_Br['values'],
    'LozS_Energy_Range': Loz_EnRange_Br['values'],
    
    'LozS_Aspect': Loz_Aspect_Br['values'],
    'LozS_Slope': Loz_Slope_Br['values'],
    
    'LozS_DiffH': Loz_DiffH_Br['values'],
})

path_LozAll = f'{save_path}/Lozere_FullArea.csv'
Loz_All.to_csv(path_LozAll, index=False, header=True, decimal='.', sep=',')
path_LozBr = f'{save_path}/Lozere_Observations.csv'
Loz_Bromus.to_csv(path_LozBr, index=False, header=True, decimal='.', sep=',')

# with similar names
Loz_All_SimNames = pd.DataFrame({
    'Sun_Mean': Loz_SunMean_All['values'],
    'Sun_Max': Loz_SunMax_All['values'],
    'Sun_Min': Loz_SunMin_All['values'],
    'Sun_Range': Loz_SunRange_All['values'],
    
    'Energy_Mean': Loz_EnMean_All['values'],
    'Energy_Max': Loz_EnMax_All['values'],
    'Energy_Min': Loz_EnMin_All['values'],
    'Energy_Range': Loz_EnRange_All['values'],
    
    'Aspect': Loz_Aspect_All['values'],
    'Slope': Loz_Slope_All['values'],
    
    'DiffH': Loz_DiffH_All['values'],
})

Loz_Bromus_SimNames = pd.DataFrame({
    'Sun_Mean': Loz_SunMean_Br['values'],
    'Sun_Max': Loz_SunMax_Br['values'],
    'Sun_Min': Loz_SunMin_Br['values'],
    'Sun_Range': Loz_SunRange_Br['values'],
    
    'Energy_Mean': Loz_EnMean_Br['values'],
    'Energy_Max': Loz_EnMax_Br['values'],
    'Energy_Min': Loz_EnMin_Br['values'],
    'Energy_Range': Loz_EnRange_Br['values'],
    
    'Aspect': Loz_Aspect_Br['values'],
    'Slope': Loz_Slope_Br['values'],
    
    'DiffH': Loz_DiffH_Br['values'],
})

path_LozAll = f'{save_path}/Lozere_FullArea_SimNames.csv'
Loz_All_SimNames.to_csv(path_LozAll, index=False, header=True, decimal='.', sep=',')
path_LozBr = f'{save_path}/Lozere_Observations_SimNames.csv'
Loz_Bromus_SimNames.to_csv(path_LozBr, index=False, header=True, decimal='.', sep=',')

# ----- Somme -----
Som_DiffH_All, Som_DiffH_Br, sample_indices_Som = process_raster_files(raster_path, "Somme_DiffH.tif", points_path_Som)

Som_EnMean_All, Som_EnMean_Br, _ = process_raster_files(raster_path, "Somme_Energie_Mean.tif", points_path_Som, sample_indices=sample_indices_Som)
Som_EnMax_All, Som_EnMax_Br, _ = process_raster_files(raster_path, "Somme_Energie_Max.tif", points_path_Som, sample_indices=sample_indices_Som)
Som_EnMin_All, Som_EnMin_Br, _ = process_raster_files(raster_path, "Somme_Energie_Min.tif", points_path_Som, sample_indices=sample_indices_Som)
Som_EnRange_All, Som_EnRange_Br, _ = process_raster_files(raster_path, "Somme_Energie_Range.tif", points_path_Som, sample_indices=sample_indices_Som)

Som_SunMean_All, Som_SunMean_Br, _ = process_raster_files(raster_path, "Somme_Sonnenstunden_Mean.tif", points_path_Som, sample_indices=sample_indices_Som)
Som_SunMax_All, Som_SunMax_Br, _ = process_raster_files(raster_path, "Somme_Sonnenstunden_Max.tif", points_path_Som, sample_indices=sample_indices_Som)
Som_SunMin_All, Som_SunMin_Br, _ = process_raster_files(raster_path, "Somme_Sonnenstunden_Min.tif", points_path_Som, sample_indices=sample_indices_Som)
Som_SunRange_All, Som_SunRange_Br, _ = process_raster_files(raster_path, "Somme_Sonnenstunden_Range.tif", points_path_Som, sample_indices=sample_indices_Som)

Som_Aspect_All, Som_Aspect_Br, _ = process_raster_files(raster_path, "Somme_Hangausrichtung.tif", points_path_Som, sample_indices=sample_indices_Som)
Som_Slope_All, Som_Slope_Br, _ = process_raster_files(raster_path, "Somme_Hangneigung.tif", points_path_Som, sample_indices=sample_indices_Som)

# with different names
Som_All = pd.DataFrame({
    'Som_Sun_Mean': Som_SunMean_All['values'],
    'Som_Sun_Max': Som_SunMax_All['values'],
    'Som_Sun_Min': Som_SunMin_All['values'],
    'Som_Sun_Range': Som_SunRange_All['values'],
    
    'Som_Energy_Mean': Som_EnMean_All['values'],
    'Som_Energy_Max': Som_EnMax_All['values'],
    'Som_Energy_Min': Som_EnMin_All['values'],
    'Som_Energy_Range': Som_EnRange_All['values'],
    
    'Som_Aspect': Som_Aspect_All['values'],
    'Som_Slope': Som_Slope_All['values'],
    
    'Som_DiffH': Som_DiffH_All['values'],
})

Som_Bromus = pd.DataFrame({
    'SomS_Sun_Mean': Som_SunMean_Br['values'],
    'SomS_Sun_Max': Som_SunMax_Br['values'],
    'SomS_Min': Som_SunMin_Br['values'],
    'SomS_Sun_Range': Som_SunRange_Br['values'],
    
    'SomS_Energy_Mean': Som_EnMean_Br['values'],
    'SomS_Energy_Max': Som_EnMax_Br['values'],
    'SomS_Energy_Min': Som_EnMin_Br['values'],
    'SomS_Energy_Range': Som_EnRange_Br['values'],
    
    'SomS_Aspect': Som_Aspect_Br['values'],
    'SomS_Slope': Som_Slope_Br['values'],
    
    'SomS_DiffH': Som_DiffH_Br['values'],
})

path_SomAll = f'{save_path}/Somme_FullArea.csv'
Som_All.to_csv(path_SomAll, index=False, header=True, decimal='.', sep=',')
path_SomBr = f'{save_path}/Somme_Observations.csv'
Som_Bromus.to_csv(path_SomBr, index=False, header=True, decimal='.', sep=',')

# with similar names
Som_All_SimNames = pd.DataFrame({
    'Sun_Mean': Som_SunMean_All['values'],
    'Sun_Max': Som_SunMax_All['values'],
    'Sun_Min': Som_SunMin_All['values'],
    'Sun_Range': Som_SunRange_All['values'],
    
    'Energy_Mean': Som_EnMean_All['values'],
    'Energy_Max': Som_EnMax_All['values'],
    'Energy_Min': Som_EnMin_All['values'],
    'Energy_Range': Som_EnRange_All['values'],
    
    'Aspect': Som_Aspect_All['values'],
    'Slope': Som_Slope_All['values'],
    
    'DiffH': Som_DiffH_All['values'],
})

Som_Bromus_SimNames = pd.DataFrame({
    'Sun_Mean': Som_SunMean_Br['values'],
    'Sun_Max': Som_SunMax_Br['values'],
    'Sun_Min': Som_SunMin_Br['values'],
    'Sun_Range': Som_SunRange_Br['values'],
    
    'Energy_Mean': Som_EnMean_Br['values'],
    'Energy_Max': Som_EnMax_Br['values'],
    'Energy_Min': Som_EnMin_Br['values'],
    'Energy_Range': Som_EnRange_Br['values'],
    
    'Aspect': Som_Aspect_Br['values'],
    'Slope': Som_Slope_Br['values'],
    
    'DiffH': Som_DiffH_Br['values'],
})

path_SomAll = f'{save_path}/Somme_FullArea_SimNames.csv'
Som_All_SimNames.to_csv(path_SomAll, index=False, header=True, decimal='.', sep=',')
path_SomBr = f'{save_path}/Somme_Observations_SimNames.csv'
Som_Bromus_SimNames.to_csv(path_SomBr, index=False, header=True, decimal='.', sep=',')
