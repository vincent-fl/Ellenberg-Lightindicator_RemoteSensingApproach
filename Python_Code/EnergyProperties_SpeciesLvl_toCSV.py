# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:35:09 2025

@author: vince
"""

from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
import rasterio
from rasterstats import point_query

from Python_Code.addLightVals import load_point_data_and_add_light_values
from Python_Code.DensityByLightValue import bin_light_values

def extract_raster_values_to_gdf(gdf, raster_dict):
    """
    Extracts raster values from one or more GeoTIFFs and attaches them to a GeoDataFrame.
    
    Parameters:
        gdf (GeoDataFrame): A GeoDataFrame containing point geometries.
        raster_dict (dict): A dictionary mapping new column names to raster file paths,
                            e.g., {'sun': 'sunlight.tif', 'diff': 'diffH.tif'}.
    
    Returns:
        GeoDataFrame: A copy of the input GeoDataFrame with new columns for each raster.
    """
    # Make a copy to avoid modifying the original data
    gdf_out = gdf.copy()

    for colname, raster_path in raster_dict.items():
        with rasterio.open(raster_path) as src:
            # Reproject GeoDataFrame to match raster CRS if needed
            if gdf_out.crs != src.crs:
                gdf_out = gdf_out.to_crs(src.crs)
            
        # Sample raster values at point locations
        gdf_out[colname] = point_query(gdf_out.geometry, raster_path)

    return gdf_out

def clean_xy_for_plotting(df, x_col='sun', y_col='diff'):
    """
    Removes rows with NaN or infinite values in x and y columns.
    """
    df_clean = df.copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna(subset=[x_col, y_col])
    return df_clean

def extract_species_light_niche(gdf, species_col='species', light_col='L'):
    """
    Extracts quantitative niche characteristics (centroid, peak density, density coverage)
    for each species-light class combination based on environmental variables (sun and diff).
    
    Parameters:
    - gdf: GeoDataFrame with observation data including species, light class, and raster values.
    - species_col: column name for species identity.
    - light_col: column name for light class (categorical or numerical bins).

    Returns:
    - DataFrame with centroid, density peak, and coverage stats for each group.
    """
    results = []

    for species in gdf[species_col].dropna().unique():
        species_df = gdf[gdf[species_col] == species]
        
        for l_val in species_df[light_col].dropna().unique():
            bin_df = species_df[species_df[light_col] == l_val]

            # Clean the data
            bin_df = clean_xy_for_plotting(bin_df, x_col='sun', y_col='diff')
            
            if len(bin_df) < 50:
                continue  # Not enough data for KDE

            x = bin_df['sun'].values
            y = bin_df['diff'].values
            xy = np.vstack([x, y])

            # Kernel Density Estimation
            kde = gaussian_kde(xy)

            # Create evaluation grid
            xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
            zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)

            # Centroid (unweighted average of points)
            centroid_x = np.average(x)
            centroid_y = np.average(y)

            # KDE peak value
            peak_density = zi.max()
            
            # Coordinates of peak density
            peak_index = np.unravel_index(np.argmax(zi), zi.shape)
            peak_x = xi[peak_index]
            peak_y = yi[peak_index]
            
            # Distance peak density to centroid
            peak_centroid_distance = np.sqrt((peak_x - centroid_x)**2 + (peak_y - centroid_y)**2)
            
            # Density area coverage: compute cumulative density function
            zi_flat = zi.ravel()
            sorted_zi = np.sort(zi_flat)[::-1]
            cumsum = np.cumsum(sorted_zi)
            cumsum /= cumsum[-1]  # Normalize to 1

            # Determine area coverage (percentage of grid points within top 68% and 95% density)
            area_68 = (cumsum <= 0.68).sum() / len(cumsum)
            area_95 = (cumsum <= 0.95).sum() / len(cumsum)
            
            # Get actual density thresholds corresponding to 68% and 95% cumulative mass
            threshold_68 = sorted_zi[(cumsum <= 0.68).sum()]
            
            # Create a boolean mask for the 68% density region
            mask_68 = zi >= threshold_68
            
            # Get the coordinate values within the 68% region
            x_vals_68 = xi[mask_68]
            y_vals_68 = yi[mask_68]
            
            # Calculate width and height of the bounding box around the 68% region
            width_68 = x_vals_68.max() - x_vals_68.min()
            height_68 = y_vals_68.max() - y_vals_68.min()
            aspect_ratio_68 = height_68 / width_68 if width_68 != 0 else np.nan
            
            # Ratio of area coverage between 68% and 95% regions
            area_ratio_68_95 = area_68 / area_95 if area_95 != 0 else np.nan
            
            # Append results
            results.append({
                'species': species,
                'light_class': l_val,
                'centroid_sun': centroid_x,
                'centroid_diff': centroid_y,
                'peak_density': peak_density,
                'peak_x': peak_x,
                'peak_y': peak_y,
                'peak_centroid_distance': peak_centroid_distance,
                'area_68': area_68,
                'area_95': area_95,
                'width_68': width_68,
                'height_68': height_68,
                'aspect_ratio_68': aspect_ratio_68,
                'area_ratio_68_95': area_ratio_68_95
            })

    return pd.DataFrame(results)


def extract_lightbin_niche(gdf, lightbin_col='light_bin'):
    """
    Extracts quantitative niche characteristics (centroid, peak density, density coverage)
    for each light bin based on environmental variables (sun and diff).

    Parameters:
    - gdf: GeoDataFrame with observation data including light bin and raster values.
    - lightbin_col: column name for the light bin (numerical).

    Returns:
    - DataFrame with centroid, density peak, and coverage stats for each light bin.
    """
    results = []

    for l_bin in gdf[lightbin_col].dropna().unique():
        bin_df = gdf[gdf[lightbin_col] == l_bin]

        # Clean the data
        bin_df = clean_xy_for_plotting(bin_df, x_col='sun', y_col='diff')
        
        if len(bin_df) < 50:
            continue  # Not enough data for KDE

        x = bin_df['sun'].values
        y = bin_df['diff'].values
        xy = np.vstack([x, y])

        # Kernel Density Estimation
        kde = gaussian_kde(xy)

        # Create evaluation grid
        xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
        zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)

        # Centroid (unweighted average of points)
        centroid_x = np.average(x)
        centroid_y = np.average(y)

        # KDE peak value
        peak_density = zi.max()
        
        # Coordinates of peak density
        peak_index = np.unravel_index(np.argmax(zi), zi.shape)
        peak_x = xi[peak_index]
        peak_y = yi[peak_index]
        
        # Distance peak density to centroid
        peak_centroid_distance = np.sqrt((peak_x - centroid_x)**2 + (peak_y - centroid_y)**2)
        
        # Density area coverage: compute cumulative density function
        zi_flat = zi.ravel()
        sorted_zi = np.sort(zi_flat)[::-1]
        cumsum = np.cumsum(sorted_zi)
        cumsum /= cumsum[-1]  # Normalize to 1

        area_68 = (cumsum <= 0.68).sum() / len(cumsum)
        area_95 = (cumsum <= 0.95).sum() / len(cumsum)
        
        threshold_68 = sorted_zi[(cumsum <= 0.68).sum()]
        mask_68 = zi >= threshold_68
        x_vals_68 = xi[mask_68]
        y_vals_68 = yi[mask_68]
        width_68 = x_vals_68.max() - x_vals_68.min()
        height_68 = y_vals_68.max() - y_vals_68.min()
        aspect_ratio_68 = height_68 / width_68 if width_68 != 0 else np.nan
        area_ratio_68_95 = area_68 / area_95 if area_95 != 0 else np.nan

        results.append({
            'light_bin': l_bin,
            'centroid_sun': centroid_x,
            'centroid_diff': centroid_y,
            'peak_density': peak_density,
            'peak_x': peak_x,
            'peak_y': peak_y,
            'peak_centroid_distance': peak_centroid_distance,
            'area_68': area_68,
            'area_95': area_95,
            'width_68': width_68,
            'height_68': height_68,
            'aspect_ratio_68': aspect_ratio_68,
            'area_ratio_68_95': area_ratio_68_95
        })

    return pd.DataFrame(results)


if __name__ == '__main__':
    #### sun hours
    raster_files_Somme = {
        "sun": "Data/Somme_Sonnenstunden_Mean.tif",
        "diff": "Data/Somme_DiffH.tif"
    }
    
    gdf_Somme = load_point_data_and_add_light_values("Data/Somme_All_EPSG2154.gpkg")
    
    gdf_Somme = extract_raster_values_to_gdf(gdf_Somme, raster_files_Somme)
    light_properties_Somme = extract_species_light_niche(gdf_Somme)
    light_properties_Somme.to_csv("Data/light_properties_Somme.csv", index=False)
    
    ##
    raster_files_Lozere = {
        "sun": "Data/Lozere_Sonnenstunden_Mean.tif",
        "diff": "Data/Lozere_DiffH.tif"
    }
    
    gdf_Lozere = load_point_data_and_add_light_values("Data/Lozere_All_EPSG2154.gpkg")
    
    gdf_Lozere = extract_raster_values_to_gdf(gdf_Lozere, raster_files_Lozere)
    light_properties_Lozere = extract_species_light_niche(gdf_Lozere)
    light_properties_Lozere.to_csv("Data/light_properties_Lozere.csv", index=False)
    
    #### energy
    raster_files_Somme = {
        "sun": "Data/Somme_Energie_Mean.tif",
        "diff": "Data/Somme_DiffH.tif"
    }
    
    gdf_Somme = load_point_data_and_add_light_values("Data/Somme_All_EPSG2154.gpkg")
    
    gdf_Somme = extract_raster_values_to_gdf(gdf_Somme, raster_files_Somme)
    energy_properties_Somme = extract_species_light_niche(gdf_Somme)
    energy_properties_Somme.to_csv("Data/energy_properties_Somme.csv", index=False)
    
    ##
    raster_files_Lozere = {
        "sun": "Data/Lozere_Energie_Mean.tif",
        "diff": "Data/Lozere_DiffH.tif"
    }
    
    gdf_Lozere = load_point_data_and_add_light_values("Data/Lozere_All_EPSG2154.gpkg")
    
    gdf_Lozere = extract_raster_values_to_gdf(gdf_Lozere, raster_files_Lozere)
    energy_properties_Lozere = extract_species_light_niche(gdf_Lozere)
    energy_properties_Lozere.to_csv("Data/energy_properties_Lozere.csv", index=False)
    
    
    ###### per light bin
    raster_files_Somme = {
        "sun": "Data/Somme_Sonnenstunden_Mean.tif",
        "diff": "Data/Somme_DiffH.tif"
    }
    
    gdf_Somme = load_point_data_and_add_light_values("Data/Somme_All_EPSG2154.gpkg")
    
    gdf_Somme = extract_raster_values_to_gdf(gdf_Somme, raster_files_Somme)
    
    gdf_Somme = bin_light_values(gdf_Somme, column='L', interval_size=1.0)
    lightbin_results = extract_lightbin_niche(gdf_Somme)
    lightbin_results.to_csv("Data/lightbin_niche_light_Somme.csv", index=False)
    
    ##
    raster_files_Lozere = {
        "sun": "Data/Lozere_Sonnenstunden_Mean.tif",
        "diff": "Data/Lozere_DiffH.tif"
    }
    
    gdf_Lozere = load_point_data_and_add_light_values("Data/Lozere_All_EPSG2154.gpkg")
    
    gdf_Lozere = extract_raster_values_to_gdf(gdf_Lozere, raster_files_Lozere)
    
    gdf_Lozere = bin_light_values(gdf_Lozere, column='L', interval_size=1.0)
    lightbin_results = extract_lightbin_niche(gdf_Lozere)
    lightbin_results.to_csv("Data/lightbin_niche_light_Lozere.csv", index=False)
    
    ####
    raster_files_Somme = {
        "sun": "Data/Somme_Energie_Mean.tif",
        "diff": "Data/Somme_DiffH.tif"
    }
    
    gdf_Somme = load_point_data_and_add_light_values("Data/Somme_All_EPSG2154.gpkg")
    
    gdf_Somme = extract_raster_values_to_gdf(gdf_Somme, raster_files_Somme)
    
    gdf_Somme = bin_light_values(gdf_Somme, column='L', interval_size=1.0)
    lightbin_results = extract_lightbin_niche(gdf_Somme)
    lightbin_results.to_csv("Data/lightbin_niche_energy_Somme.csv", index=False)
    
    ##
    raster_files_Lozere = {
        "sun": "Data/Lozere_Energie_Mean.tif",
        "diff": "Data/Lozere_DiffH.tif"
    }
    
    gdf_Lozere = load_point_data_and_add_light_values("Data/Lozere_All_EPSG2154.gpkg")
    
    gdf_Lozere = extract_raster_values_to_gdf(gdf_Lozere, raster_files_Lozere)
    
    gdf_Lozere = bin_light_values(gdf_Lozere, column='L', interval_size=1.0)
    lightbin_results = extract_lightbin_niche(gdf_Lozere)
    lightbin_results.to_csv("Data/lightbin_niche_energy_Lozere.csv", index=False)
