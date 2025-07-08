# -*- coding: utf-8 -*-
"""
Created on Tue May 20 13:22:05 2025

@author: vince
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.sample import sample_gen
import seaborn as sns
import matplotlib.pyplot as plt

from addMoistureVals import load_point_data_and_add_moisture_values

def sample_raster_values(raster_path, sample_size):
    """
    Randomly samples raster values and their coordinates from a given raster.
    """
    with rasterio.open(raster_path) as src:
        band = src.read(1)
        transform = src.transform
        height, width = band.shape

        total_pixels = height * width
        flat_indices = np.random.choice(total_pixels, size=sample_size, replace=False)
        rows, cols = np.unravel_index(flat_indices, band.shape)
        coords = [rasterio.transform.xy(transform, row, col) for row, col in zip(rows, cols)]
        values = band[rows, cols]

        return pd.DataFrame({'x': [c[0] for c in coords],
                             'y': [c[1] for c in coords],
                             'value': values})


def extract_raster_values_at_points(raster_path, points, nodata=None):
    """
    Extracts raster values at point coordinates (e.g., observation points).
    """
    with rasterio.open(raster_path) as src:
        values = list(sample_gen(src, points))
        result = [v[0] if v[0] != src.nodata else np.nan for v in values]
        return result


def filter_top_species(gdf, min_count, max_species):
    """
    Filters the GeoDataFrame to only include the top most frequent species
    that exceed a minimum observation count.
    """
    species_counts = gdf['species'].value_counts()
    selected_species = species_counts[species_counts >= min_count].head(max_species).index
    return gdf[gdf['species'].isin(selected_species)].copy(), selected_species


def prepare_observation_data(gpkg_path, raster_crs, sun_path, diff_path, min_count, max_species):
    """
    Loads and filters the GPKG observation data, and extracts raster values.
    """
    # Load and reproject observation data
    gdf = gpd.read_file(gpkg_path).to_crs(raster_crs)
    gdf, selected_species = filter_top_species(gdf, min_count, max_species)

    # Extract raster values at observation locations
    coords = [(geom.x, geom.y) for geom in gdf.geometry]
    gdf['sun'] = extract_raster_values_at_points(sun_path, coords)
    gdf['diff'] = extract_raster_values_at_points(diff_path, coords)

    return gdf.dropna(subset=['sun', 'diff']), selected_species


def plot_density_comparison(df_all, df_obs, selected_species):
    """
    Creates 2D kernel density plots for each selected species.
    """
    sns.set(style="white")
    fig, axes = plt.subplots(nrows=len(selected_species), figsize=(8, 5 * len(selected_species)))

    if len(selected_species) == 1:
        axes = [axes]

    for ax, species in zip(axes, selected_species):
        # Plot full raster sample
        sns.kdeplot(
            data=df_all, x='sun', y='diff',
            cmap="Blues", fill=True, alpha=0.3, levels=20, ax=ax, label="All pixels"
        )

        # Plot species-specific observation density
        sns.kdeplot(
            data=df_obs[df_obs['species'] == species],
            x='sun', y='diff', cmap="Reds", fill=True, alpha=0.5, levels=20, ax=ax, label=species
        )

        ax.set_title(f'2D Density Plot: {species}')
        ax.set_ylim(-1, 30)
        ax.legend()

    plt.tight_layout()
    plt.show()


def run_environmental_density_analysis(
        tif_sun_path, tif_diff_path, gpkg_path,
        min_count=50, max_species=5, sample_size=500_000):
    """
    Main function to compare environmental densities between all raster pixels
    and observation locations for selected species.
    """
    # Sample random values from the sun and height-difference rasters
    df_sun = sample_raster_values(tif_sun_path, sample_size)
    df_diff = sample_raster_values(tif_diff_path, sample_size)

    # Merge the sampled values
    df_all = pd.DataFrame({
        'sun': df_sun['value'],
        'diff': df_diff['value']
    }).dropna()

    # Read CRS for projection match
    with rasterio.open(tif_sun_path) as src:
        raster_crs = src.crs

    # Extract raster values at observation points and filter by species
    df_obs, selected_species = prepare_observation_data(
        gpkg_path, raster_crs, tif_sun_path, tif_diff_path, min_count, max_species
    )

    # Plot comparisons
    plot_density_comparison(df_all, df_obs, selected_species)

run_environmental_density_analysis(
    tif_sun_path="C:/Users/vince/FernerkundungsProjekt_SoSe25/Frage1_allgemeineEffekte/Daten_Frage1/Lozere_BrVeg_Energie_Mean.tif",
    tif_diff_path="C:/Users/vince/FernerkundungsProjekt_SoSe25/Frage1_allgemeineEffekte/Daten_Frage1/Lozere_DiffH.tif",
    gpkg_path=load_point_data_and_add_moisture_values('Lozere_All_EPSG2154.gpkg'), 
    min_count=30,
    max_species=7,
    sample_size=500_000
)


# ---------- all observations in one plot ------------

def plot_density_all_observations(df_all, df_obs):
    """
    Plots a 2D kernel density plot comparing all raster pixels vs. all observations.
    """
    sns.set(style="white")
    plt.figure(figsize=(8, 6))

    sns.kdeplot(
        data=df_all, x='sun', y='diff',
        cmap="Blues", fill=True, alpha=0.3, levels=20, label="All pixels"
    )
    sns.kdeplot(
        data=df_obs, x='sun', y='diff',
        cmap="Reds", fill=True, alpha=0.5, levels=20, label="Observations"
    )

    plt.title("2D Density Plot: All Observations vs. All Pixels")
    plt.xlabel("Potential Sunlight Hours")
    plt.ylabel("Height Difference (DSM - DTM)")
    plt.ylim(-1, 30)
    plt.legend()
    plt.tight_layout()
    plt.show()

def sample_raster_values2(raster_path, sample_size):
    with rasterio.open(raster_path) as src:
        band = src.read(1)
        transform = src.transform
        height, width = band.shape

        total_pixels = height * width
        flat_indices = np.random.choice(total_pixels, size=sample_size, replace=False)
        rows, cols = np.unravel_index(flat_indices, band.shape)
        coords = [rasterio.transform.xy(transform, row, col) for row, col in zip(rows, cols)]
        values = band[rows, cols]

        return pd.DataFrame({'x': [c[0] for c in coords],
                             'y': [c[1] for c in coords],
                             'value': values})


def extract_raster_values_at_points2(raster_path, points):
    with rasterio.open(raster_path) as src:
        values = list(sample_gen(src, points))
        result = [v[0] if v[0] != src.nodata else np.nan for v in values]
        return result


def run_all_observations_density_analysis(
    tif_sun_path, tif_diff_path, gpkg_path, sample_size=500_000
):
    """
    Loads raster and observation data, extracts values, and plots a 2D KDE
    comparing all raster pixels and all observation points.
    """

    # Sample from rasters
    df_sun = sample_raster_values2(tif_sun_path, sample_size)
    df_diff = sample_raster_values2(tif_diff_path, sample_size)

    df_all = pd.DataFrame({
        'sun': df_sun['value'],
        'diff': df_diff['value']
    }).dropna()

    # Load and reproject GPKG data
    with rasterio.open(tif_sun_path) as src:
        raster_crs = src.crs

    gdf = gpd.read_file(gpkg_path).to_crs(raster_crs)

    # Extract raster values at observation points
    coords = [(geom.x, geom.y) for geom in gdf.geometry]
    gdf['sun'] = extract_raster_values_at_points2(tif_sun_path, coords)
    gdf['diff'] = extract_raster_values_at_points2(tif_diff_path, coords)

    df_obs = gdf.dropna(subset=['sun', 'diff'])

    # Plot
    plot_density_all_observations(df_all, df_obs)

run_all_observations_density_analysis(
    tif_sun_path="C:/Users/vince/FernerkundungsProjekt_SoSe25/Frage1_allgemeineEffekte/Daten_Frage1/Lozere_BrVeg_Energie_Mean.tif",
    tif_diff_path="C:/Users/vince/FernerkundungsProjekt_SoSe25/Frage1_allgemeineEffekte/Daten_Frage1/Lozere_DiffH.tif",
    gpkg_path="C:/Users/vince/FernerkundungsProjekt_SoSe25/Frage1_allgemeineEffekte/AlleArten/Lozere_All_EPSG2154.gpkg",
    sample_size=500_000
)

