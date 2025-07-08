# -*- coding: utf-8 -*-
"""
Created on Thu May 22 11:15:28 2025

@author: vince
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.sample import sample_gen
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from Python_Code.addLightVals import load_point_data_and_add_light_values

def load_bromus_data(area):
    bromus_pts = gpd.read_file(area)
    return bromus_pts


def extract_raster_values_at_points(raster_path, points):
    """
    Extracts raster values at given coordinates.
    """
    with rasterio.open(raster_path) as src:
        values = list(sample_gen(src, points))
        return [v[0] if v[0] != src.nodata else np.nan for v in values]


def bin_light_values(df, column='L', interval_size=1.0):
    """
    Converts string light values to float and assigns them to bins
    whose midpoints are always integers.
    """
    df = df.copy()
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(subset=[column])

    # Shift by 0.5*interval to center the bins on integers, then floor
    df['light_bin'] = df[column].apply(lambda x: int(np.floor((x + interval_size / 2) / interval_size)))

    return df

def plot_density_by_light(df_obs_binned, light_bins, x_title, bromus_pts, save_name, species_name, legend_title, plot_bromus=True):
    """
    Plots 2D KDE plots for each light bin, including:
    - KDE of general data (grey tones) and optional Bromus overlay (red)
    - Centroids and peak locations for both
    - Clarified legend entries
    """
    sns.set(style="white")
    fig, axes = plt.subplots(nrows=len(light_bins), figsize=(8, 5 * len(light_bins)))

    if len(light_bins) == 1:
        axes = [axes]

    for ax, f_bin in zip(axes, light_bins):
        bin_obs = df_obs_binned[df_obs_binned['light_bin'] == f_bin]
        x = bin_obs['sun']
        y = bin_obs['diff']

        if len(x) < 10:
            ax.set_title(f'light Bin {f_bin:.1f} — Not enough data')
            continue

        # General KDE
        values = np.vstack([x, y])
        kde = gaussian_kde(values)
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kde(positions).T, X.shape)

        def compute_level_thresholds(z_vals, levels=[0.68, 0.95]):
            sorted_vals = np.sort(z_vals.ravel())[::-1]
            cumsum = np.cumsum(sorted_vals)
            cumsum /= cumsum[-1]
            thresholds = [sorted_vals[np.searchsorted(cumsum, lv)] for lv in levels]
            return sorted(thresholds)

        levels = compute_level_thresholds(Z)

        # Plot general KDE
        sns.kdeplot(
            x=x, y=y, ax=ax, cmap="Greys", fill=True, alpha=0.4,
            levels=20, linewidths=0.5, label='Light Bin Data'
        )
        ax.contour(X, Y, Z, levels=levels, colors=['black', 'dimgray'], linestyles=['--', '-'], linewidths=1.0)

        # General centroid and peak
        x_mean = x.mean()
        y_mean = y.mean()
        ax.plot(x_mean, y_mean, 'o', color='black', markersize=8, label='Centroid (Light Bin)')
        peak_idx = np.unravel_index(np.argmax(Z), Z.shape)
        peak_x = X[peak_idx]
        peak_y = Y[peak_idx]
        ax.plot(peak_x, peak_y, marker='+', color='black', markersize=12, label='Peak Density (Light Bin)')

        # Bromus overlay (optional)
        if plot_bromus:
            x_b = bromus_pts['sun']
            y_b = bromus_pts['diff']
            if len(x_b) >= 10:
                b_values = np.vstack([x_b, y_b])
                b_kde = gaussian_kde(b_values)
                b_X, b_Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                b_positions = np.vstack([b_X.ravel(), b_Y.ravel()])
                b_Z = np.reshape(b_kde(b_positions).T, b_X.shape)
                b_levels = compute_level_thresholds(b_Z)

                ax.contour(b_X, b_Y, b_Z, levels=b_levels, colors=['red', 'darkred'],
                           linestyles=['--', '-'], linewidths=1.5, label='Bromus Contours')

                x_b_mean = x_b.mean()
                y_b_mean = y_b.mean()
                ax.plot(x_b_mean, y_b_mean, 'o', color='red', markersize=6, label='Centroid (Bromus)')

                b_peak_idx = np.unravel_index(np.argmax(b_Z), b_Z.shape)
                b_peak_x = b_X[b_peak_idx]
                b_peak_y = b_Y[b_peak_idx]
                ax.plot(b_peak_x, b_peak_y, marker='+', color='red', markersize=9, label='Peak Density (Bromus)')

        # Styling
        ax.set_title(f'Light indicator value {f_bin:.0f}')
        ax.set_xlabel(x_title)
        ax.set_ylabel('Δh (DSM - DTM) [m]')
        ax.set_ylim(-10, 30)

        # Unique legend entries
        manual_handles = [
            plt.Line2D([0], [0], color='grey', lw=6, label='light bin data'),
            plt.Line2D([0], [0], color='red', lw=6, label=f'${species_name}$ data'),
        ]

        ax.legend(handles=manual_handles, loc='upper left', title=f'{legend_title}')
    fig.tight_layout()
    plt.savefig(f'Plots/{save_name}', dpi=600)
    plt.show()


def run_density_analysis_by_light(
        tif_sun_path, tif_diff_path, gpkg_path, 
        x_title, area, save_name,
        species_name, legend_title, 
        light_volumn='L', interval_size=1.0,
        subsample=True, threshold=10):
    """
    Main function to analyze and compare 2D densities by light intervals.
    """
    
    # Get raster CRS for reprojection
    with rasterio.open(tif_sun_path) as src:
        raster_crs = src.crs
    
    # Load GPKG and reproject
    gdf = load_point_data_and_add_light_values(gpkg_path).to_crs(raster_crs)

    # Bin light values
    gdf = bin_light_values(gdf, column=light_volumn, interval_size=interval_size)

    # Extract raster values at observation points with all species / light bins
    coords = [(geom.x, geom.y) for geom in gdf.geometry]
    gdf['sun'] = extract_raster_values_at_points(tif_sun_path, coords)
    gdf['diff'] = extract_raster_values_at_points(tif_diff_path, coords)
    
    df_obs = gdf.dropna(subset=['sun', 'diff', 'light_bin'])
    unique_bins = sorted(df_obs['light_bin'].unique())
    
    # Load Bromus erectus point data
    bromus_pts = load_bromus_data(area=area)
    
    # Extract raster values at observation points for bromus erectus point data
    coords_br = [(geom.x, geom.y) for geom in bromus_pts.geometry]
    bromus_pts['sun'] = extract_raster_values_at_points(tif_sun_path, coords_br)
    bromus_pts['diff'] = extract_raster_values_at_points(tif_diff_path, coords_br)

    bromus_pts = bromus_pts.dropna(subset=['sun', 'diff'])

    df_obs = gdf.dropna(subset=['sun', 'diff', 'light_bin'])
    unique_bins = sorted(df_obs['light_bin'].unique())

    # Print bin stats
    print("\n--- Light Bin Summary ---")
    bin_counts = {}
    for b in unique_bins:
        bin_data = df_obs[df_obs['light_bin'] == b]
        num_species = bin_data['species'].nunique() if 'species' in bin_data.columns else 'N/A'
        print(f"Bin {b}: {len(bin_data)} points — {num_species} species")
        bin_counts[b] = len(bin_data)

    # Optional subsampling
    if subsample:
        print("\n--- Subsampling Data ---")
        valid_bins = {b: c for b, c in bin_counts.items() if c >= threshold}
        if valid_bins:
            min_count = min(valid_bins.values())
            print(f"Subsampling all bins ≥ {threshold} points to {min_count} points")
            
            subsampled = []
            for b in unique_bins:
                bin_data = df_obs[df_obs['light_bin'] == b]
                if len(bin_data) >= threshold:
                    bin_sampled = bin_data.sample(n=min_count, random_state=42)
                    subsampled.append(bin_sampled)
                else:
                    subsampled.append(bin_data)  # keep as is
                    
            df_obs = pd.concat(subsampled)
            
            # Recompute and print summary after subsampling
            print("\n--- Post-Subsampling Summary ---")
            for b in sorted(df_obs['light_bin'].unique()):
                bin_data = df_obs[df_obs['light_bin'] == b]
                num_species = bin_data['species'].nunique() if 'species' in bin_data.columns else 'N/A'
                print(f"Bin {b}: {len(bin_data)} points — {num_species} species")
            
            else:
                print("No bins met the threshold for subsampling.")
    
    coords_br = [(geom.x, geom.y) for geom in bromus_pts.geometry]
    bromus_pts['sun'] = extract_raster_values_at_points(tif_sun_path, coords_br)
    bromus_pts['diff'] = extract_raster_values_at_points(tif_diff_path, coords_br)
    bromus_pts = bromus_pts.dropna(subset=['sun', 'diff'])

    # Plot
    plot_density_by_light(df_obs, sorted(df_obs['light_bin'].unique()), x_title, 
                          bromus_pts, save_name, species_name, legend_title, 
                          plot_bromus=True)


# -------- RUN ANALYSIS --------
if __name__ == '__main__':
    
    run_density_analysis_by_light(
        tif_sun_path="Data/Somme_Energie_Mean.tif",
        tif_diff_path="Data/Somme_DiffH.tif",
        gpkg_path="Data/Somme_All_EPSG2154.gpkg",
        x_title="Energy [$W\ m^{-2}$]",
        area="Data/Somme_BromusErectus_EPSG2154.gpkg",
        interval_size=1.0,
        subsample=True,
        threshold=1300,
        save_name='LightIndicator_Somme_EnergyMean.png',
        species_name='B. erectus',
        legend_title='Somme'
    )
    
    run_density_analysis_by_light(
        tif_sun_path="Data/Lozere_Energie_Mean.tif",
        tif_diff_path="Data/Lozere_DiffH.tif",
        gpkg_path="Data/Lozere_All_EPSG2154.gpkg",
        x_title="Energy [$W\ m^{-2}$]",
        area="Data/Lozere_BromusErectus_EPSG2154.gpkg",
        interval_size=1.0,
        subsample=True,
        threshold=1300,
        save_name='LightIndicator_Lozere_EnergyMean.png',
        species_name='B. erectus',
        legend_title='Lozère'
    )
    
    run_density_analysis_by_light(
        tif_sun_path="Data/Somme_Sonnenstunden_Mean.tif",
        tif_diff_path="Data/Somme_DiffH.tif",
        gpkg_path="Data/Somme_All_EPSG2154.gpkg",
        x_title="potential light [$h\ d^{-1}$]",
        area="Data/Somme_BromusErectus_EPSG2154.gpkg",
        interval_size=1.0,
        subsample=True,
        threshold=1300,
        save_name='LightIndicator_Somme_SunhoursMean.png',
        species_name='B. erectus',
        legend_title='Somme'
    )
    
    run_density_analysis_by_light(
        tif_sun_path="Data/Lozere_Sonnenstunden_Mean.tif",
        tif_diff_path="Data/Lozere_DiffH.tif",
        gpkg_path="Data/Lozere_All_EPSG2154.gpkg",
        x_title="potential light [$h\ d^{-1}$]",
        area="Data/Lozere_BromusErectus_EPSG2154.gpkg",
        interval_size=1.0,
        subsample=True,
        threshold=1300,
        save_name='LightIndicator_Lozere_SunhoursMean.png',
        species_name='B. erectus',
        legend_title='Lozère'
    )
