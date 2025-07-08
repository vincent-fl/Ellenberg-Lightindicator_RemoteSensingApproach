# -*- coding: utf-8 -*-
"""
Created on Tue May 20 11:49:07 2025

@author: vince
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, ks_2samp
import matplotlib.pyplot as plt

def load_aspect_data(filename, colname):
    df = pd.read_csv(filename)
    return np.deg2rad(df[colname].dropna().values)  # in Radiant

def estimate_density(data, angle_grid):
    kde = gaussian_kde(data, bw_method='scott')
    return kde(angle_grid)

def detrend_density(species_density, background_density):
    # Division mit kleinem Offset um Division durch 0 zu vermeiden
    return species_density / (background_density + 1e-9)

# --- Hauptfunktion ---
def main():
    # 1. Dateien einlesen
    south_all = load_aspect_data('Data/Lozere_FullArea.csv',
                                 colname='Loz_Aspect')
    south_species = load_aspect_data('Data/Lozere_Observations.csv',
                                     colname='LozS_Aspect')
    north_all = load_aspect_data('Data/Somme_FullArea.csv',
                                 colname='Som_Aspect')
    north_species = load_aspect_data('Data/Somme_Observations.csv',
                                     colname='SomS_Aspect')
    
    # 2. KDEs berechnen
    angle_grid = np.linspace(0, 2*np.pi, 180)

    south_all_density = estimate_density(south_all, angle_grid)
    south_species_density = estimate_density(south_species, angle_grid)
    north_all_density = estimate_density(north_all, angle_grid)
    north_species_density = estimate_density(north_species, angle_grid)

    # 3. Detrending
    detrended_south = detrend_density(south_species_density, south_all_density)
    detrended_north = detrend_density(north_species_density, north_all_density)

    # 4. Visualisierung
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angle_grid, detrended_south, label='Lozère', 
            color='grey', linestyle='--')
    ax.plot(angle_grid, detrended_north, label='Somme', 
            color='grey', linestyle='-')
    
    # Füge die Hintergrundfarben hinzu
    ax.fill_between(angle_grid, 1, 1.4, color='cyan', alpha=0.1)  # Oberhalb von 1
    ax.fill_between(angle_grid, 0.5, 1, color='yellow', alpha=0.1)    # Unterhalb von 1
    
    # Hauptänderungen hier:
    ax.set_theta_zero_location('E')  # 0° = Osten
    ax.set_theta_direction(1)        # Gegen den Uhrzeigersinn
    
    # Stelle die Winkelbeschriftungen ein
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(['E', 'N', 'W', 'S'])
    
    ax.set_title("Detrended aspect for all plant observations.")
    ax.legend(loc='lower right', bbox_to_anchor=(1.4, -0.1))
    plt.tight_layout()
    plt.show()
        
    # 5. Signifikanztest (KS-Test auf Detrended-Verteilungen)
    ks_stat, ks_p = ks_2samp(detrended_south, detrended_north)
    print(f"[KS-Test] statistics: {ks_stat:.3f}, p-value: {ks_p:.4f}")


if __name__ == '__main__':
    main() 