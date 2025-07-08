# -*- coding: utf-8 -*-
"""
Created on Tue May 27 08:31:36 2025

@author: vince
"""

from Python_Code.DensityByLightValue import run_density_analysis_by_light


# -------- RUN ANALYSIS --------
if __name__ == '__main__':
    
    ## energy
    run_density_analysis_by_light(
        tif_sun_path="Data/Somme_Energie_Mean.tif",
        tif_diff_path="Data/Somme_DiffH.tif",
        gpkg_path="Data/Somme_All_EPSG2154.gpkg",
        x_title="Energy [$W\ m^{-2}$]",
        area="Data/OtherSpecies/Somme_ErigeronCanadensis_EPSG2154.gpkg",
        interval_size=1.0,
        subsample=True,
        threshold=1300,  # subsamples light bin data when above the threshold. Down to the lowest one above the threshold
        save_name='TestSpecies/Test_ErigeronCanadensis_LightIndicator_Somme_EnergyMean.png',
        species_name='E. canadensis',
        legend_title='Somme'
    )
    
    run_density_analysis_by_light(
        tif_sun_path="Data/Somme_Energie_Mean.tif",
        tif_diff_path="Data/Somme_DiffH.tif",
        gpkg_path="Data/Somme_All_EPSG2154.gpkg",
        x_title="Energy [$W\ m^{-2}$]",
        area="Data/OtherSpecies/Somme_HelNumm_EPSG2154.gpkg",
        interval_size=1.0,
        subsample=True,
        threshold=1300,
        save_name='TestSpecies/Test_HelianthemumNummularium_LightIndicator_Somme_EnergyMean.png',
        species_name='M. nummularium',
        legend_title='Somme'
    )
    
    run_density_analysis_by_light(
        tif_sun_path="Data/Somme_Energie_Mean.tif",
        tif_diff_path="Data/Somme_DiffH.tif",
        gpkg_path="Data/Somme_All_EPSG2154.gpkg",
        x_title="Energy [$W\ m^{-2}$]",
        area="Data/OtherSpecies/Somme_PlantLanc_EPSG2154.gpkg",
        interval_size=1.0,
        subsample=True,
        threshold=1300,
        save_name='TestSpecies/Test_PlantagoLanceolata_LightIndicator_Somme_EnergyMean.png',
        species_name='P. lanceolata',
        legend_title='Somme'
    )
    
    
    
    run_density_analysis_by_light(
        tif_sun_path="Data/Lozere_Energie_Mean.tif",
        tif_diff_path="Data/Lozere_DiffH.tif",
        gpkg_path="Data/Lozere_All_EPSG2154.gpkg",
        x_title="Energy [$W\ m^{-2}$]",
        area="Data/OtherSpecies/Lozere_ErigeronCanadensis_EPSG2154.gpkg",
        interval_size=1.0,
        subsample=True,
        threshold=1300,
        save_name='TestSpecies/Test_ErigeronCanadensis_LightIndicator_Lozere_EnergyMean.png',
        species_name='E. canadensis',
        legend_title='Lozère'
    )
    
    run_density_analysis_by_light(
        tif_sun_path="Data/Lozere_Energie_Mean.tif",
        tif_diff_path="Data/Lozere_DiffH.tif",
        gpkg_path="Data/Lozere_All_EPSG2154.gpkg",
        x_title="Energy [$W\ m^{-2}$]",
        area="Data/OtherSpecies/Lozere_HelNumm_EPSG2154.gpkg",
        interval_size=1.0,
        subsample=True,
        threshold=1300,
        save_name='TestSpecies/Test_HelianthemumNummularium_LightIndicator_Lozere_EnergyMean.png',
        species_name='M. nummularium',
        legend_title='Lozère'
    )
    
    run_density_analysis_by_light(
        tif_sun_path="Data/Lozere_Energie_Mean.tif",
        tif_diff_path="Data/Lozere_DiffH.tif",
        gpkg_path="Data/Lozere_All_EPSG2154.gpkg",
        x_title="Energy [$W\ m^{-2}$]",
        area="Data/OtherSpecies/Lozere_PlantLanc_EPSG2154.gpkg",
        interval_size=1.0,
        subsample=True,
        threshold=1300,
        save_name='TestSpecies/Test_PlantagoLanceolata_LightIndicator_Lozere_EnergyMean.png',
        species_name='P. lanceolata',
        legend_title='Lozère'
    )
    
    
    ## sun
    run_density_analysis_by_light(
        tif_sun_path="Data/Somme_Sonnenstunden_Mean.tif",
        tif_diff_path="Data/Somme_DiffH.tif",
        gpkg_path="Data/Somme_All_EPSG2154.gpkg",
        x_title="Energy [$h\ d^{-1}$]",
        area="Data/OtherSpecies/Somme_ErigeronCanadensis_EPSG2154.gpkg",
        interval_size=1.0,
        subsample=True,
        threshold=1300,
        save_name='TestSpecies/Test_ErigeronCanadensis_LightIndicator_Somme_SunhoursMean.png',
        species_name='E. canadensis',
        legend_title='Somme'
    )
    
    run_density_analysis_by_light(
        tif_sun_path="Data/Somme_Sonnenstunden_Mean.tif",
        tif_diff_path="Data/Somme_DiffH.tif",
        gpkg_path="Data/Somme_All_EPSG2154.gpkg",
        x_title="Energy [$h\ d^{-1}$]",
        area="Data/OtherSpecies/Somme_HelNumm_EPSG2154.gpkg",
        interval_size=1.0,
        subsample=True,
        threshold=1300,
        save_name='TestSpecies/Test_HelianthemumNummularium_LightIndicator_Somme_SunhoursMean.png',
        species_name='M. nummularium',
        legend_title='Somme'
    )
    
    run_density_analysis_by_light(
        tif_sun_path="Data/Somme_Sonnenstunden_Mean.tif",
        tif_diff_path="Data/Somme_DiffH.tif",
        gpkg_path="Data/Somme_All_EPSG2154.gpkg",
        x_title="Energy [$h\ d^{-1}$]",
        area="Data/OtherSpecies/Somme_PlantLanc_EPSG2154.gpkg",
        interval_size=1.0,
        subsample=True,
        threshold=1300,
        save_name='TestSpecies/Test_PlantagoLanceolata_LightIndicator_Somme_SunhoursMean.png',
        species_name='P. lanceolata',
        legend_title='Somme'
    )
    
    
    
    run_density_analysis_by_light(
        tif_sun_path="Data/Lozere_Sonnenstunden_Mean.tif",
        tif_diff_path="Data/Lozere_DiffH.tif",
        gpkg_path="Data/Lozere_All_EPSG2154.gpkg",
        x_title="Energy [$h\ d^{-1}$]",
        area="Data/OtherSpecies/Lozere_ErigeronCanadensis_EPSG2154.gpkg",
        interval_size=1.0,
        subsample=True,
        threshold=1300,
        save_name='TestSpecies/Test_ErigeronCanadensis_LightIndicator_Lozere_SunhoursMean.png',
        species_name='E. canadensis',
        legend_title='Lozère'
    )
    
    run_density_analysis_by_light(
        tif_sun_path="Data/Lozere_Sonnenstunden_Mean.tif",
        tif_diff_path="Data/Lozere_DiffH.tif",
        gpkg_path="Data/Lozere_All_EPSG2154.gpkg",
        x_title="Energy [$h\ d^{-1}$]",
        area="Data/OtherSpecies/Lozere_HelNumm_EPSG2154.gpkg",
        interval_size=1.0,
        subsample=True,
        threshold=1300,
        save_name='TestSpecies/Test_HelianthemumNummularium_LightIndicator_Lozere_SunhoursMean.png',
        species_name='M. nummularium',
        legend_title='Lozère'
    )
    
    run_density_analysis_by_light(
        tif_sun_path="Data/Lozere_Sonnenstunden_Mean.tif",
        tif_diff_path="Data/Lozere_DiffH.tif",
        gpkg_path="Data/Lozere_All_EPSG2154.gpkg",
        x_title="Energy [$h\ d^{-1}$]",
        area="Data/OtherSpecies/Lozere_PlantLanc_EPSG2154.gpkg",
        interval_size=1.0,
        subsample=True,
        threshold=1300,
        save_name='TestSpecies/Test_PlantagoLanceolata_LightIndicator_Lozere_SunhoursMean.png',
        species_name='P. lanceolata',
        legend_title='Lozère'
    )
