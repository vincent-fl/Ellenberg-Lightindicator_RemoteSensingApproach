# -*- coding: utf-8 -*-
"""
Created on Thu May 22 16:01:00 2025

@author: vince
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from pysr import PySRRegressor
import matplotlib.pyplot as plt
import seaborn as sns

from Python_Code.DensityByLightValue import bin_light_values

def prepare_data_with_differences(species_df: pd.DataFrame, lightbin_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares data by computing differences between species niche metrics
    and the corresponding light bin niche metrics.
    """
    species_df = species_df.copy()
    species_df['light_bin'] = pd.to_numeric(species_df['light_class'], errors='coerce').dropna().astype(int)
    
    # Merge species with light bin data on bin
    merged_df = species_df.merge(lightbin_df, how='inner', left_on='light_bin', right_on='light_bin', suffixes=('', '_bin'))

    # Compute difference columns
    for col in ['centroid_sun', 'centroid_diff', 'peak_density', 'peak_x', 'peak_y', 
                'peak_centroid_distance', 'area_68', 'area_95', 'width_68', 'height_68', 
                'aspect_ratio_68', 'area_ratio_68_95']:
        merged_df[f'diff_{col}'] = merged_df[col] - merged_df[f'{col}_bin']

    # Keep only difference columns and target
    feature_cols = [f'diff_{col}' for col in [
        'centroid_sun', 'centroid_diff', 'peak_density', 'peak_x', 'peak_y', 
        'peak_centroid_distance', 'area_68', 'area_95', 'width_68', 'height_68', 
        'aspect_ratio_68', 'area_ratio_68_95']]
    
    merged_df = bin_light_values(merged_df, 'light_class', 1)
    merged_df = merged_df.dropna(subset=['light_bin'])
    
    return merged_df[feature_cols + ['light_bin']]

def plot_confusion_like_matrix(y_true, y_pred, save_name, title="True vs Predicted light Bins"):
    """
    Plots a color-only confusion-matrix-style heatmap for binned regression predictions.
    """
    y_true = np.array(y_true).astype(int)
    y_pred = np.round(y_pred).astype(int)

    all_bins = np.arange(min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()) + 1)
    y_true_cat = pd.Categorical(y_true, categories=all_bins, ordered=True)
    y_pred_cat = pd.Categorical(y_pred, categories=all_bins, ordered=True)

    matrix = pd.crosstab(y_true_cat, y_pred_cat, rownames=['True'], colnames=['Predicted'], dropna=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=False,
        cmap="Blues",
        cbar=True,
        cbar_kws={"label": "sample count"},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title(title, fontsize=14)
    plt.xlabel("predicted bin", fontsize=12)
    plt.ylabel("true bin", fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'Plots/{save_name}', dpi=600)
    plt.show()



def train_and_save_model(save_path: str, name: str, df: pd.DataFrame, save_name_plot: str):
    X = df[[f'diff_{col}' for col in [
        'centroid_sun', 'centroid_diff', 'peak_density', 'peak_x', 
        'peak_y', 'peak_centroid_distance', 'area_68', 'area_95', 
        'width_68', 'height_68', 'aspect_ratio_68', 'area_ratio_68_95']]].values
    y = df['light_bin'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = PySRRegressor(
        model_selection="best",
        niterations=50,
        unary_operators=["sin", "exp", "log"],
        binary_operators=["+", "-", "*", "/"],
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluation metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    best_equation = str(model.sympy())
    joblib.dump(model, f"{save_path}/light_model_{name}.pkl")

    # Confusion-matrix-like plot
    plot_confusion_like_matrix(y_test, y_pred, save_name_plot, title=f"True vs Predicted light bins: {name}")

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MedAE": medae,
        "R2": r2,
        "BestEquation": best_equation
    }, model


def load_species_and_bin_data(path, species_file, bin_file):
    species_df = pd.read_csv(f"{path}/{species_file}")
    bin_df = pd.read_csv(f"{path}/{bin_file}")
    return prepare_data_with_differences(species_df, bin_df)

# === Execution Block ===
path = 'Data'

#[
#    'centroid_sun',             # x0
#    'centroid_diff',            # x1
#    'peak_density',             # x2
#    'peak_x',                   # x3
#    'peak_y',                   # x4
#    'peak_centroid_distance',   # x5
#    'area_68',                  # x6
#    'area_95',                  # x7
#    'width_68',                 # x8
#    'height_68',                # x9
#    'aspect_ratio_68',          # x10
#    'area_ratio_68_95'          # x11
#]


# Load and train on energy data (Somme)
df1 = load_species_and_bin_data(path, 'energy_properties_Somme.csv', 'lightbin_niche_energy_Somme.csv')
res1, model1 = train_and_save_model(path, 'energy_Som_equation_diff', df1, 'model_matrix_energy_somme.png')
print(res1)
# 50 iterations {'RMSE': 1.158497807536293, 'MAE': 0.8448447766853822, 'MedAE': 0.5186903530547546, 'R2': 0.05625372907538406, 'BestEquation': 'exp(x11 - x6 + sin(-x6 + x7 + 5.0244803) + 4.1317534) - 1*18.526627'}

# Load and train on energy data (Lozere)
df2 = load_species_and_bin_data(path, 'energy_properties_Lozere.csv', 'lightbin_niche_energy_Lozere.csv')
res2, model2 = train_and_save_model(path, 'energy_Loz_equation_diff', df2, 'model_matrix_energy_lozere.png')
print(res2)
#{'RMSE': 1.0481100465185709, 'MAE': 0.7800059087287626, 'MedAE': 0.590552738158209, 'R2': 0.34327774554566937, 'BestEquation': '(-1*11.918328 + (x11 + x7)*5.9077005/x6)*(x11 + x7) - 1*(-4.8624325)'}

# Light properties (Somme)
df3 = load_species_and_bin_data(path, 'light_properties_Somme.csv', 'lightbin_niche_light_Somme.csv')
res3, model3 = train_and_save_model(path, 'light_Som_equation_diff', df3, 'model_matrix_light_somme.png')
print(res3)
#{'RMSE': 0.8568755521182806, 'MAE': 0.683289238346746, 'MedAE': 0.47598727149410447, 'R2': 0.48370214578683535, 'BestEquation': 'exp(-x6 + x7 + 1.7527944) + sin(x0 - x3 - x4)'}

# Light properties (Lozere)
df4 = load_species_and_bin_data(path, 'light_properties_Lozere.csv', 'lightbin_niche_light_Lozere.csv')
res4, model4 = train_and_save_model(path, 'light_Loz_equation_diff', df4, 'model_matrix_light_lozere.png')
print(res4)
#{'RMSE': 1.0741945588038486, 'MAE': 0.7985418674118653, 'MedAE': 0.6558480630129813, 'R2': 0.3101830490068842, 'BestEquation': '-sin(x5) + sin(exp(x2)*79.63829) + 6.12952'}

#### Energy species data with light bin data

# Light properties (Somme)
df5 = load_species_and_bin_data(path, 'energy_properties_Somme.csv', 'lightbin_niche_light_Somme.csv')
res5, model5 = train_and_save_model(path, 'lightEnergy_Som_equation_diff', df5, 'model_matrix_energySpecies_lightBin_somme.png')
print(res5)
#{'RMSE': 0.30506056477396637, 'MAE': 0.167188771181749, 'MedAE': 0.09651137928266884, 'R2': 0.9345609544986976, 'BestEquation': 'x10*0.71723354 + exp((x2/sin(-1.7326702/((-2.14864)*x2)) + sin(exp(2.520573 - sin(x10 + 7.670051))))*1.6382794) + 5.7234364'}

# Light properties (Lozere)
df6 = load_species_and_bin_data(path, 'energy_properties_Lozere.csv', 'lightbin_niche_light_Lozere.csv')
res6, model6 = train_and_save_model(path, 'lightEnergy_Loz_equation_diff', df6, 'model_matrix_energySpecies_lightBin_lozere.png')
print(res6)
#{'RMSE': 0.2769009601741706, 'MAE': 0.1948451563834724, 'MedAE': 0.11231423229373405, 'R2': 0.9541629257426105, 'BestEquation': '-10.815202/(x10 - 0.34553176)'}

#### Light species data with energy bin data

# Light properties (Somme)
df7 = load_species_and_bin_data(path, 'light_properties_Somme.csv', 'lightbin_niche_energy_Somme.csv')
res7, model7 = train_and_save_model(path, 'energyLight_Som_equation_diff', df7, 'model_matrix_lightSpecies_energyBin_somme.png')
print(res7)
#{'RMSE': 0.3597239650847926, 'MAE': 0.23878840835330337, 'MedAE': 0.12864625722139866, 'R2': 0.9090079269077179, 'BestEquation': 'x5*(-0.0174463) - (x0*0.022170808 + 79.91702) + sin(-0.008055054/x11)'}

# Light properties (Lozere)
df8 = load_species_and_bin_data(path, 'light_properties_Lozere.csv', 'lightbin_niche_energy_Lozere.csv')
res8, model8 = train_and_save_model(path, 'energyLight_Loz_equation_diff', df8, 'model_matrix_lightSpecies_energyBin_lozere.png')
print(res8)
#{'RMSE': 0.4261296637977164, 'MAE': 0.30516949931602055, 'MedAE': 0.2900736208610333, 'R2': 0.8914445828321931, 'BestEquation': '-0.006469508*(x0 + x4/(x6 + 0.041878436)) - 18.61019'}

'''
import os
import subprocess
import zipfile
import urllib.request

# Define version and URL
julia_version = "1.10.3"
julia_url = f"https://julialang-s3.julialang.org/bin/winnt/x64/{julia_version[:4]}/julia-{julia_version}-win64.zip"
install_path = os.path.expanduser("~/Julia")

# Create target folder
os.makedirs(install_path, exist_ok=True)
zip_path = os.path.join(install_path, "julia.zip")

# Download Julia
print("Downloading Julia...")
urllib.request.urlretrieve(julia_url, zip_path)

# Extract
print("Extracting Julia...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(install_path)

# Remove zip
os.remove(zip_path)

# Optional: Add to PATH temporarily in Python session
julia_bin = os.path.join(install_path, f"julia-{julia_version}", "bin")
os.environ["PATH"] = julia_bin + os.pathsep + os.environ["PATH"]

# Test installation
print("Julia installed to:", julia_bin)
subprocess.run(["julia", "--version"])

subprocess.run([
    "julia", "-e",
    'using Pkg; Pkg.rm("PythonCall"); Pkg.add("PythonCall"); Pkg.precompile()'
])


import shutil
import os

cache_path = os.path.expanduser("~/.julia/compiled/v1.10/PythonCall")
shutil.rmtree(cache_path, ignore_errors=True)
'''
