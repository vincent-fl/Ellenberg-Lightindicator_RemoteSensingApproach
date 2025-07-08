# -*- coding: utf-8 -*-
"""
Created on Tue May 20 08:53:24 2025

@author: vince
"""

import geopandas as gpd
import numpy as np
from scipy.stats import poisson, chi2, norm, gaussian_kde
from scipy.spatial import distance
import shapely.geometry
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from shapely.geometry import Point

# =============================================================================
# ----- 1.1 load department polygons (e.g. via ADMIN borders of France) -----

# Load GeoJSON files
departements = gpd.read_file("C:/Users/vince/FernerkundungsProjekt_SoSe25/Frage1_allgemeineEffekte/AlleArten/departements.geojson")

# Project onto EPSG 2154
departements = departements.to_crs("EPSG:2154")

# ----- 1.2 Load point data -----

point_path = "C:/Users/vince/FernerkundungsProjekt_SoSe25/Frage1_allgemeineEffekte/AlleArten/"

somme_points = gpd.read_file(f"{point_path}Somme_All_EPSG2154.gpkg")
somme_points = somme_points.to_crs("EPSG:2154")

lozere_points = gpd.read_file(f"{point_path}Lozere_All_EPSG2154.gpkg")
lozere_points = lozere_points.to_crs("EPSG:2154")



# =============================================================================
# ----- 2. define functions for test types ----- 

def test_uniform_distribution(points, polygon, grid_size=5, alpha=0.05):
    """
    Uniformity test: Chi² test for uniform distribution over grid cells,
    manual calculation of test statistic, p-value, and critical value.

    Parameters:
    - points: GeoDataFrame or similar with point geometries
    - polygon: shapely Polygon defining the study area
    - grid_size: number of cells per axis (grid_size x grid_size)
    - alpha: significance level for critical value (default 0.05)

    Returns:
    - dict with chi2 statistic, p-value, critical value, counts per cell,
      and 'reject_null' boolean indicating test decision
    """
    bounds = polygon.bounds
    minx, miny, maxx, maxy = bounds
    dx = (maxx - minx) / grid_size
    dy = (maxy - miny) / grid_size

    counts = []

    for i in range(grid_size):
        for j in range(grid_size):
            cell = shapely.geometry.box(
                minx + i * dx, miny + j * dy,
                minx + (i + 1) * dx, miny + (j + 1) * dy
            )
            cell = cell.intersection(polygon)
            if not cell.is_empty:
                count = points[points.within(cell)].shape[0]
                counts.append(count)

    observed = np.array(counts)
    expected = np.full_like(observed, observed.mean())

    # Calculate Chi² statistic manually
    chi2_stat = np.sum((observed - expected)**2 / expected)

    # Degrees of freedom = number of cells - 1
    df = len(observed) - 1

    # Calculate p-value from Chi² survival function (1-CDF)
    p_value = chi2.sf(chi2_stat, df)

    # Calculate critical value for given alpha
    critical_value = chi2.ppf(1 - alpha, df)
    
    # Decide if null hypothesis should be rejected
    reject_null = chi2_stat > critical_value

    return {
        "Chi2 statistic": chi2_stat,
        "p-value": p_value,
        "critical value": critical_value,
        "Counts": counts,
        "reject_null": reject_null
    }

def test_random_distribution(points, polygon, grid_size=5, alpha=0.05):
    """
    Random distribution test using Poisson model for cell counts.

    Parameters:
    - points: GeoDataFrame or similar with point geometries
    - polygon: shapely Polygon defining the study area
    - grid_size: number of cells per axis (grid_size x grid_size)
    - alpha: significance level for the test (default 0.05)

    Returns:
    - dict with lambda (mean count), observed frequency counts of cell counts,
      expected frequencies under Poisson, chi2 statistic, p-value, critical value,
      and test decision (reject_null)
    """
    bounds = polygon.bounds
    minx, miny, maxx, maxy = bounds
    dx = (maxx - minx) / grid_size
    dy = (maxy - miny) / grid_size

    counts = []

    # Count points in each grid cell intersected with polygon
    for i in range(grid_size):
        for j in range(grid_size):
            cell = shapely.geometry.box(
                minx + i * dx, miny + j * dy,
                minx + (i + 1) * dx, miny + (j + 1) * dy
            )
            cell = cell.intersection(polygon)
            if not cell.is_empty:
                count = points[points.within(cell)].shape[0]
                counts.append(count)

    counts = np.array(counts)
    lambda_ = counts.mean()  # mean count per cell

    # Count frequency of each count value (how many cells have 0 points, 1 point, 2 points, ...)
    max_count = counts.max()
    observed_freq = np.array([np.sum(counts == k) for k in range(max_count + 1)])

    # Calculate expected frequencies under Poisson distribution
    expected_freq = poisson.pmf(np.arange(max_count + 1), mu=lambda_) * len(counts)

    # Combine tail probabilities for small expected values to meet Chi² assumptions:
    # Merge last bins with expected freq < 5 into one bin
    while expected_freq.size > 1 and expected_freq[-1] < 5:
        expected_freq[-2] += expected_freq[-1]
        observed_freq[-2] += observed_freq[-1]
        expected_freq = expected_freq[:-1]
        observed_freq = observed_freq[:-1]

    # Calculate Chi² statistic
    chi2_stat = np.sum((observed_freq - expected_freq) ** 2 / expected_freq)

    # Degrees of freedom = (number of bins) - 1 - (parameters estimated, here 1 for lambda)
    df = len(observed_freq) - 1 - 1

    # Calculate p-value and critical value
    p_value = chi2.sf(chi2_stat, df)
    critical_value = chi2.ppf(1 - alpha, df)

    reject_null = chi2_stat > critical_value

    return {
        "Lambda": lambda_,
        "Observed frequencies": observed_freq.tolist(),
        "Expected frequencies": expected_freq.tolist(),
        "Chi2 statistic": chi2_stat,
        "p-value": p_value,
        "critical value": critical_value,
        "reject_null": reject_null
    }


def test_clustering(points, alpha=0.05):
    """
    Nearest neighbor test for spatial randomness.

    Parameters:
    - points: numpy array of shape (n, 2) with x,y coordinates
    - alpha: significance level for the test

    Returns:
    - dict with observed mean nearest neighbor distance,
      expected distance under CSR, Z-score, critical Z value,
      and test decision (reject_null)
    """
    points = np.array([[pt.x, pt.y] for pt in points.geometry])
    
    # Compute pairwise distances
    distances = distance.pdist(points)
    distarray = distance.squareform(distances)

    # Compute nearest neighbor distances for first 100 points (or less if fewer points)
    n = min(100, len(points))
    nearest = np.zeros(n)
    for i in range(n):
        distarray[i, i] = np.Inf  # ignore self-distance
        nearest[i] = np.min(distarray[i, :])

    observednearest = np.mean(nearest)

    # Calculate study area bounding rectangle
    maparea = (np.max(points[:, 0]) - np.min(points[:, 0])) * (np.max(points[:, 1]) - np.min(points[:, 1]))

    # Expected mean nearest neighbor distance under complete spatial randomness (CSR)
    expectednearest = 0.5 * np.sqrt(maparea / len(points))

    # Standard error of the mean nearest neighbor distance
    se = 0.26136 / np.sqrt(len(points) ** 2 / maparea)

    # Z-score for the observed vs expected nearest neighbor distance
    Z = (observednearest - expectednearest) / se

    # Critical Z value for two-tailed test
    critical_z = norm.ppf(1 - alpha / 2)

    reject_null = abs(Z) > critical_z

    return {
        "Observed nearest neighbor distance": observednearest,
        "Expected nearest neighbor distance": expectednearest,
        "Z-score": Z,
        "Critical Z value": critical_z,
        "Reject null hypothesis": reject_null
    }

# =============================================================================
# ----- 3. define function to test ----- 

def analyze_department(department_gdf, department_identifier, points_gdf, test="uniform", grid_size=5):
    """
    dept_identifier: Name or code of the department
    test: "uniform", "random", "cluster"
    """
    # Filter Department (nach Code oder Name)
    if "code" in department_gdf.columns and isinstance(department_identifier, str) and department_identifier.isdigit():
        dept = department_gdf[department_gdf["code"] == department_identifier]
    else:
        dept = department_gdf[department_gdf["nom"] == department_identifier]

    if dept.empty:
        raise ValueError(f"Departement '{department_identifier}' not found.")

    poly = dept.geometry.values[0]
    points_in = points_gdf[points_gdf.within(poly)]

    print(f"--- {department_identifier} ---")
    print(f"Points in polygon: {len(points_in)}")

    if test == "uniform":
        result = test_uniform_distribution(points_in, poly, grid_size)
    elif test == "random":
        result = test_random_distribution(points_in, poly, grid_size)
    elif test == "cluster":
        result = test_clustering(points_in)
    else:
        raise ValueError(f"Unknown test type: '{test}'")

    return result

# =============================================================================
# ----- 4. test -----
# Department identifier: Somme (80) and Lozère (48)

# ----- 4.1 Somme -----
somme_uniform = analyze_department(departements, "80", somme_points, "uniform", grid_size=50)
somme_random = analyze_department(departements, "80", somme_points, "random", grid_size=50)
somme_cluster = analyze_department(departements, "80", somme_points, "cluster", grid_size=50)

print("Uniform test Somme")
for key, value in somme_uniform.items():
    if key != "Counts":
        print(f"{key}: {value}")
print('======================================================================')
print("Random test Somme")
for key, value in somme_random.items():
    if (key != "Observed frequencies") and (key != "Expected frequencies"):
        print(f"{key}: {value}")
print('======================================================================')
print("Cluster test Somme")
print(somme_cluster)
print('======================================================================')

# ----- 4.1 Lozère -----
lozere_uniform = analyze_department(departements, "48", lozere_points, "uniform", grid_size=50)
lozere_random = analyze_department(departements, "48", lozere_points, "random", grid_size=50)
lozere_cluster = analyze_department(departements, "48", lozere_points, "cluster", grid_size=50)

print("Uniform test Lozère")
for key, value in lozere_uniform.items():
    if key != "Counts":
        print(f"{key}: {value}")
print('======================================================================')
print("Random test Lozère")
for key, value in lozere_random.items():
    if (key != "Observed frequencies") and (key != "Expected frequencies"):
        print(f"{key}: {value}")
print('======================================================================')
print("Cluster test Lozère")
print(lozere_cluster)
print('======================================================================')

# =============================================================================
# ----- 5. Plot -----
# Department identifier: Somme (80) and Lozère (48)

def plot_3d_density(department_gdf, department_identifier, points, department_name, bandwidth=1000, grid_size=100):

    # Filter Department (by code or name)
    if "code" in department_gdf.columns and isinstance(department_identifier, str) and department_identifier.isdigit():
        dept = department_gdf[department_gdf["code"] == department_identifier]
    else:
        dept = department_gdf[department_gdf["nom"] == department_identifier]

    if dept.empty:
        raise ValueError(f"Department '{department_identifier}' not found.")

    # Clip points inside department polygon
    points_in_poly = points[points.within(dept.unary_union)]

    if points_in_poly.empty:
        raise ValueError("No points inside the department polygon.")

    coords = np.vstack([points_in_poly.geometry.x, points_in_poly.geometry.y])

    if coords.shape[1] < 2:
        raise ValueError("Need at least two points to compute KDE.")

    # Bounding box
    minx, miny, maxx, maxy = dept.total_bounds

    x_grid = np.linspace(minx, maxx, grid_size)
    y_grid = np.linspace(miny, maxy, grid_size)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    grid_coords = np.vstack([x_mesh.ravel(), y_mesh.ravel()])

    # KDE with bandwidth scaling
    kde = gaussian_kde(coords, bw_method=bandwidth / np.std(coords, axis=1).mean())

    density = kde(grid_coords).reshape(grid_size, grid_size)

    # Mask points outside polygon
    mask = np.array([dept.unary_union.contains(Point(x, y)) for x, y in zip(grid_coords[0], grid_coords[1])])
    density_flat = density.ravel()
    density_flat[~mask] = np.nan
    density = density_flat.reshape(grid_size, grid_size)

    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x_mesh, y_mesh, density, cmap='viridis', edgecolor='none', alpha=0.8)

    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Density')
    ax.set_title(f'3D Density Plot of Points in {department_name}')

    plt.show()

plot_3d_density(departements, '80', somme_points, "Somme")
plot_3d_density(departements, '48', lozere_points, "Lozère")
