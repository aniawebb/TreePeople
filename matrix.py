#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 08:32:45 2024

@author: aniawebb
"""

## IMPORTS ##
import geopandas as gpd
import pandas as pd
import jenkspy
import shapefile
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from shapely import wkt
import os

## FUNCTIONS ##
def read_shapefile(shp_path, encoding='utf-8'):
    """
    Read a shapefile into a GeoPandas dataframe with the specified encoding.
    """
    try:
        gdf = gpd.read_file(shp_path, encoding=encoding)
        # Convert to Pandas dataframe if needed
        df = pd.DataFrame(gdf)
        return df

    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError: {e}")
        # Handle or raise the error appropriately
        raise


def apply_jenks(column, num_classes, labels):
    # Calculate Jenks natural breaks
    breaks = jenkspy.jenks_breaks(column.dropna(), n_classes=num_classes)
    # Ensure unique breaks and reduce the number of classes if necessary
    unique_breaks = sorted(set(breaks))
    if len(unique_breaks) < num_classes + 1:  # fewer breaks than requested classes
        print(f"Warning: Only {len(unique_breaks) - 1} unique breaks found. Adjusting classes to match.")
        num_classes = len(unique_breaks) - 1  # adjust class count
        new_labels = labels[:num_classes]  # adjust labels to match available breaks
    else:
        new_labels = labels
    
    # Apply the cut function with the correct number of labels
    return pd.cut(column, bins=unique_breaks, labels=new_labels, include_lowest=True, duplicates='drop')

def apply_jenks_reversed(column, num_classes, labels):
    # Calculate Jenks natural breaks
    breaks = jenkspy.jenks_breaks(column.dropna(), n_classes=num_classes)
    # Ensure unique breaks and reduce the number of classes if necessary
    unique_breaks = sorted(set(breaks))
    if len(unique_breaks) < num_classes + 1:  # fewer breaks than requested classes
        print(f"Warning: Only {len(unique_breaks) - 1} unique breaks found. Adjusting classes to match.")
        num_classes = len(unique_breaks) - 1  # adjust class count
        new_labels = labels[:num_classes]  # adjust labels to match available breaks
    else:
        new_labels = labels
    
    # Reverse the labels
    new_labels = new_labels[::-1]
    
    # Apply the cut function with the correct number of labels
    return pd.cut(column, bins=unique_breaks, labels=new_labels, include_lowest=True, duplicates='drop')


def age_score(row):
    # Since Pop_10_64 has a score of 0, it does not contribute to the score.
    # Child_10 and Elderly65 are both given a score of 1, and since these are percentages, the score can be calculated by simply adding these percentages.
    return row['Child_10'] + row['Elderly65']

## IMPORT DATA ##
CalEnviroScreen40_CHAT_CensusData_LandUse_df = read_shapefile("/Users/aniawebb/Desktop/Desktop/work/treepeople/data/CalEnviroScreen40_CHAT_Census_LandUse_NLCD.shp")

## CREATE NEW COLUMNS ##
# Create YoungAndElderly column
CalEnviroScreen40_CHAT_CensusData_LandUse_df['YoungAndElderly'] = CalEnviroScreen40_CHAT_CensusData_LandUse_df.apply(age_score, axis=1)

# Create NonWhite column
CalEnviroScreen40_CHAT_CensusData_LandUse_df['NonWhite'] = CalEnviroScreen40_CHAT_CensusData_LandUse_df['Hispanic'] + CalEnviroScreen40_CHAT_CensusData_LandUse_df['AfricanAm'] + CalEnviroScreen40_CHAT_CensusData_LandUse_df['NativeAm'] + CalEnviroScreen40_CHAT_CensusData_LandUse_df['OtherMult']

# Create Open Space (Y/N) column
CalEnviroScreen40_CHAT_CensusData_LandUse_df['OpenSpace_Binary'] = np.where(CalEnviroScreen40_CHAT_CensusData_LandUse_df['ucd_descri'] == 'Open space and public lands', 0, 1)

# Create the 'Forest_Binary' column based on multiple terms
CalEnviroScreen40_CHAT_CensusData_LandUse_df['Forest_Binary'] = (~CalEnviroScreen40_CHAT_CensusData_LandUse_df['NLCD_Land'].str.contains('Forest|Wetlands|Shrub')).astype(int)

# Define the mapping of land use categories to their respective group numbers
land_use_mapping = {
    'Open space and public lands': 1,
    'Shrub/Scrub': 1,
    'Water': 1,
    'Open Water': 1,
    'Barren Land': 1,
    'Herbaceous': 1,
    'Agricultural': 2,
    'Planned development': 2,
    'Very low density residential': 2,
    'Low density commercial': 3,
    'Developed, Low Intensity': 3,
    'Low density residential': 3,
    'Mixed use of residential and commercial': 4,
    'Medium density residential': 4,
    'Developed, Medium Intensity': 4,
    'High density commercial': 5,
    'Industrial': 5,
    'High density residential': 5
}

# Create the new column Land_Use_Binary
CalEnviroScreen40_CHAT_CensusData_LandUse_df['Land_Use_Binary'] = CalEnviroScreen40_CHAT_CensusData_LandUse_df['ucd_descri'].map(land_use_mapping)

## CLEAN THE DATA ##
columns_to_drop = [
    "ApproxLoc", "TotPop19", "CIscore", "CIscoreP", "Pesticide", "PesticideP",
    "Tox_Rel", "Tox_Rel_P", "Lead", "Lead_P", "Cleanup", "CleanupP", "HazWaste",
    "HazWasteP", "SolWaste", "SolWasteP", "LowBirtWt", "LowBirWP", "Cardiovas",
    "CardiovasP", "Educatn", "EducatP", "HousBurd", "HousBurdP", "PopChar",
    "PopCharSc", "PopCharP", "Shape_Leng", "Shape_Area", "AAPI",'OzoneP',
    'PM2_5_P','DieselPM_P','TrafficP', 'DrinkWatP', 'GWThreatP', 'ImpWatBodP',
    'PolBurdP', 'AsthmaP', 'Ling_IsolP', 'PovertyP', 'UnemplP', "Hispanic", 
    "AfricanAm", "NativeAm", "OtherMult", 'Child_10', 'Elderly65'
]

CalEnviroScreen40_CHAT_CensusData_LandUse_df = CalEnviroScreen40_CHAT_CensusData_LandUse_df.drop(columns=columns_to_drop)

# Replace -999 values with 0 for consistency (no data)
CalEnviroScreen40_CHAT_CensusData_LandUse_df.replace(-999, 0, inplace=True)

# Replace 'Other - Not Determined' in 'ucd_descri' with the corresponding value in 'NLCD_Land'
CalEnviroScreen40_CHAT_CensusData_LandUse_df['ucd_descri'] = CalEnviroScreen40_CHAT_CensusData_LandUse_df.apply(lambda row: row['NLCD_Land'] if row['ucd_descri'] == 'Other - Not Determined' else row['ucd_descri'], axis=1)

## NORMALIZE THE DATA ##
# Applying Jenks Natural Breaks for each factor
quantile_labels = [1, 2, 3, 4, 5]

# Exposures
CalEnviroScreen40_CHAT_CensusData_LandUse_df['Ozone_Jenks'] = apply_jenks(CalEnviroScreen40_CHAT_CensusData_LandUse_df['Ozone'], 5, quantile_labels)
CalEnviroScreen40_CHAT_CensusData_LandUse_df['PM2_5_Jenks'] = apply_jenks(CalEnviroScreen40_CHAT_CensusData_LandUse_df['PM2_5'], 5, quantile_labels)
CalEnviroScreen40_CHAT_CensusData_LandUse_df['DieselPM_Jenks'] = apply_jenks(CalEnviroScreen40_CHAT_CensusData_LandUse_df['DieselPM'], 5, quantile_labels)
CalEnviroScreen40_CHAT_CensusData_LandUse_df['Traffic_Jenks'] = apply_jenks(CalEnviroScreen40_CHAT_CensusData_LandUse_df['Traffic'], 5, quantile_labels)
CalEnviroScreen40_CHAT_CensusData_LandUse_df['GWThreat_Jenks'] = apply_jenks(CalEnviroScreen40_CHAT_CensusData_LandUse_df['GWThreat'], 5, quantile_labels)
CalEnviroScreen40_CHAT_CensusData_LandUse_df['ImpWatBod_Jenks'] = apply_jenks(CalEnviroScreen40_CHAT_CensusData_LandUse_df['ImpWatBod'], 5, quantile_labels)
CalEnviroScreen40_CHAT_CensusData_LandUse_df['Asthma_Jenks'] = apply_jenks(CalEnviroScreen40_CHAT_CensusData_LandUse_df['Asthma'], 5, quantile_labels)
CalEnviroScreen40_CHAT_CensusData_LandUse_df['Poverty_Jenks'] = apply_jenks(CalEnviroScreen40_CHAT_CensusData_LandUse_df['Poverty'], 5, quantile_labels)
CalEnviroScreen40_CHAT_CensusData_LandUse_df['Extreme_Heat_Jenks'] = apply_jenks(CalEnviroScreen40_CHAT_CensusData_LandUse_df['tmax'], 5, quantile_labels)

# Vulnerability
CalEnviroScreen40_CHAT_CensusData_LandUse_df['Median_Income_Jenks'] = apply_jenks_reversed(CalEnviroScreen40_CHAT_CensusData_LandUse_df['mhi18'], 5, quantile_labels)
CalEnviroScreen40_CHAT_CensusData_LandUse_df['Ling_Isol_Jenks'] = apply_jenks_reversed(CalEnviroScreen40_CHAT_CensusData_LandUse_df['Ling_Isol'], 5, quantile_labels)
CalEnviroScreen40_CHAT_CensusData_LandUse_df['YoungAndElderly_Jenks'] = apply_jenks(CalEnviroScreen40_CHAT_CensusData_LandUse_df['YoungAndElderly'], 5, quantile_labels)
CalEnviroScreen40_CHAT_CensusData_LandUse_df['NonWhite_Jenks'] = apply_jenks(CalEnviroScreen40_CHAT_CensusData_LandUse_df['NonWhite'], 5, quantile_labels)

columns = ['Land_Use_Binary', 'Forest_Bin', 'OpenSpace_Binary', 'Ozone_Jenks', 'PM2_5_Jenks', 'Poverty_Jenks', 'DieselPM_Jenks', 'Asthma_Jenks', 'Traffic_Jenks', 'GWThreat_Jenks', 'ImpWatBod_Jenks','Median_Income_Jenks', 'Extreme_Heat_Jenks', 'Ling_Isol_Jenks', 'NonWhite_Jenks', 'YoungAndElderly_Jenks']

for column in columns: 
    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(CalEnviroScreen40_CHAT_CensusData_LandUse_df[column], bins=30, alpha=0.75, color='blue')
    plt.title('Histogram of ' + column)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
## CALCULATE WEIGHTED SCORES FOR EACH CATEGORY ##

# Exposures 40%
exposure_factors = [
    'Ozone_Jenks', 'PM2_5_Jenks', 'DieselPM_Jenks', 
    'Traffic_Jenks', 'GWThreat_Jenks', 'ImpWatBod_Jenks', 
    'Asthma_Jenks', 'Poverty_Jenks', 'Extreme_Heat_Jenks'
]

CalEnviroScreen40_CHAT_CensusData_LandUse_df['Exposure_Score'] = CalEnviroScreen40_CHAT_CensusData_LandUse_df[exposure_factors].astype(float).mean(axis=1) * 0.4

# Vulnerability 40%
demographic_factors = ['Median_Income_Jenks', 'Ling_Isol_Jenks',
                        'YoungAndElderly_Jenks', 'NonWhite_Jenks']

CalEnviroScreen40_CHAT_CensusData_LandUse_df['Demographics_Score'] = CalEnviroScreen40_CHAT_CensusData_LandUse_df[demographic_factors].astype(float).mean(axis=1)

CalEnviroScreen40_CHAT_CensusData_LandUse_df['Vulnerability_Score'] = (
    0.3 * CalEnviroScreen40_CHAT_CensusData_LandUse_df['Forest_Bin'] +
    0.5 * CalEnviroScreen40_CHAT_CensusData_LandUse_df['Demographics_Score'] +
    0.2 * CalEnviroScreen40_CHAT_CensusData_LandUse_df['Land_Use_Binary']
) * 0.4

# Opportunities 20%
CalEnviroScreen40_CHAT_CensusData_LandUse_df['Opportunities_Score'] = (
    CalEnviroScreen40_CHAT_CensusData_LandUse_df['OpenSpace_Binary'] 
) * 0.2

# Combine all scores into a final prioritization score
CalEnviroScreen40_CHAT_CensusData_LandUse_df['Prioritization_Score'] = (
    CalEnviroScreen40_CHAT_CensusData_LandUse_df['Exposure_Score'] +
    CalEnviroScreen40_CHAT_CensusData_LandUse_df['Vulnerability_Score'] +
    CalEnviroScreen40_CHAT_CensusData_LandUse_df['Opportunities_Score']
)

## VISUALIZATION ##

# Plot histogram of the final prioritization scores
plt.figure(figsize=(10, 6))
plt.hist(CalEnviroScreen40_CHAT_CensusData_LandUse_df['Prioritization_Score'], bins=30, alpha=0.75, color='green')
plt.title('Histogram of Prioritization Scores')
plt.xlabel('Prioritization Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Convert the DataFrame to a GeoDataFrame
gdf = gpd.GeoDataFrame(CalEnviroScreen40_CHAT_CensusData_LandUse_df, geometry='geometry')

# Get unique counties
counties = gdf['County'].unique()

# Create a plot for each county
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=False, sharey=False)

for ax, county in zip(axs, counties):
    county_gdf = gdf[gdf['County'] == county]
    county_gdf.plot(column='Prioritization_Score', cmap='viridis', linewidth=0.8, edgecolor='0.8', legend=True, ax=ax)
    
    # Set title and labels
    ax.set_title(f'Prioritization Score by Census Tract in {county}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Set the limits to zoom into the extent of the shape
    minx, miny, maxx, maxy = county_gdf.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

# Adjust layout
#plt.tight_layout()

# Show the plot
plt.show()

# Save the final dataframe as shapefile
# Define the columns you want to export in the desired order
columns_to_export = [
    'County', 'census_tra', 'Prioritization_Score', 'Exposure_Score', 'Demographics_Score', 
    'Vulnerability_Score', 'Opportunities_Score', 'OpenSpace_Binary', 'Forest_Bin', 'Land_Use_Binary', 
    'Ozone_Jenks', 'PM2_5_Jenks', 'DieselPM_Jenks', 'Traffic_Jenks', 'GWThreat_Jenks', 'ImpWatBod_Jenks', 
    'Asthma_Jenks', 'Poverty_Jenks', 'Extreme_Heat_Jenks', 'Median_Income_Jenks', 'Ling_Isol_Jenks', 
    'YoungAndElderly_Jenks', 'NonWhite_Jenks', 'geometry'
]

# Ensure your GeoDataFrame (gdf) contains these columns
gdf_selected = gdf[columns_to_export]

# Ensure the geometry column is preserved
gdf_selected = gpd.GeoDataFrame(gdf_selected, geometry='geometry')

# Convert categorical columns to integers
for col in gdf_selected.select_dtypes(include=['category']).columns:
    gdf_selected[col] = gdf_selected[col].astype(int)

# Define the output shapefile path
output_shapefile_path = '/Users/aniawebb/Desktop/Desktop/work/treepeople/data/final_prioritization_scores_ALLCOUNTIES.shp'

# Save the selected columns to a new shapefile
gdf_selected.to_file(output_shapefile_path, driver='ESRI Shapefile')

## SAVE DATA PER COUNTY ##
# Define the output directory for shapefiles
output_directory = '/Users/aniawebb/Desktop/Desktop/work/treepeople/data/'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Get the unique values in the 'County' column
unique_counties = gdf_selected['County'].unique()

# Loop through each unique county and create separate shapefiles
for county in unique_counties:
    # Create a GeoDataFrame for the current county
    gdf_county = gdf_selected[gdf_selected['County'] == county]
    
    # Define the output shapefile path for the current county
    output_shapefile_path = os.path.join(output_directory, f'final_prioritization_scores_{county}.shp')
    
    # Save the GeoDataFrame to a shapefile
    gdf_county.to_file(output_shapefile_path, driver='ESRI Shapefile')

    