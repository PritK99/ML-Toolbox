"""
Script Description:
This script processes a data/mumbai_house_prices.csv and performs preprocessing on the data. The new dataset is saved to data/modified_mumbai_house_prices.csv

Preprocessing steps:
1. Uses "price" and "price_unit" to convert the price from different units to Crores only.
2. Retrieves latitude and longitude for each unique combination of "locality" and "region" using the Bing Maps API.
3. Converts nominal and ordinal data (type, age, status) to numerical values.
4. Drops unnecessary columns like price_unit, locality, and region.
"""

import pandas as pd
import random
import geocoder

API_KEY = "<API KEY>"

# Read the input CSV file
data = "../data/mumbai_house_prices.csv"
house_price = pd.read_csv(data)

# Step 1: Convert price_unit to price in Crores
for i in range(len(house_price)):
    price_unit = house_price.loc[i, "price_unit"]
    if price_unit == "L":
        house_price.at[i, "price"] = house_price.at[i, "price"] / 100

# Step 2: Obtain unique locations
unique_locations = set()
for i in range(len(house_price)):
    location = house_price.loc[i, "locality"] + ", " + house_price.loc[i, "region"] + ", Mumbai"
    unique_locations.add(location)
print("Total number of unique locations: ", len(unique_locations))

# Step 3: Retrieve latitude and longitude for each unique location
lat_long_dict = {}
unknown_locations = []

count = 0  # To keep track of processed locations
for location in unique_locations:
    g = geocoder.bing(location, key=API_KEY, timeout=20)
    results = g.json

    if results is None:
        unknown_locations.append(location)
    else:
        latitude = results['lat']
        longitude = results['lng']
        lat_long_dict[location] = [latitude, longitude]

    count += 1
    if count % 100 == 0:
        print(f"{count} locations processed")

print(f'Total {len(unknown_locations)} locations are unknown: {unknown_locations}')

# Step 4: Assign latitude and longitude to the dataset
del_idx = []
for i in range(len(house_price)):
    location = house_price.loc[i, "locality"] + ", " + house_price.loc[i, "region"] + ", Mumbai"

    if location in unknown_locations:
        del_idx.append(i)
    else:
        lat_long = lat_long_dict[location]
        house_price.at[i, "latitude"] = lat_long[0]
        house_price.at[i, "longitude"] = lat_long[1]

# Remove rows with unknown locations
house_price.drop(del_idx, inplace=True)
house_price.reset_index(drop=True, inplace=True)

# Step 5: Drop unnecessary columns
house_price.drop(['price_unit', 'locality', 'region'], axis=1, inplace=True)

# Step 6: Convert nominal and ordinal data to numerical values
type_mapping = {"Studio Apartment": 0, "Apartment": 0.2, "Independent House": 0.4, "Villa": 0.6, "Penthouse": 1}
age_mapping = {"Resale": 0, "Unknown": 0.5, "New": 1}
status_mapping = {"Under Construction": 0, "Ready to move": 1}

house_price["type"].replace(type_mapping, inplace=True)
house_price["age"].replace(age_mapping, inplace=True)
house_price["status"].replace(status_mapping, inplace=True)

# Save the modified dataset to a new CSV file
output_file = "../data/modified_mumbai_house_prices.csv"
house_price.to_csv(output_file, index=False)

print("Data processing completed. Modified dataset saved to", output_file)