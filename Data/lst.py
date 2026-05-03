
# LST

# Define the CHIRPS dataset
chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
# Define the list of countries for West Africa
region_countries = [
    "Tunisia"
]
# Use the USDOS/LSIB_SIMPLE/2017 dataset for country boundaries
countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
# Define the date range for the dataset
start_year = 2000
end_year = 2026

# MODIS dataset
dataset = ee.ImageCollection('MODIS/061/MOD11A1')

# Output directory
output_base_dir = '/kaggle/working/EarthEngineExports'

# Loop through each country
for country in region_countries :
    print(f"Processing {country}...")

    # Adjust folder name for "Cote d'Ivoire"
    folder_name = country.replace(" ", "_")
    output_dir = os.path.join(output_base_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Filter region by country name
    region = countries.filter(ee.Filter.eq('country_na', country))

    # Process each year and month
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            start_date = f"{year:04d}-{month:02d}-01"
            end_date = ee.Date(start_date).advance(1, 'month').format('YYYY-MM-dd').getInfo()

            # Calculate monthly mean
            monthly_mean = dataset.filterDate(start_date, end_date) \
                                  .select('LST_Day_1km') \
                                  .mean() \
                                  .clip(region)

            # Apply conversion: Kelvin to Celsius
            converted_mean = monthly_mean.multiply(0.02).subtract(273.15)

            # 🔹 FIX: Mask NoData pixels outside the country
            converted_mean = converted_mean.updateMask(converted_mean)

            # Export image to file
            file_name = f"LST_MonthlyMean_{year}_{month:02d}.tif"
            file_path = os.path.join(output_dir, file_name)

            geemap.ee_export_image(
                converted_mean,
                filename=file_path,
                scale=1000,
                region=region.geometry(),
                crs='EPSG:4326'
            )

    # Compress and remove original folder
    zip_path = f"{output_dir}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                zf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), output_dir))
    print(f"Compressed {country} folder into {zip_path}")

    # Remove unzipped folder
    for root, _, files in os.walk(output_dir):
        for file in files:
            os.remove(os.path.join(root, file))
    os.rmdir(output_dir)

print("Processing complete!")
