# NDVI
# Define the date range for the MODIS NDVI dataset
begin_date = '2000-01-01'
end_date = '2026-12-31'
date_format = "%Y-%m-%d"
region_countries = ["Tunisia"]
# Directory in Kaggle to save .tif files
save_dir = '/kaggle/working/NDVI_Images'
os.makedirs(save_dir, exist_ok=True)

# Function to sanitize country names for safe filenames
def sanitize_country_name(country_name):
    return country_name.replace("'", "").replace(" ", "_")

# Function to download image as GeoTIFF with WGS84 projection
def download_image(image, description, region, save_dir, scale=1000, crs="EPSG:4326"):
    path = os.path.join(save_dir, f"{description}.tif")
    geemap.ee_export_image(image, filename=path, scale=scale, region=region, crs=crs, file_per_band=False)
    print(f"Image saved as {path}")

# Function to zip and delete folder
def zip_and_delete_folder(folder_path):
    zip_path = folder_path + ".zip"
    shutil.make_archive(folder_path, 'zip', folder_path)
    shutil.rmtree(folder_path)  # Delete the original folder
    print(f"Zipped and deleted folder: {zip_path}")

# Define start and end dates for filtering
start_date = datetime.strptime(begin_date, date_format)
end_date = datetime.strptime(end_date, date_format)
# Loop over each country to download images separately
for country_name in region_countries:
    # Sanitize the country name for filename
    safe_country_name = sanitize_country_name(country_name)
    
    # Create a separate folder for each country inside the main directory
    country_save_dir = os.path.join(save_dir, safe_country_name)
    os.makedirs(country_save_dir, exist_ok=True)
    
    # Filter the specific country
    country = countries.filter(ee.Filter.eq('country_na', country_name))
    geometry = country.geometry()
    
    # Loop through each month between 2000 and 2023
    current_date = start_date
    while current_date <= end_date:
        # Move to the next month
        next_month = current_date.replace(day=28) + timedelta(days=4)
        next_month = next_month.replace(day=1)  # Set to the first day of the next month
        
        # Filter the MODIS image collection for the current month and country
        dataset = ee.ImageCollection('MODIS/061/MOD13A3') \
            .filterDate(current_date.strftime(date_format), next_month.strftime(date_format)) \
            .map(lambda image: image.clip(geometry))
        
        # Get the NDVI image (one per month)
        ndvi_image = dataset.select('NDVI').first().multiply(0.0001)  # Scale NDVI values
        
        # Save the image to the country's folder with sanitized country and date-specific naming
        description = f'NDVI_{safe_country_name}_{current_date.strftime("%Y_%m")}'
        try:
            download_image(ndvi_image, description, geometry, country_save_dir, scale=1000, crs="EPSG:4326")
        except Exception as e:
            print(f"Error downloading {description}: {e}")
        
        # Move to the next month
        current_date = next_month

    # Zip the country folder and delete the unzipped files
    zip_and_delete_folder(country_save_dir)

print("All images downloaded, zipped, and cleaned up successfully.")
