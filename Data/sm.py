# Soil Moisture
# Define the date range for GLDAS data (soil moisture)
begin_date = '2000-01-01'
end_date = '2026-12-31'
date_format = "%Y-%m-%d"

# Directory in Kaggle to save .tif files
save_dir = '/kaggle/working/GLDAS_Soil_Moisture'
os.makedirs(save_dir, exist_ok=True)

# Function to sanitize country names for safe filenames
def sanitize_country_name(country_name):
    return country_name.replace("'", "").replace(" ", "_").replace(",", "")

# Function to download image as GeoTIFF to Kaggle
def download_image(image, description, region, save_dir, scale=1000):
    path = os.path.join(save_dir, f"{description}.tif")
    geemap.ee_export_image(image, filename=path, scale=scale, region=region, file_per_band=False)
    print(f"Image saved as {path}")

# Function to zip a folder and delete the original folder
def zip_and_delete_folder(folder_path):
    zip_path = f"{folder_path}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
    print(f"Folder zipped as {zip_path}")
    # Delete the original folder
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
    os.rmdir(folder_path)
    print(f"Deleted folder: {folder_path}")

# Loop over each country to download images separately
for country_name in region_countries:
    # Sanitize the country name for filename
    safe_country_name = sanitize_country_name(country_name)
    
    # Create a subdirectory for the country
    country_dir = os.path.join(save_dir, safe_country_name)
    os.makedirs(country_dir, exist_ok=True)
    
    # Filter the specific country
    country = countries.filter(ee.Filter.eq('country_na', country_name))
    geometry = country.geometry()
    
    # Define start and end dates
    current_date = datetime.strptime(begin_date, date_format)
    end_date_obj = datetime.strptime(end_date, date_format)
    
    # Loop through each month within the date range
    while current_date <= end_date_obj:
        # Calculate the end of the current month
        next_month = current_date.replace(day=28) + timedelta(days=4)  # Move to next month
        end_of_month = next_month - timedelta(days=next_month.day)
        
        # Select the GLDAS soil moisture images for the current month
        dataset = ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H') \
            .filterDate(current_date.strftime(date_format), end_of_month.strftime(date_format)) \
            .map(lambda image: image.clip(geometry).mask(ee.Image().paint(geometry, 1)))
        
        # Calculate the monthly average soil moisture
        monthly_soil_moisture = dataset.select('SoilMoi0_10cm_inst').mean()
        
        # Check if image is available for the specific month
        if monthly_soil_moisture:
            # Save the image to the country's directory with sanitized country and date-specific naming
            description = f'GLDAS_SoilMoisture_{safe_country_name}_{current_date.strftime("%Y_%m")}'
            try:
                download_image(monthly_soil_moisture, description, geometry, country_dir, scale=1000)  # Scale set to 1km for GLDAS
            except Exception as e:
                print(f"Error downloading {description}: {e}")
        
        # Move to the next month
        current_date = end_of_month + timedelta(days=1)
    
    # Zip the country's folder and delete the original folder
    zip_and_delete_folder(country_dir)

print("All monthly soil moisture images downloaded, zipped, and folders deleted successfully.")
