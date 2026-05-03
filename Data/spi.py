# Define the CHIRPS dataset
chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
# Define the list of countries 
region_countries = ["Tunisia"]
# Use the USDOS/LSIB_SIMPLE/2017 dataset for country boundaries
countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
# Directory in Kaggle to save .tif files
save_dir = '/kaggle/working/SPI_Images'
os.makedirs(save_dir, exist_ok=True)
# Function to sanitize country names for safe filenames
def sanitize_country_name(country_name):
    return country_name.replace("'", "").replace(" ", "_")
# Function to get climatological monthly statistics (1981-2010 standard period)
def get_monthly_climatology(month):
    """Calculate mean and stdDev of monthly precipitation sums for specific month across 30 years"""
    years = ee.List.sequence(1981, 2025)
    
    def process_year(year):
        year = ee.Number(year)
        monthly_sum = chirps \
            .filter(ee.Filter.calendarRange(year, year, 'year')) \
            .filter(ee.Filter.calendarRange(month, month, 'month')) \
            .select('precipitation') \
            .sum()
        return monthly_sum.set('year', year)
    
    yearly_sums = ee.ImageCollection.fromImages(years.map(process_year))
    mean = yearly_sums.mean().rename('mean')
    std_dev = yearly_sums.reduce(ee.Reducer.stdDev()).rename('stdDev')
    
    return mean.addBands(std_dev)

# Function to calculate SPI for a country, year, and month
def calculate_spi(country_geometry, year, month):
    # Get climatological statistics for this month
    climatology = get_monthly_climatology(month)
    mean = climatology.select('mean')
    std_dev = climatology.select('stdDev')
    
    # Current month precipitation sum
    current_month_precipitation = chirps \
        .filter(ee.Filter.calendarRange(year, year, 'year')) \
        .filter(ee.Filter.calendarRange(month, month, 'month')) \
        .select('precipitation') \
        .sum() \
        .rename('current_month_sum')
    # Calculate SPI: (current - mean) / stdDev
    spi = current_month_precipitation.subtract(mean).divide(std_dev).rename('SPI')
    spi = spi.max(-3).min(3)
    return spi.clip(country_geometry)

# Process SPI for each country
for country_name in region_countries:
    print(f"Processing SPI for {country_name}...")
    # Sanitize country name for file paths
    safe_country_name = sanitize_country_name(country_name)
    # Create a separate folder for the country
    country_save_dir = os.path.join(save_dir, safe_country_name)
    os.makedirs(country_save_dir, exist_ok=True)
    # Get the geometry for the country (server-side)
    country = countries.filter(ee.Filter.eq('country_na', country_name)).first()
    geometry = country.geometry()
    # Get region bounds for export (client-side safe)
    region_info = geometry.bounds().getInfo()
    export_region = ee.Geometry.Polygon(region_info['coordinates'])
    
    # Process each year and month
    for year in range(2000, 2025):
        for month in range(1, 13):
            print(f"  Processing {year}-{month:02d}...")
            spi_image = calculate_spi(geometry, year, month)
            description = f"{safe_country_name}_SPI_{year}_{month:02d}"
            file_path = os.path.join(country_save_dir, f"{description}.tif")
            # Export SPI image
            geemap.ee_export_image(
                spi_image,
                filename=file_path,
                scale=1000,  # ~5km matching CHIRPS resolution
                region=export_region,
                file_per_band=False
            )
            print(f"    Exported: {description}")
    # Compress the folder into a ZIP file
    zip_file_path = f"/kaggle/working/{safe_country_name}.zip"
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(country_save_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, country_save_dir)
                zipf.write(file_path, arcname)
    # Delete the country folder to save space
    shutil.rmtree(country_save_dir)
    print(f"SPI data for {country_name} saved and compressed as {zip_file_path}.")
