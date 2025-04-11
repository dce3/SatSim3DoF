""" This code is to download the de440.bsp planetary ephemeris from NASA's 
Navigation and Ancillary Information Facility's FTP. The file cannot be added 
to the github directly because it is over 100mb"""

import os
import urllib.request


# URL of the file to download
url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp"

# Specify the directory where the file should be saved
download_dir = 'ephems'  # Replace with your desired directory

# Ensure that the download directory exists; if not, create it
os.makedirs(download_dir, exist_ok=True)

# Extract the file name from the URL
file_name = os.path.basename(url)
file_path = os.path.join(download_dir, file_name)

# Check if the file already exists
if not os.path.exists(file_path):
    print(f"Downloading {file_name} to {download_dir}...")
    try:
        # Download the file from the URL and save it to file_path
        urllib.request.urlretrieve(url, file_path)
        print("Download complete!")
    except Exception as e:
        print(f"An error occurred during download: {e}")
else:
    print(f"{file_name} already exists in {download_dir}.")
