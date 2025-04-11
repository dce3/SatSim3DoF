import urllib.request
import os

def download_file(url, dest_directory, chunk_size=8192):
    """
    Download a file from the given URL into the specified destination directory.

    Parameters:
        url (str): The URL to download from.
        dest_directory (str): The directory where the file will be saved.
        chunk_size (int): The size of chunks to use while reading the file (default: 8192 bytes).

    Returns:
        str: The path of the downloaded file.
    """
    # Ensure the destination directory exists; create if it doesn't.
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    
    # Extract the file name from the URL.
    file_name = os.path.basename(url)
    dest_path = os.path.join(dest_directory, file_name)
    
    print(f"Downloading {url} to {dest_path}...")
    
    # Open the URL and the destination file.
    with urllib.request.urlopen(url) as response, open(dest_path, 'wb') as out_file:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:  # End of file.
                break
            out_file.write(chunk)
    
    print("Download completed.")
    return dest_path

if __name__ == "__main__":
    url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp"
    
    # Replace 'your_directory_here' with the actual directory path where you want the file saved.
    destination_directory = "/dtest"
    
    downloaded_file = download_file(url, destination_directory)
    print(f"File downloaded to: {downloaded_file}")
