import os
import urllib.request
import zipfile
import ssl

# Bypass SSL verification for older academic sites
ssl._create_default_https_context = ssl._create_unverified_context

# The 6 action categories in the KTH dataset
categories = ['walking', 'jogging', 'running', 'boxing', 'handwaving', 'handclapping']
base_url = 'http://www.nada.kth.se/cvap/actions/'
output_dir = 'dataset_kth'

def download_and_extract(category):
    zip_url = f"{base_url}{category}.zip"
    zip_path = f"{category}.zip"
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading {category}.zip...")
    try:
        # Define a custom user agent to avoid HTTP 403 Forbidden errors
        req = urllib.request.Request(zip_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
    except Exception as e:
        print(f"Failed to download {category}: {e}")
        return

    print(f"Extracting {category}.zip...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract all files into the dataset_kth folder
            zip_ref.extractall(output_dir)
    except zipfile.BadZipFile:
        print(f"Failed to extract {category}.zip (Bad Zip File).")
        
    print(f"Cleaning up {category}.zip...\n")
    try:
        os.remove(zip_path)
    except Exception as e:
        print(f"Could not remove {zip_path}: {e}")

def main():
    print("Starting KTH Dataset Download (~2.1 GB)...")
    for category in categories:
        download_and_extract(category)
    print(f"All files have been extracted to the '{output_dir}/' directory.")

if __name__ == '__main__':
    main()
