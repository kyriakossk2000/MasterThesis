import urllib.request
import zipfile
import os

# URL of the zip file
url = 'https://files.grouplens.org/datasets/movielens/ml-20m.zip'

# Download the zip file
zip_file_path, _ = urllib.request.urlretrieve(url)

# Extract only the ratings.csv file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    for file in zip_ref.namelist():
        if file == 'ml-20m/ratings.csv':
            zip_ref.extract(file)

# Rename the extracted file to ratings.csv
os.rename('ml-20m/ratings.csv', 'ratings.csv')

# Remove the extracted folder
os.rmdir('ml-20m')
