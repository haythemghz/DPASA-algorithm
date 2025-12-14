
import os
import urllib.request
import time

BASE_URL = "https://raw.githubusercontent.com/thieu1995/opfunu/master/opfunu/cec_based/data_2022/"
DEST_DIR = "sdm-algorithm-git/cec2022_impl/data_2022"

if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)

files = []
# F1 (Already have, but safe to overwrite/skip)
# Shift data 1-12
for i in range(1, 13):
    files.append(f"shift_data_{i}.txt")

# Matrix data M_1_D10 to M_12_D10
for i in range(1, 13):
    files.append(f"M_{i}_D10.txt")

# Shuffle data for F6, F7, F8
for i in [6, 7, 8]:
    files.append(f"shuffle_data_{i}_D10.txt")

print(f"Downloading {len(files)} files to {DEST_DIR}...")

for filename in files:
    url = BASE_URL + filename
    dest = os.path.join(DEST_DIR, filename)
    
    if os.path.exists(dest):
        print(f"Skipping {filename} (exists)")
        continue
        
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, dest)
        time.sleep(0.1) # Be nice
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

print("Download complete.")
