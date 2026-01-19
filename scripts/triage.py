import os
import shutil

# 1. The data you pasted (I'll put a small part here as an example)
metadata = {
    "aagfhgtpmv.mp4":{"label":"FAKE","original":"vudstovrck.mp4"},
    "abarnvbtwb.mp4":{"label":"REAL","original":null},
    # ... your script will read the whole JSON file
}

# 2. Define your paths
source_dir = "C:/path/to/your/downloaded/videos"
real_dir = "../data/real"
fake_dir = "../data/fake"

# 3. Sort the files
for filename, info in metadata.items():
    source_path = os.path.join(source_dir, filename)
    
    if os.path.exists(source_path):
        if info["label"] == "REAL":
            shutil.copy(source_path, os.path.join(real_dir, filename))
        else:
            shutil.copy(source_path, os.path.join(fake_dir, filename))
        print(f"Sorted: {filename} as {info['label']}")