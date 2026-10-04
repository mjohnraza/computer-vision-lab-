import urllib.request
import re
import os

headers = {'User-Agent': 'Mozilla/5.0'}

# 1. OpenCV tutorial coins image
try:
    req = urllib.request.Request('https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html', headers=headers)
    html = urllib.request.urlopen(req).read().decode('utf-8', errors='ignore')
    imgs = re.findall(r'src="([^"]+\.(?:png|jpg|jpeg))"', html)
    print("Found images on tutorial page:", imgs)
    for img_rel in imgs:
        if 'coin' in img_rel.lower():
            full_url = 'https://docs.opencv.org/4.x/d3/db4/' + img_rel
            print("Downloading coin from:", full_url)
            urllib.request.urlretrieve(full_url, 'coins.jpg')
            urllib.request.urlretrieve(full_url, 'water_coins.jpg')
            break
except Exception as e:
    print("Error fetching coins:", e)

# 2. Dog image from Packt repository
# https://github.com/PacktPublishing/Computer-Vision-Projects-with-OpenCV-and-Python-3
# Let's search github api or raw urls
try:
    # Check Packt github repo
    req = urllib.request.Request('https://api.github.com/repos/PacktPublishing/Computer-Vision-Projects-with-OpenCV-and-Python-3/git/trees/master?recursive=1', headers=headers)
    import json
    data = json.loads(urllib.request.urlopen(req).read().decode())
    for item in data.get('tree', []):
        if 'dog' in item['path'].lower():
            raw_url = f"https://raw.githubusercontent.com/PacktPublishing/Computer-Vision-Projects-with-OpenCV-and-Python-3/master/{item['path']}"
            print("Found dog image:", item['path'], raw_url)
            urllib.request.urlretrieve(raw_url, 'dog.jpeg')
            urllib.request.urlretrieve(raw_url, 'dogimg.jpg')
            break
except Exception as e:
    print("Error fetching dog:", e)

# 3. Brain MRI image
try:
    req = urllib.request.Request('https://www.tsijournals.com/articles/region-growing-image-segmentation-for-newborn-brain-mri.html', headers=headers)
    html = urllib.request.urlopen(req).read().decode('utf-8', errors='ignore')
    imgs = re.findall(r'<img[^>]+src="([^">]+\.(?:jpg|png|jpeg))"', html)
    print("Found brain images:", imgs[:5])
    for img_url in imgs:
        if 'fig' in img_url.lower() or 'image' in img_url.lower() or 'mri' in img_url.lower():
            if not img_url.startswith('http'):
                img_url = 'https://www.tsijournals.com' + (img_url if img_url.startswith('/') else '/' + img_url)
            print("Downloading brain MRI from:", img_url)
            urllib.request.urlretrieve(img_url, 'brain_mri.png')
            break
except Exception as e:
    print("Error fetching brain MRI:", e)

print("Downloaded files check:")
for f in ['sudoku.png', 'smarties.png', 'coins.jpg', 'water_coins.jpg', 'dog.jpeg', 'brain_mri.png']:
    print(f, os.path.exists(f), os.path.getsize(f) if os.path.exists(f) else 0)
