import pypdf
import os

reader = pypdf.PdfReader('Lab 05 Tasks.pdf')
os.makedirs('extracted_tasks_images', exist_ok=True)
count = 0
for p_idx, page in enumerate(reader.pages):
    for img_name, img_obj in page.images.items():
        clean_name = img_name.strip('/')
        fn = f'extracted_tasks_images/p{p_idx}_{clean_name}_{img_obj.name}'
        with open(fn, 'wb') as f:
            f.write(img_obj.data)
        count += 1
print(f'Extracted {count} images')
