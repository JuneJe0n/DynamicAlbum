import os
import numpy as np

import cv2
# from diffusers.utils import load_image    
# from simple_lama_inpainting import SimpleLama
from PIL import Image, ImageOps, ImageFilter

# ----------- Convert all extensions to .png -------------
# def convert2png(open_path, save_path):
#     os.makedirs(save_path, exist_ok=True)
#     num_converts = 0
#     for filename in os.listdir(open_path):
#         file_path = os.path.join(open_path, filename)
#         name, ext = os.path.splitext(filename)
#         ext_lower = ext.lower()
        
#         num_converts += 1
#         with Image.open(file_path) as img:
#             png_filename = name + '.png'
#             png_path = os.path.join(save_path, png_filename)
            
#             if img.mode in ('RGBA', 'LA'):
#                 img.save(png_path, 'PNG')
#             else:
#                 img_rgb = img.convert('RGB')
#                 img_rgb.save(png_path, 'PNG')
#     print(f"\n 🎨 {num_converts} images are converted to PNG and saved at {save_path}.")
    
#     return

# ------------ Expand mask -----------
def expand_mask(open_path, output_path, expansion_pixels=10):
    os.makedirs(output_path, exist_ok=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*expansion_pixels+1, 2*expansion_pixels+1))
    num_expands = 0
    
    for filename in os.listdir(open_path):
        mask_path = os.path.join(open_path, filename)
        
        with Image.open(mask_path) as mask:
            mask_arr = np.array(mask)
            expanded_mask = cv2.dilate(mask_arr, kernel, iterations=1) # expand 3 pixels from every border pixel
            expaneded_mask_to_save = Image.fromarray(expanded_mask, mode='L') # save as gray-scale
            
            mask_dir = os.path.join(output_path, filename)
            expaneded_mask_to_save.save(mask_dir, 'PNG')
        num_expands += 1
        
    print(f"\n 🎨 {num_expands} masks are expanded by {expansion_pixels} pixels for every border pixel.")
    
    return 
            
        
# ------------ Save (origin image, mask) pair -------------
def save_paired_data(image_path, mask_path, paired_dir):
    os.makedirs(paired_dir, exist_ok=True)
    file_name = image_path.split('/')[-1].split('.')[0]
    
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    
    image_dir = os.path.join(paired_dir, f"{file_name}.png")
    mask_dir = os.path.join(paired_dir, f"{file_name}_mask.png")
    
    image.save(image_dir, 'PNG')
    mask.save(mask_dir, 'PNG')
    
    return    

# ------------- Run inpaint ---------------
def run_inpaint(data_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    # filename = image_path.split('/')[-1].split('.')[0]
    # save_path = os.path.join(output_dir, f"{filename}.png")
    origin_dir = os.getcwd()
    lama_path = os.path.join(origin_dir, "lama")
    os.chdir(lama_path)
    print(f"Change Directory to {lama_path}...")

    command = f'PYTHONPATH=. TORCH_HOME={lama_path} python bin/predict.py model.path={lama_path}/big-lama indir={data_path} outdir={output_dir} dataset.img_suffix=.png'
    os.system(command)
    
    os.chdir(origin_dir)
    print(f"Change Directory to {origin_dir}...")
    
    # simple_lama = SimpleLama()
    # print(f"model is loaded.")

    #img_path = "image.png"
    #mask_path = "mask.png"

    # image = Image.open(image_path).resize((1024, 1024))
    # mask = Image.open(mask_path).resize((1024, 1024)).convert('L')

    # mask_image = mask_image.filter(ImageFilter.MaxFilter(13))

    # result = simple_lama(image, mask)
    # result.save(save_path, 'PNG')
    
    return 