from utils.utils import *
import glob

def main():
    # define paths
    image_path = "/Users/minair/DynamicAlbum/data/test_10"
    mask_path = "/Users/minair/DynamicAlbum/data/mask"
    expand_mask_path = "/Users/minair/DynamicAlbum/data/expanded_mask"
    paired_dir = "/Users/minair/DynamicAlbum/data/paired_data"
    output_dir = "/Users/minair/DynamicAlbum/data/inpainted_results"
    
    # expand mask
    expand_mask(open_path=mask_path, output_path=expand_mask_path)
    
    # pair (origin image, mask) && run inpaint
    images = sorted(glob.glob(f"{image_path}/*.jpg")+glob.glob(f"{image_path}/*.png"))
    masks = sorted(glob.glob(f"{expand_mask_path}/*.png"))
    for _, (impath, mpath) in enumerate(zip(images, masks)):
        save_paired_data(image_path=impath, mask_path=mpath, paired_dir=paired_dir)
    run_inpaint(data_path=paired_dir, output_dir=output_dir)
    print(f"\n🎨 {len(images)} image is inpainted and saved at {output_dir}.")
    
if __name__ == '__main__':
    main()