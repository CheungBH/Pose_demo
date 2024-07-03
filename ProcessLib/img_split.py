import os
import random
import shutil

def split_folder(folder_path, num_parts):
    # Create the output folders
    output_folders = []
    for i in range(num_parts):
        output_folder = f"output_{i+1}"
        os.makedirs(output_folder, exist_ok=True)
        output_folders.append(output_folder)

    # Get a list of all image files in the input folder
    image_files = [file for file in os.listdir(folder_path)]# if file.endswith(('.jpg', '.jpeg', '.png'))]

    # Shuffle the image files randomly
    random.shuffle(image_files)

    # Calculate the number of images in each part
    num_images_per_part = len(image_files) // num_parts

    # Copy and save each part of the images
    for i in range(num_parts):
        start_index = i * num_images_per_part
        end_index = start_index + num_images_per_part

        # If it's the last part, include any remaining images
        if i == num_parts - 1:
            end_index = None

        # Get the images for this part
        part_images = image_files[start_index:end_index]

        # Copy the images to the respective output folder
        for image in part_images:
            src_path = os.path.join(folder_path, image)
            dst_path = os.path.join(output_folders[i], image)
            shutil.copy(src_path, dst_path)

        print(f"Part {i+1} created with {len(part_images)} images.")

    print("Splitting folder completed successfully.")

# Usage example
folder_path = "/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/cls_sources/cls_sources_0628/image_merge_cls_total/zhy_images"
num_parts = 5
split_folder(folder_path, num_parts)
