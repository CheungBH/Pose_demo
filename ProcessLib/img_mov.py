import os
import shutil


def move_images(source_folder, destination_folder, copy_img):
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                source_file = os.path.join(root, file)
                destination_file = os.path.join(destination_folder, file)
                if copy_img is False:
                    shutil.move(source_file, destination_file)
                    print(f'Moved {source_file} to {destination_file}')
                else:
                    shutil.copy(source_file, destination_file)
                    print(f'Copied {source_file} to {destination_file}')


if __name__ == "__main__":
    source_folder = "/media/hkuit164/WD20EJRX/ESTRNN_dataset/Tennis_demo/AI_model_advanced/highlight_recognition"
    destination_folder = "/media/hkuit164/WD20EJRX/ESTRNN_dataset/Tennis_demo/AI_model_advanced/hl_frame_reduce"
    copy_img = False
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    move_images(source_folder, destination_folder, copy_img)
    print("Done")
