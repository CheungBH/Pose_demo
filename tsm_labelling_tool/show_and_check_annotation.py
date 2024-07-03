import shutil

import cv2
import os

from tqdm import tqdm

# Current working directory
cwd = 'bb-dataset-cropped-upper/images'
# Label (0=Normal, 1=Shaking, 2=Hitting)
label = 1


def rename_dir(old_dir, new_dir):
    while os.path.exists(new_dir):
        prefix, ppl = new_dir.rsplit('ppl', maxsplit=1)
        new_dir = prefix + 'ppl' + str(int(ppl) + 1)

    # suffices = ["", "_v05", "_v15"]
    suffices = [""]

    for suffix in suffices:
        os.makedirs(new_dir + suffix)

        for file in os.listdir(old_dir + suffix):
            shutil.move(os.path.join(old_dir + suffix, file), os.path.join(new_dir + suffix, file))

        os.rmdir(old_dir + suffix)

        print(f'{old_dir + suffix} -> {new_dir + suffix}')


folder_list = sorted([folder for folder in os.listdir(cwd) if not folder.endswith('_v05') and not folder.endswith('_v15') and folder[3] == str(label)])
# Loop through each folder in the current working directory
for folder in tqdm(folder_list, desc="Check annotations"):
    folder_path = os.path.join(cwd, folder)
    # Check if the folder is a directory
    if os.path.isdir(folder_path):
        # print(folder)

        while True:
            # Loop through each image file in the folder
            for image_file in sorted(os.listdir(folder_path)):
                # Check if the file is a JPG image
                if image_file.endswith(".jpg"):
                    # Get the path to the image file
                    image_path = os.path.join(folder_path, image_file)

                    # Read the image using OpenCV
                    image = cv2.imread(image_path)

                    # Display the image using OpenCV
                    cv2.imshow(f'{folder}: Class {folder[3]}', image)
                    cv2.waitKey(1000 // 20)

            key = cv2.waitKey(0)
            if key == ord(' '):  # Correct / Neutral
                break
            elif key == ord('r'):  # replay
                continue
            elif key == ord('0'):
                new_name = os.path.join(cwd, folder[:3] + '0' + folder[4:])
                rename_dir(folder_path, new_name)
                break
            elif key == ord('1'):
                new_name = os.path.join(cwd, folder[:3] + '1' + folder[4:])
                rename_dir(folder_path, new_name)
                break
            elif key == ord('2'):
                new_name = os.path.join(cwd, folder[:3] + '2' + folder[4:])
                rename_dir(folder_path, new_name)
                break
            elif key == ord('q'):
                exit(0)

        cv2.destroyAllWindows()
