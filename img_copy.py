import os
import shutil

def compare_folders(folder1, folder2, output_folder):
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))
    os.makedirs(output_folder, exist_ok=True)
    idx = 0
    #####
    for file1 in files1:
        parts1 = file1.split("_")

        for file2 in files2:

            parts2 = file2.split("_")
            if len(parts1) == 8 and len(parts2) == 8 and parts1[0] == parts2[0] \
                    and parts1[1] == parts2[1] and parts1[2] == parts2[2] \
                    and parts1[3] == parts2[3] and parts1[4] == parts2[4] \
                    and parts1[5] == parts2[5] and parts1[7] == parts2[7]:

                src_path = os.path.join(folder2, file2)
                dst_path = os.path.join(output_folder, file2)
                shutil.copyfile(src_path, dst_path)
                idx += 1

            if len(parts1) == 9 and len(parts2) == 9 and parts1[0] == parts2[0] \
                    and parts1[1] == parts2[1] and parts1[2] == parts2[2] \
                    and parts1[3] == parts2[3] and parts1[4] == parts2[4] \
                    and parts1[5] == parts2[5] and parts1[6] == parts2[6] and parts1[8] == parts2[8]:
                src_path = os.path.join(folder2, file2)
                dst_path = os.path.join(output_folder, file2)
                shutil.copyfile(src_path, dst_path)
                idx += 1

            print(f"{idx} images copied")

folder1 = "/media/hkuit164/Backup/2324_data/0208_high/thermal/images"
folder2 = "/media/hkuit164/Backup/2324_data/0208_high/OneDrive_1_10-30-2023/rgb/20230208_night_rgb"
output_folder = "/media/hkuit164/Backup/2324_data/0208_high/rgb/images"

compare_folders(folder1, folder2, output_folder)
