video_folders1 = sorted(os.listdir(path1))
import shutil
import numpy as np


def MerageThermalPicture_300A100(path1, path2):
    video_folders1 = sorted(os.listdir(path1))
    video_folders2 = sorted(os.listdir(path2))

    for video_folder1 in video_folders1:
        folder_path1 = os.path.join(path1,video_folder1)
        same_name1 = folder_path1[folder_path1.find("Afternoon_")+10:]
        new_name = "2022_10_25_sunny_" + same_name1
        newFolder_path1 = os.path.join(path1,new_name)
        os.rename(folder_path1, newFolder_path1)
        for video_folder2 in video_folders2:
            folder_path2 = os.path.join(path2,video_folder2)
            same_name2 = folder_path2[folder_path2.find("sunny_")+6:]
            if same_name1 == same_name2:
                os.remove(newFolder_path1)
                print("deleted")
                break
def FindRelatedPictures(path1, path2, path3):
    image_folders1 = sorted(os.listdir(path1))
    for image_folder1 in image_folders1:
        oldName = image_folder1
        image_folder1 = image_folder1.replace("thermal", "rgb")
        fiel_path1 = os.path.join(path1,oldName)
        file_path2 = os.path.join(path2,image_folder1)
        file_path3 = os.path.join(path3,image_folder1)
        if os.path.isfile(file_path2):
            shutil.copyfile(file_path2, file_path3)
        else:
            os.remove(fiel_path1)


if __name__ == '__main__':
    #MerageThermalPicture_300A100('/media/hkuit164/Backup/yolov3-channel-and-layer-pruning/data/2022-10-25-afternoon-100-thermal/temp',"/media/hkuit164/Backup/thermal_sunny")
    FindRelatedPictures("/media/hkuit164/Backup/ThermalProject/thermal", "/media/hkuit164/Backup/ThermalProject/rgb/20221103_cloudy_rgb", "/media/hkuit164/Backup/ThermalProject/3")
