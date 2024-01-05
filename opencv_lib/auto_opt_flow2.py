import os

src_folder = "/media/hkuit164/Backup/xjl/reverse_cls/classified_video"
save_folder = "/media/hkuit164/Backup/xjl/reverse_cls/classified_video_ofss"


sub_folders = os.listdir(src_folder)
sub_folders_path = [os.path.join(src_folder, sub_folder) for sub_folder in os.listdir(src_folder)]
os.makedirs(save_folder, exist_ok=True)

for save_sub_folder in sub_folders:
    os.makedirs(os.path.join(save_folder, save_sub_folder), exist_ok=True)

for sub_folder, sub_folder_path in zip(sub_folders, sub_folders_path):
    video_paths = [os.path.join(sub_folder_path, video_path) for video_path in os.listdir(sub_folder_path)]

    for video_path in video_paths:
        video_name = os.path.basename(video_path)

        cmd = "python opencv_lib/optical_flow2.py --source {} --output {}/{}/opt_flow_{}".format(video_path, save_folder, sub_folder, video_name)
        print(cmd)
        os.system(cmd)
