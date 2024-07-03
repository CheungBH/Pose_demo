import os
import random
import shutil

from tqdm import tqdm


def count_files(directory):
    return len(os.listdir(directory))


def collect_directory_info(root_directory, train_file, val_file, val_portion):
    train = open(train_file, 'a')
    val = open(val_file, 'a')
    for subdir in tqdm(os.listdir(root_directory), desc='gen txt'):
        subdir = os.path.join(root_directory, subdir)
        if os.path.isdir(subdir):
            if '_flipped' not in subdir and '_v05' not in subdir and '_v15' not in subdir:  # Flipped and original video need to be put in the same mode
                # print(root)
                file_count = count_files(subdir)
                line = f"{subdir.split('/')[-1]} {file_count} {os.path.basename(subdir)[3]}\n"
                # line += f"{subdir.split('/')[-1]}_v05 {file_count} {os.path.basename(subdir)[3]}\n"
                # line += f"{subdir.split('/')[-1]}_v15 {file_count} {os.path.basename(subdir)[3]}\n"
                # line += f"{subdir.split('/')[-1]}_flipped {file_count} {os.path.basename(subdir)[3]}\n"
                # line += f"{subdir.split('/')[-1]}_flipped_v05 {file_count} {os.path.basename(subdir)[3]}\n"
                # line += f"{subdir.split('/')[-1]}_flipped_v15 {file_count} {os.path.basename(subdir)[3]}\n"

                if random.random() > val_portion:
                    # with open(train_file, 'a') as train:
                    train.write(line)
                    train.flush()
                else:
                    # with open(val_file, 'a') as val:
                    val.write(line)
                    val.flush()
    train.close()
    val.close()


# Provide the root directory and output file path
root_directory = 'bb-dataset-cropped-upper/images'
train_file = 'bb-dataset-cropped-upper/train.txt'
val_file = 'bb-dataset-cropped-upper/val.txt'
val_portion = 0.1

source_directory = "bb-dataset-cropped-upper/images_hsv"
if os.path.exists(source_directory):
    for file in os.listdir(source_directory):
        shutil.move(os.path.join(source_directory, file), root_directory)
    os.rmdir(source_directory)

if os.path.exists(train_file):  # Check if the file exists
    os.remove(train_file)  # Remove the file
if os.path.exists(val_file):  # Check if the file exists
    os.remove(val_file)  # Remove the file
collect_directory_info(root_directory, train_file, val_file, val_portion)
print("Done generate txt files")
