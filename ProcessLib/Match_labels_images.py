import os
import shutil

"""
该程序用来将数据集中的labels的train和val, 在images中分别找出对应的图片, 
并且分成train和val, 这样labels的train和val能和images的train和val一一对应
"""

# 设置路径
images_dir = '/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/pose_cls_sources/pose_cls_0718/total_images'
labels_dir = '/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/pose_cls_sources/pose_cls_0718/labels'

# 创建新的目录来存储整理后的图片
new_images_train_dir = '/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/pose_cls_sources/pose_cls_0718/new_images/train'
new_images_val_dir = '/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/pose_cls_sources/pose_cls_0718/new_images/val'

os.makedirs(new_images_train_dir, exist_ok=True)
os.makedirs(new_images_val_dir, exist_ok=True)

def copy_images(label_dir, image_dir, new_image_dir):
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            image_file = label_file.replace('.txt', '.jpg')
            image_path = os.path.join(image_dir, image_file)
            if os.path.exists(image_path):
                shutil.copy(image_path, new_image_dir)
            else:
                print(f"Image {image_file} not found for label {label_file}")

# 复制训练集的图片
copy_images(os.path.join(labels_dir, 'train'), os.path.join(images_dir), new_images_train_dir)

# 复制验证集的图片
copy_images(os.path.join(labels_dir, 'val'), os.path.join(images_dir), new_images_val_dir)

print("Images copied successfully.")
