import random
import os
import shutil

def moveFile(fileDir):
        #取图片的原始路径
        pathDir = os.listdir(fileDir)
        file_number = len(pathDir)

        #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
        rate = 0.1

        #按照rate比例从文件夹中取一定数量图片
        # pick_number = int(file_number * rate)
        pick_number = 200

        #随机选取picknumber数量的样本图片
        sample = random.sample(pathDir, pick_number)
        print(sample)
        for name in sample:
            print(fileDir + name)
            shutil.move(fileDir + "/" + name, tarDir + "/" + name)
        return

if __name__ == '__main__':
    #源图片文件夹路径
    fileDir = "/media/hkuit164/WD20EJRX/Aiden/hksi/HKSI/datasets/practice_2/cut_frame/total_images"
    #移动到新的文件夹路径
    tarDir = '/media/hkuit164/WD20EJRX/Aiden/hksi/HKSI/datasets/practice_2/selec_frame/auto_selec'
    moveFile(fileDir)
















