import os

def main():
    folder_path = "/media/hkuit164/Backup/2324_data/0206+0208_high/thermal/pose/train"
    image_txt_file = "/media/hkuit164/Backup/2324_data/0206+0208_high/thermal/pose/trainimage_names.txt"
    compare_txt_file = "/media/hkuit164/Backup/2324_data/0206+0208_high/thermal/yolo/train (copy).txt"
    output_txt_file1 = "/media/hkuit164/Backup/2324_data/0206+0208_high/thermal/pose/trainimage_names.txt"
    output_txt_file2 = "/media/hkuit164/Backup/2324_data/0206+0208_high/thermal/pose/differences.txt"

    images = [f for f in os.listdir(folder_path)
              if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))]

    with open(image_txt_file, "w") as f:
        for image in images:
            f.write(f"{image}\n")

    with open(image_txt_file, "r") as f1:
        file1_lines = f1.readlines()

    with open(compare_txt_file, "r") as f2:
        file2_lines = f2.readlines()

    sorted_lines1 = sorted(file1_lines, reverse=True)
    sorted_lines2 = sorted(file2_lines, reverse=True)

    with open(output_txt_file1, "w") as output1:
        for line1 in sorted_lines1:
            output1.write(line1)
    with open(output_txt_file2, "w") as output2:
        for line2 in sorted_lines2:
            output2.write(line2)

if __name__ == "__main__":
    main()
