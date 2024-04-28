import os
import glob

# 指定要删除图片的文件夹路径
folder_path = 'Urban100/image_SRF_2'

# 获取文件夹中以 LR.png 结尾的所有图片文件路径
image_files = glob.glob(os.path.join(folder_path, '*LR.png'))

# 遍历图片文件列表并删除每个文件
for file_path in image_files:
    os.remove(file_path)