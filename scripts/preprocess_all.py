import glob
import os

files_pre = glob.glob(f'/data/aigc/tzy/zjw/dreamgaussian/natural-images/*/*rgba.png')
new_data_root = "/data/aigc/tzy/zjw/dreamgaussian/data2"
for file in files_pre:
    data_dir = os.path.basename(os.path.dirname(file))
    name = f"{data_dir}.png"
    os.system(f"cp {file} {os.path.join(new_data_root, name)}")