import dirsync
from glob import glob
from utils import get_img_list, write_to_file

# sync data dir
src_dir = '/private/personal/hanjialu/Datasets/Port_Dataset/'      ### only need modify this line
dst_dir = '/ssd/Port_Dataset/'

dirsync.sync(src_dir, dst_dir, 'sync', verbose=True)

# change img_list inside dst_dir
img_lists = glob(dst_dir+'*/*.txt')

print('=== Modify img_list ===')
for file_name in img_lists:
    print('Modifing img_list: ', file_name)
    img_list = get_img_list(file_name)
    img_list = [line.replace(src_dir, dst_dir) for line in img_list]
    write_to_file(img_list, file_name, override=True)
