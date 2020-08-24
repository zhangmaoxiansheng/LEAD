import os
data_path = '/mnt/data1/jianing/kitti'

# with open('./train_files.txt','r') as f:
#     lines = f.readlines()
# lines2 = lines.copy()
# for line in lines:
#     con = line.split()
#     frame_index = int(con[1])
#     f_str = "{:010d}.png".format(frame_index)
#     folder = con[0]
#     folder = folder[11:]
#     depth_path1 = os.path.join(
#             data_path,
#             'kitti_depth','train',
#             folder,
#             "proj_depth/groundtruth/image_0{}".format(2),
#             f_str)
#     depth_path2 = os.path.join(
#             data_path,
#             'kitti_depth','train',
#             folder,
#             "proj_depth/groundtruth/image_0{}".format(3),
#             f_str)
#     if not os.path.exists(depth_path1):
#         lines2.remove(line)

# with open('./train_files_p.txt','w') as f:
#     f.writelines(lines2)

with open('./val_files.txt','r') as f:
    lines = f.readlines()
lines2 = lines.copy()
for line in lines:
    con = line.split()
    frame_index = int(con[1])
    f_str = "{:010d}.png".format(frame_index)
    folder = con[0]
    folder = folder[11:]
    depth_path1 = os.path.join(
            data_path,
            'kitti_depth','val',
            folder,
            "proj_depth/groundtruth/image_0{}".format(2),
            f_str)
    depth_path2 = os.path.join(
            data_path,
            'kitti_depth','val',
            folder,
            "proj_depth/groundtruth/image_0{}".format(3),
            f_str)
    depth_path3 = os.path.join(
            data_path,
            'kitti_depth','train',
            folder,
            "proj_depth/groundtruth/image_0{}".format(2),
            f_str)
    depth_path4 = os.path.join(
            data_path,
            'kitti_depth','train',
            folder,
            "proj_depth/groundtruth/image_0{}".format(3),
            f_str)
    if (not os.path.exists(depth_path1) or not os.path.exists(depth_path2)) and (not os.path.exists(depth_path3) or not os.path.exists(depth_path4)):
        lines2.remove(line)

with open('./val_files_p2.txt','w') as f:
    f.writelines(lines2)