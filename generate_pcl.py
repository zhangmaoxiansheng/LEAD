import sys
import re
import numpy as np
from skimage.io import imread
from skimage.transform import resize
#from PIL import Image
def readpfm(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if (sys.version[0]) == '3':
        header = header.decode('utf-8')
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    if (sys.version[0]) == '3':
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    else:
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    if (sys.version[0]) == '3':
        scale = float(file.readline().rstrip().decode('utf-8'))
    else:
        scale = float(file.readline().rstrip())
        
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, shape
def generate_pointcloud(image, depth, ply_file, h_thr = 8, r_thr = 800):
    '''
    Convert RGB image and depth map to 3D color pointcloud
    h_thr is height threshold
    r_thr means range threshold
    for a better visulization, we crop the pointcloud according to h_thr and r_thr
    '''
    focalLength_x = 1050
    focalLength_y = 1050
    centerX = 0.5 * image.shape[0]
    centerY = 0.5 * image.shape[1]
    # focalLength_x = 9.597910e+02
    # focalLength_y = 9.569251e+02
    # centerX = 6.960217e+02
    # centerY = 2.241806e+02
    #depth_scale = 0.5613700555560622

    # get height scale:
    # depth_scale = get_scale(depth)
    # print('unscale height is:', depth_scale)

    points = []    
    for v in range(image.shape[1]):
        for u in range(image.shape[0]):
            #color = image.getpixel((u,v))
            color = image[u,v,:]
            #Z = depth.getpixel((u,v))*depth_scale
            Z = depth[u,v]
            if Z==0: continue
            X = (u - centerX) * Z / focalLength_x
            Y = (v - centerY) * Z / focalLength_y
            if Z > r_thr or Y < -h_thr:
                continue
            points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))
    file = open(ply_file,"w")
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
'''%(len(points),"".join(points)))
    file.close()
def disp2depth(disp):
    depth = 2.5 * 2625.3 / disp
    return depth

if __name__ == '__main__':
    depth = np.load('f0_disp.npy')
    depth = depth.squeeze()
    depth = depth * 50
    
    #depth = disp2depth(disp)
    img = imread('f0.png')
    depth = resize(depth,[img.shape[0],img.shape[1]])
    roi = depth[169:274,619:679]
    roi_mean = np.mean(roi)
    depth[169:274,619:679] = roi_mean
    #print(img.size[1])
    generate_pointcloud(img, depth, './f0.ply')
