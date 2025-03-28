import os
import numpy as np
from scipy.spatial.distance import cdist
#import torch
from multiprocessing import Pool

#part_num = 5000
src_path = "/dataset/sparse/"
data_path = "/userhome/GMM/S3DIS_node/"
save_path = '/userhome/GMM/S3DIS/'
file_name = 'Area_6_hallway_2.1.ele'
file_path = os.path.join(data_path, file_name)
tet = np.loadtxt(file_path,skiprows=1)
tet = tet[:,1:]
src = os.path.join(src_path, file_name.split('.')[0]+'.npy')
points = np.load(src)
coor = points[:, :3]
label = points[:, 6]
mask = points[:, 10]
label_i = label[mask == 1]
coor_i = coor[mask == 1]
matrix=coor_i[tet.astype(int)-1]
matrix=np.concatenate([matrix,np.ones([tet.shape[0],4,1])],-1)
       
part_num = int(20000*200/tet.shape[0])

num_processes = 16  # You can adjust this based on your system's capabilities

def process_file(i):
    if i == int(points.shape[0] / part_num) - 1:
        data_i = coor[i * part_num:]
    else:
        data_i = coor[i * part_num:(i + 1) * part_num]
    coor_ = np.expand_dims(data_i, 1).repeat(matrix.shape[0], 1)
    matrix_ = np.expand_dims(matrix, [0, 2]).repeat(data_i.shape[0], 0).repeat(5, 2)
    matrix_[:, :, 0, 0, :3] = coor_
    matrix_[:, :, 1, 1, :3] = coor_
    matrix_[:, :, 2, 2, :3] = coor_
    matrix_[:, :, 3, 3, :3] = coor_
    #det = torch.det(torch.tensor(matrix_))
    #det = det.numpy()
    det = np.linalg.det(matrix_)
    is_in = np.all(det > 0, axis=2) | np.all(det < 0, axis=2)
    index = np.argmax(is_in, 1)
    not_in = np.where(np.max(is_in, 1) == False)[0]
    tet_index = tet[index]
    tet_label = label_i[tet_index.astype(int) - 1]
    np.save(os.path.join(save_path, file_name.split('.')[0]+'_label', str(i)+'.npy'),tet_label)
    np.save(os.path.join(save_path, file_name.split('.')[0]+'_notin', str(i)+'.npy'),not_in + i * part_num)
    print('computing ' + file_name + ' in part/parts ' + str(i) + '/' + str(int(points.shape[0] / part_num)))

inds=np.arange(int(points.shape[0] / part_num))
os.mkdir(os.path.join(save_path, file_name.split('.')[0]+'_label'))
os.mkdir(os.path.join(save_path, file_name.split('.')[0]+'_notin'))
with Pool(processes=num_processes) as pool:
    pool.map(process_file, inds)

tet_label_t=list(inds)
not_in_t=list(inds)
for i in os.listdir(os.path.join(save_path, file_name.split('.')[0]+'_label')):
    tet_label_t[int(i.split('.')[0])] = np.load(os.path.join(save_path, file_name.split('.')[0]+'_label',i))
    not_in_t[int(i.split('.')[0])] = np.load(os.path.join(save_path, file_name.split('.')[0]+'_notin',i))
tet_label_t = np.concatenate(tet_label_t, 0)
not_in_t = np.concatenate(not_in_t, 0)
# deal not_in points
dis = cdist(coor[not_in_t], coor_i)
tet_label_t[not_in_t] = np.hstack([label_i[np.argpartition(dis, 3, 1)[:, :3]], 255 * np.ones([not_in_t.shape[0], 1])])
# deal labeled points
tet_label_t[mask == 1] = np.expand_dims(label_i, 1).repeat(4, 1)

print('finish and saving ' + file_name)
np.save(os.path.join(save_path, file_name.split('.')[0]+'.npy'),tet_label_t)

