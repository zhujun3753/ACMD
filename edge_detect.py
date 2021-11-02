import enum
import cv2
import os
import pdb
import numpy as np
from numpy.core.numeric import zeros_like
from numpy.lib.function_base import append, piecewise
from numpy.lib.shape_base import column_stack
from scipy.signal.filter_design import zpk2tf
from scipy.stats.mstats_basic import count_tied_groups, winsorize
from scipy.stats.stats import mode
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import copy
import time
from numba import cuda
import math
import sys
from data_io import *
from plyfile import PlyData, PlyElement



# read an image
def read_img(filename,denoising=False):
    # img = Image.open(filename) #! RGB
    img=cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # scale 0~255 to 0~1
    if denoising:
        img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    np_img = np.array(img, dtype=np.float32) / 255.
    original_h, original_w, _ = np_img.shape
    # np_img = cv2.resize(np_img, img_wh, interpolation=cv2.INTER_LINEAR)
    return np_img, original_h, original_w

def norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))

def test_edge_detect(img):
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_gray=cv2.GaussianBlur(img_gray,(5,5),20)
    height,width=img_gray.shape
    w=9
    step_size=2
    mask=np.zeros_like(img_gray)
    rect=False
    invalid_width=20
    row_min_max=[invalid_width,height-invalid_width]  #* [20,460]
    col_min_max=[invalid_width,width-invalid_width]  #* [20,620]
    for row in (range(row_min_max[0],row_min_max[1],step_size)):
        part=img_gray[row:row+w,:].copy()
        #* 梯度
        # part_blur=cv2.GaussianBlur(part,(5,5),20)
        part_blur=part
        part_sum=np.sum(part_blur,0).astype(np.float32)
        part_sum_gard=abs(np.gradient(part_sum))
        part_sum_for=np.zeros_like(part_sum_gard,dtype=np.float32)
        part_sum_bak=np.zeros_like(part_sum_gard,dtype=np.float32)
        part_sum_gard[part_sum_gard<20]=0
        part_sum_for[1:-1]=part_sum_gard[2:]-part_sum_gard[1:-1]
        part_sum_bak[1:-1]=part_sum_gard[1:-1]-part_sum_gard[0:-2]
        part_sum_mul=part_sum_bak*part_sum_for
        part_sum_gard[part_sum_mul>0]=0
        mask[row:row+w,part_sum_gard>0]=1
        if rect:
            fig=plt.figure()
            ax=fig.add_subplot(2,2,1)
            ax.imshow(part,'gray')
            ax.set_title('orig')
            ax.axis('off')
            ax=fig.add_subplot(2,2,2)
            ax.imshow(part_blur,'gray')
            ax.set_title('blur')
            ax.axis('off')
            ax=fig.add_subplot(2,2,3)
            cv2.rectangle(img,(col_min_max[0],row),(col_min_max[1],row+w),(0,0,0))
            ax.imshow(img,'gray')
            ax.set_title('sobel')
            ax.axis('off')
            ax=fig.add_subplot(2,2,4)
            ax.plot(part_sum)
            ax.plot(part_sum_gard*10,label='g')
            # ax.plot(part_sum_gard_grad*10,label='gg')
            ax.legend()
            plt.show()
            exit()
    # plt.figure()
    # plt.imshow(mask)
    # plt.show()
    for col in (range(col_min_max[0],col_min_max[1],step_size)):
        part=img_gray[:,col:col+w].copy().T
        #* 梯度
        # part_blur=cv2.GaussianBlur(part,(5,5),20)
        part_blur=part
        part_sum=np.sum(part_blur,0).astype(np.float32)
        part_sum_gard=abs(np.gradient(part_sum))
        part_sum_for=np.zeros_like(part_sum_gard,dtype=np.float32)
        part_sum_bak=np.zeros_like(part_sum_gard,dtype=np.float32)
        part_sum_gard[part_sum_gard<20]=0
        part_sum_for[1:-1]=part_sum_gard[2:]-part_sum_gard[1:-1]
        part_sum_bak[1:-1]=part_sum_gard[1:-1]-part_sum_gard[0:-2]
        part_sum_mul=part_sum_bak*part_sum_for
        part_sum_gard[part_sum_mul>0]=0
        mask[part_sum_gard>0,col:col+w]=1

    return mask,img

def region_detect(edge_mask):
    height,width=edge_mask.shape
    w=1
    step_size=1
    mask=np.zeros_like(edge_mask,dtype=np.int32)
    rect=False
    invalid_width=20
    region_index=0
    region_dict={}
    row_min_max=[invalid_width,height-invalid_width]  #* [20,460]
    col_min_max=[invalid_width,width-invalid_width]  #* [20,620]
    for row in (range(row_min_max[0],row_min_max[1],step_size)):
        # if row !=102:
        #     continue
        part=edge_mask[row,:].copy()
        start_p=[]
        end_p=[]
        start_f=False
        end_f=True
        for part_region_i,part_region_v in enumerate(part):
            if not end_f:
                if part_region_v>0 and part[part_region_i-1]==0:
                    end_p.append(part_region_i)
                    start_f=False
                    end_f=True
            if not start_f and part_region_i<width-1: #* 检测起点
                if part_region_v>0 and part[part_region_i+1]==0:
                    start_p.append(part_region_i)
                    start_f=True
                    end_f=False
            
        num_reg=len(end_p)
        if num_reg<1:
            continue
        start_p=np.array(start_p[:num_reg])
        end_p=np.array(end_p)
        len_reg=end_p-start_p
        start_p=start_p[len_reg>10]
        end_p=end_p[len_reg>10]
        num_reg=len(end_p)
        if num_reg<1:
            continue
        last_row_mask=mask[row-1,:]
        for num_reg_i in range(num_reg):
            last_row_mask_seg=last_row_mask[start_p[num_reg_i]:end_p[num_reg_i]]
            if max(last_row_mask_seg)==0:
                region_index+=1
                region_dict[region_index]=[[start_p[num_reg_i],row],[end_p[num_reg_i],row]]
                mask[row,start_p[num_reg_i]:end_p[num_reg_i]]=region_index
            else:
                last_row_mask_seg=last_row_mask_seg[last_row_mask_seg>0]
                diff_values=np.unique(last_row_mask_seg)
                if len(diff_values)==1:
                    mask[row,start_p[num_reg_i]:end_p[num_reg_i]]=diff_values[0]
                    # if diff_values[0]==72:
                    #     pdb.set_trace()
                    se=region_dict[diff_values[0]]
                    # se.append([[start_p[num_reg_i],row],[end_p[num_reg_i],row]])
                    region_dict[diff_values[0]]=se+[[start_p[num_reg_i],row],[end_p[num_reg_i],row]]
                else:
                    diff_values_num=np.array([len(region_dict[v]) for v in diff_values])
                    # pdb.set_trace()
                    try:
                        max_num_v=diff_values[diff_values_num==max(diff_values_num)][0]
                    except:
                        pdb.set_trace()
                    mask[row,start_p[num_reg_i]:end_p[num_reg_i]]=max_num_v
                    # if max_num_v==72:
                    #     pdb.set_trace()
                    se=region_dict[max_num_v]
                    # se.append([[start_p[num_reg_i],row],[end_p[num_reg_i],row]])
                    region_dict[max_num_v]=se+[[start_p[num_reg_i],row],[end_p[num_reg_i],row]]
                    for diff_values_v in diff_values:
                        if diff_values_v!=max_num_v:
                            region_dict[max_num_v]=region_dict[max_num_v]+region_dict[diff_values_v]
                            mask[mask==diff_values_v]=max_num_v
    mask[:,0:col_min_max[0]]=0
    mask[:,col_min_max[1]:]=0
    return mask,region_dict

def mask_process(edge_mask):
    height,width=edge_mask.shape
    w=1
    step_size=10
    mask=edge_mask.copy()
    invalid_width=20
    row_min_max=[invalid_width,height-invalid_width]  #* [20,460]
    col_min_max=[invalid_width,width-invalid_width]  #* [20,620]
    for row in (range(row_min_max[0],row_min_max[1],1)):
        for col in range(col_min_max[0],col_min_max[1],1):
            if mask[row,col]>0 and mask[row+1,col]==0:
                if sum(mask[row+1:row+step_size,col])>0:
                    mask[row+1:row+step_size,col]=255
    return mask

def region_seg(scan_folder,ref_view):
    imgpath=os.path.join(scan_folder,'images/edge_detect/{:0>8}.jpg'.format(ref_view))
    img=cv2.imread(imgpath)
    mask,img=test_edge_detect(img)
    #* 加入提取的直线
    ref_lines_file_path=os.path.join(scan_folder, 'images/save_lines/{:0>8}_lines.txt'.format(ref_view))
    ref_lines = np.loadtxt(ref_lines_file_path,delimiter=',')
    line_mask=zeros_like(mask)
    for i in range(ref_lines.shape[0]):
        ptStart = tuple(ref_lines[i,0:2].astype(np.int32))
        ptEnd = tuple(ref_lines[i,2:].astype(np.int32))
        cv2.line(line_mask, ptStart, ptEnd, 255,2)
    line_mask[mask>0]=255
    line_mask=mask_process(line_mask)
    region_mask,region_dict=region_detect(line_mask)
    return region_mask,region_dict

def get_region_edge_points(region_mask,img,region_index):
    if region_index>region_mask.max():
        print('Too large index!')
        exit()
    height,width=region_mask.shape
    img_region=img.copy()
    img_region[region_mask==region_index]=[200,0,0]
    if np.sum(np.sum(region_mask==region_index))<height*width*0.005:
        print('Too few points')
        plt.figure()
        plt.imshow(img_region)
        plt.show()
        exit()
    #* 提取区域边界点
    edge_mask=np.zeros_like(region_mask)
    edge_points=[]
    invalid_width=22
    row_min_max=[invalid_width,height-invalid_width]  #* [20,460]
    col_min_max=[invalid_width,width-invalid_width]  #* [20,620]
    for row in (range(row_min_max[0],row_min_max[1],1)):
        for col in (range(col_min_max[0],col_min_max[1],1)):
            if region_mask[row,col]==region_index and \
                (region_mask[row,col-1]!=region_index or region_mask[row,col+1]!=region_index \
                    or region_mask[row-1,col]!=region_index or region_mask[row+1,col]!=region_index):
                edge_points.append([col,row])
                edge_mask[row,col]=1
    edge_points=np.array(edge_points).reshape(-1,2)
    return edge_points,edge_mask

def edge_lines_detect(edge_points,edge_mask):
#* 边界直线检测
#! 对于开环线
    height,width=region_mask.shape

    end_points=[]
    invalid_width=22
    row_min_max=[invalid_width,height-invalid_width]  #* [20,460]
    col_min_max=[invalid_width,width-invalid_width]  #* [20,620]
    for row in (range(row_min_max[0],row_min_max[1],1)):
        for col in (range(col_min_max[0],col_min_max[1],1)):
            if edge_mask[row,col]==1:
                if np.sum(edge_mask[row-1:row+2,col-1:col+2])==2:
                    end_points.append([col,row])
                # if np.sum(edge_mask[row-1:row+2,col-1:col+2])>=4:
                #     print([col,row])
    # print(end_points)
    lines=[]
    if len(end_points)>0:
        for end_p in end_points:
            # print(end_p)
            line=[]
            line.append(end_p)
            edge_mask[end_p[1],end_p[0]]=0
            row_p,col_p=end_p[1],end_p[0]
            while(np.sum(edge_mask[row_p-1:row_p+2,col_p-1:col_p+2])>0):
                last_p=line[-2] if len(line)>2 else [col_p,row_p-1]
                cols=[col_p+(col_p-last_p[0]),col_p-0,col_p-0,col_p-1,col_p+1]
                rows=[row_p+(row_p-last_p[1]),row_p+1,row_p-1,row_p-0,row_p-0]
                find_flag=False
                for row,col in zip(rows,cols):
                    if edge_mask[row,col]==1:
                        edge_mask[row,col]=0
                        if np.sum(edge_mask[row-1:row+2,col-1:col+2])>0:
                            row_p,col_p=row,col
                            line.append([col,row])
                            find_flag=True
                            break
                if find_flag:
                    continue
                cols=[col_p+(col_p-last_p[0]),col_p+1,col_p-1,col_p-1,col_p+1]
                rows=[row_p+(row_p-last_p[1]),row_p+1,row_p-1,row_p+1,row_p-1]
                find_flag=False
                for row,col in zip(rows,cols):
                    if edge_mask[row,col]==1:
                        edge_mask[row,col]=0
                        if np.sum(edge_mask[row-1:row+2,col-1:col+2])>0:
                            row_p,col_p=row,col
                            line.append([col,row])
                            find_flag=True
                            break
            if len(line)>edge_points.shape[0]*0.1 or 0:
                lines.append(line)
    #! 对于闭环线
    start_p=[]
    for p in edge_points:
        if edge_mask[p[1],p[0]]==1:
            if np.sum(edge_mask[p[1]-1:p[1]+2,p[0]-1:p[0]+2])==3:
                start_p=p
                break
    if len(start_p)>0 and np.sum(edge_mask)>edge_points.shape[0]*0.2:
        line=[]
        line.append(start_p)
        edge_mask[start_p[1],start_p[0]]=0
        row_p,col_p=start_p[1],start_p[0]
        while(np.sum(edge_mask[row_p-1:row_p+2,col_p-1:col_p+2])>0):
            last_p=line[-2] if len(line)>2 else [col_p,row_p-1]
            cols=[col_p+(col_p-last_p[0]),col_p-0,col_p-0,col_p-1,col_p+1]
            rows=[row_p+(row_p-last_p[1]),row_p+1,row_p-1,row_p-0,row_p-0]
            find_flag=False
            for row,col in zip(rows,cols):
                if edge_mask[row,col]==1:
                    edge_mask[row,col]=0
                    if np.sum(edge_mask[row-1:row+2,col-1:col+2])>0:
                        row_p,col_p=row,col
                        line.append([col,row])
                        find_flag=True
                        break

            if find_flag:
                continue
            cols=[col_p+(col_p-last_p[0]),col_p+1,col_p-1,col_p-1,col_p+1]
            rows=[row_p+(row_p-last_p[1]),row_p+1,row_p-1,row_p+1,row_p-1]
            find_flag=False
            for row,col in zip(rows,cols):
                if edge_mask[row,col]==1:
                    edge_mask[row,col]=0
                    if np.sum(edge_mask[row-1:row+2,col-1:col+2])>0:
                        row_p,col_p=row,col
                        line.append([col,row])
                        find_flag=True
                        break
        if len(line)>edge_points.shape[0]*0.1 or 0:
            lines.append(line)
    return lines

scan_folder='/home/zhujun/MVS/data/scannet/scans_test/scene0707_00'
ref_view=0
src_view=29
imgpath=os.path.join(scan_folder,'images/edge_detect/{:0>8}.jpg'.format(ref_view))
imgpath2=os.path.join(scan_folder,'images/edge_detect/{:0>8}.jpg'.format(src_view))
img=cv2.imread(imgpath)
img2=cv2.imread(imgpath2)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
height,width=img_gray.shape

region_mask,region_dict=region_seg(scan_folder,ref_view)
region_mask2,region_dict2=region_seg(scan_folder,src_view)

mask_color=cv2.applyColorMap(cv2.convertScaleAbs(region_mask.astype(np.float32)/region_mask.max()*255,alpha=1),cv2.COLORMAP_JET)
mask_color[region_mask==0]=[0,0,0]
mask_color2=cv2.applyColorMap(cv2.convertScaleAbs(region_mask2.astype(np.float32)/region_mask2.max()*255,alpha=1),cv2.COLORMAP_JET)
mask_color2[region_mask2==0]=[0,0,0]
save_dir='/home/zhujun/MVS/data/scannet/scans_test/scene0707_00/images/edge_detect/边界分割'

region_index=1
edge_points,edge_mask=get_region_edge_points(region_mask,img,region_index)
lines=edge_lines_detect(edge_points,edge_mask)

# #* 边界直线检测
# #! 对于开环线
# end_points=[]
# invalid_width=22
# row_min_max=[invalid_width,height-invalid_width]  #* [20,460]
# col_min_max=[invalid_width,width-invalid_width]  #* [20,620]
# for row in (range(row_min_max[0],row_min_max[1],1)):
#     for col in (range(col_min_max[0],col_min_max[1],1)):
#         if edge_mask[row,col]==1:
#             if np.sum(edge_mask[row-1:row+2,col-1:col+2])==2:
#                 end_points.append([col,row])
#             # if np.sum(edge_mask[row-1:row+2,col-1:col+2])>=4:
#             #     print([col,row])
# # print(end_points)
# lines=[]
# if len(end_points)>0:
#     for end_p in end_points:
#         # print(end_p)
#         line=[]
#         line.append(end_p)
#         edge_mask[end_p[1],end_p[0]]=0
#         row_p,col_p=end_p[1],end_p[0]
#         while(np.sum(edge_mask[row_p-1:row_p+2,col_p-1:col_p+2])>0):
#             last_p=line[-2] if len(line)>2 else [col_p,row_p-1]
#             cols=[col_p+(col_p-last_p[0]),col_p-0,col_p-0,col_p-1,col_p+1]
#             rows=[row_p+(row_p-last_p[1]),row_p+1,row_p-1,row_p-0,row_p-0]
#             find_flag=False
#             for row,col in zip(rows,cols):
#                 if edge_mask[row,col]==1:
#                     edge_mask[row,col]=0
#                     if np.sum(edge_mask[row-1:row+2,col-1:col+2])>0:
#                         row_p,col_p=row,col
#                         line.append([col,row])
#                         find_flag=True
#                         break
#             if find_flag:
#                 continue
#             cols=[col_p+(col_p-last_p[0]),col_p+1,col_p-1,col_p-1,col_p+1]
#             rows=[row_p+(row_p-last_p[1]),row_p+1,row_p-1,row_p+1,row_p-1]
#             find_flag=False
#             for row,col in zip(rows,cols):
#                 if edge_mask[row,col]==1:
#                     edge_mask[row,col]=0
#                     if np.sum(edge_mask[row-1:row+2,col-1:col+2])>0:
#                         row_p,col_p=row,col
#                         line.append([col,row])
#                         find_flag=True
#                         break
#         if len(line)>edge_points.shape[0]*0.1 or 0:
#             lines.append(line)
# #! 对于闭环线
# start_p=[]
# for p in edge_points:
#     if edge_mask[p[1],p[0]]==1:
#         if np.sum(edge_mask[p[1]-1:p[1]+2,p[0]-1:p[0]+2])==3:
#             start_p=p
#             break
# if len(start_p)>0 and np.sum(edge_mask)>edge_points.shape[0]*0.2:
#     line=[]
#     line.append(start_p)
#     edge_mask[start_p[1],start_p[0]]=0
#     row_p,col_p=start_p[1],start_p[0]
#     while(np.sum(edge_mask[row_p-1:row_p+2,col_p-1:col_p+2])>0):
#         last_p=line[-2] if len(line)>2 else [col_p,row_p-1]
#         cols=[col_p+(col_p-last_p[0]),col_p-0,col_p-0,col_p-1,col_p+1]
#         rows=[row_p+(row_p-last_p[1]),row_p+1,row_p-1,row_p-0,row_p-0]
#         find_flag=False
#         for row,col in zip(rows,cols):
#             if edge_mask[row,col]==1:
#                 edge_mask[row,col]=0
#                 if np.sum(edge_mask[row-1:row+2,col-1:col+2])>0:
#                     row_p,col_p=row,col
#                     line.append([col,row])
#                     find_flag=True
#                     break

#         if find_flag:
#             continue
#         cols=[col_p+(col_p-last_p[0]),col_p+1,col_p-1,col_p-1,col_p+1]
#         rows=[row_p+(row_p-last_p[1]),row_p+1,row_p-1,row_p+1,row_p-1]
#         find_flag=False
#         for row,col in zip(rows,cols):
#             if edge_mask[row,col]==1:
#                 edge_mask[row,col]=0
#                 if np.sum(edge_mask[row-1:row+2,col-1:col+2])>0:
#                     row_p,col_p=row,col
#                     line.append([col,row])
#                     find_flag=True
#                     break
#     if len(line)>edge_points.shape[0]*0.1 or 0:
#         lines.append(line)


print('len lines: ',len(lines))
plt.figure()
for line in lines:
    line=np.array(line).reshape(-1,2)
    plt.plot(line[:,0],line[:,1])
    print(len(line))
line=lines[0]
line_l=len(line)
line=np.array(line).reshape(-1,2)

dir_x=np.gradient(line[:,0])
dir_x[abs(dir_x)<=0.5]=0
dir_y=np.gradient(line[:,1])
dir_y[abs(dir_y)<=0.5]=0

dir_lx=dir_x.copy()
dir_ly=dir_y.copy()

step_size=10
from scipy import stats

for i in range(step_size,line_l-step_size):
    nums=dir_x[i-step_size:i+step_size]
    nums1=dir_x[i-step_size:i]
    nums2=dir_x[i:i+step_size]
    mode1=stats.mode(nums1)[0][0]
    mode2=stats.mode(nums2)[0][0]
    if mode1==mode2 and dir_x[i]!=mode1:
        if np.sum(nums==mode1)>step_size*2*0.8:
            dir_lx[i]=mode1
for i in range(step_size,line_l-step_size):
    nums=dir_y[i-step_size:i+step_size]
    nums1=dir_y[i-step_size:i]
    nums2=dir_y[i:i+step_size]
    mode1=stats.mode(nums1)[0][0]
    mode2=stats.mode(nums2)[0][0]
    if mode1==mode2 and dir_y[i]!=mode1:
        if np.sum(nums==mode1)>step_size*2*0.8:
            dir_ly[i]=mode1

valid_index=np.logical_and(np.gradient(dir_lx)==0,np.gradient(dir_ly)==0)
start_flag=False
valid_seg=[]
for i in range(len(valid_index)):
    if valid_index[i] and not start_flag:
        start_i=i
        start_flag=True
        continue
    if (start_flag and not valid_index[i]) or (i==len(valid_index)-1 and start_flag):
        end_i=i
        start_flag=False
        if end_i-start_i<len(valid_index)*0.05:
            valid_index[start_i:end_i]=False
        else:
            valid_seg.append([start_i,end_i])

print(valid_seg)
plt.figure()
for seg in valid_seg:
    start_i,end_i=seg
    # print(end_i-start_i)
    plt.plot(line[start_i:end_i,0],line[start_i:end_i,1])

# plt.figure()
# plt.plot(valid_index)
# plt.savefig(os.path.join(save_dir,'有效index'),dpi=720)
valid_seg_index=0
line_seg=line[valid_seg[valid_seg_index][0]:valid_seg[valid_seg_index][1]]
line_region=np.zeros_like(img_gray,dtype=np.uint8)
line_region_mask=np.zeros_like(img_gray,dtype=bool)
img_gray_blur=cv2.GaussianBlur(img_gray,(5,5),20)
sobelx64f = cv2.Sobel(img_gray_blur,cv2.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)
line_region_grad=np.zeros_like(img_gray,dtype=np.uint8)
z1=np.polyfit(line_seg[:,1], line_seg[:,0], 1)
line_seg_fit=[[int(z1[0]*p[1]+z1[1]),p[1]] for p in line_seg]
line_seg_fit=np.array(line_seg_fit).reshape(-1,2)
line_seg_fit_ver_vect=np.array([1/np.sqrt(z1[0]*z1[0]+1),-z1[0]/np.sqrt(z1[0]*z1[0]+1)])

img_gray_copy=img_gray.copy()
for point in line_seg_fit:
    cv2.circle(img_gray_copy, (int(point[0]),int(point[1])), 1, (255,0,0), 0)
plt.figure()
plt.imshow(img_gray_copy,cmap ='gray')

img_gray_f32=img_gray.astype(np.float32)
step_max=5
thed=10
delta_i=[]
for point in line_seg_fit:
    current=img_gray_f32[point[1],point[0]]
    add_i=0
    for i in range(1,step_max+1):
        up_new_point=point+line_seg_fit_ver_vect*i
        up_new_point=[int(up_new_point[0]),int(up_new_point[1])]
        low_new_point=point-line_seg_fit_ver_vect*i
        low_new_point=[int(low_new_point[0]),int(low_new_point[1])]
        upper=abs(img_gray_f32[up_new_point[1],up_new_point[0]]-current)
        lower=abs(img_gray_f32[low_new_point[1],low_new_point[0]]-current)
        if upper>lower and upper>thed:
            point[:]=up_new_point
            add_i=i
            break
        if lower>upper and lower>thed:
            point[:]=low_new_point
            add_i=i
            break
    delta_i.append(add_i)
z2=np.polyfit(line_seg_fit[:,1], line_seg_fit[:,0], 1)
line_seg_fit2=[[int(z2[0]*p[1]+z2[1]),p[1]] for p in line_seg_fit]
line_seg_fit2=np.array(line_seg_fit2).reshape(-1,2)

# pdb.set_trace()
# plt.figure()
# plt.plot(line_seg[:,1],line_seg[:,0],label='line')
# plt.plot(line_seg_fit[:,1],line_seg_fit[:,0],label='fit')
# plt.legend()
for point in line_seg_fit2:
    cv2.circle(img_gray, (int(point[0]),int(point[1])), 1, (255,0,0), 0)
plt.figure()
plt.imshow(img_gray,cmap ='gray')

plt.show()
exit()

# pdb.set_trace()

mask_p0=np.array(region_dict[region_index]).reshape(-1,1,2).astype(np.float32)
p0=mask_p0.copy()
delete_index=[]
for i,p in  enumerate(p0):
    if p[0,0]>30 and p[0,1]<width-20:
        delete_index.append(i)
p0=p0[delete_index]
p1, st, err = cv2.calcOpticalFlowPyrLK(img_gray, img_gray2, p0,None)
good_new = p1[st == 1]
good_old = p0[st == 1]
err_valid=err[st==1]

sparse_pt=list(range(0,good_new.shape[0],1))
good_new_sparse=good_new[sparse_pt]
good_old_sparse=good_old[sparse_pt]
err_valid_sparse=err_valid[sparse_pt]

good_new_kp = [cv2.KeyPoint(good_new_sparse[i][0], good_new_sparse[i][1], 1) for i in range(good_new_sparse.shape[0])]
good_old_kp = [cv2.KeyPoint(good_old_sparse[i][0], good_old_sparse[i][1], 1) for i in range(good_old_sparse.shape[0])]
matches=[cv2.DMatch(i,i,0) for i in range(len(good_new_sparse))]
mathcing=cv2.drawMatches(img_region,good_old_kp,img2,good_new_kp,matches,None)

# plt.figure()
# plt.imshow(mathcing)
# plt.draw()

target_dict=[]
for i in range(good_new_sparse.shape[0]):
    x,y=int(good_new_sparse[i][0]),int(good_new_sparse[i][1])
    if region_mask2[y,x]>0:
        target_dict.append(region_mask2[y,x])
from scipy import stats
m=stats.mode(np.array(target_dict))[0][0]
img_region2=img2.copy()
img_region2[region_mask2==m]=[200,0,0]
plt.figure()
plt.imshow(img_region)
plt.figure()
plt.imshow(img_region2)

pdb.set_trace()
plt.show()
exit()




savedir='/home/zhujun/MVS/data/scannet/scans_test/scene0707_00/images/edge_detect/optical_filter'
img_name=os.path.join(savedir,'mathcing.png')
plt.savefig(img_name)
# plt.show()
# pdb.set_trace()

# plt.show()
# H, mask = cv2.findHomography(good_new_sparse, good_old_sparse, cv2.RANSAC,5.0)
# wrap = cv2.warpPerspective(img2, H, (img2.shape[1]+img2.shape[1] , img2.shape[0]+img2.shape[0]))
# img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# wrap[0:img2.shape[0], 0:img2.shape[1]] = (wrap[0:img2.shape[0], 0:img2.shape[1]]*0.5).astype(np.uint8)+ (img1*0.5).astype(np.uint8)
# rows, cols = np.where(wrap[:,:,0] !=0)
# min_row, max_row = min(rows), max(rows) +1
# min_col, max_col = min(cols), max(cols) +1
# result = wrap[min_row:max_row,min_col:max_col,:]#去除黑色无用部分
# plt.figure()
# plt.imshow(result)


ref_depth_gt = read_pfm(os.path.join(scan_folder,'depth/{:0>8}.pfm'.format(ref_view)))[0]
ref_depth_gt=np.squeeze(ref_depth_gt)
ref_intrinsics, ref_extrinsics = read_cam_file(os.path.join(scan_folder,'cams_1/{:0>8}_cam.txt'.format(ref_view)))[0:2]
src_intrinsics, src_extrinsics = read_cam_file(os.path.join(scan_folder,'cams_1/{:0>8}_cam.txt'.format(src_view)))[0:2]
K1=np.mat(ref_intrinsics)
R1=np.mat(ref_extrinsics[0:3,0:3])
t1=np.mat(ref_extrinsics[0:3,-1]).T
K2=np.mat(src_intrinsics)
R2=np.mat(src_extrinsics[0:3,0:3])
t2=np.mat(src_extrinsics[0:3,-1]).T
Rr=R2*R1.T
tr=R2*(R2.T*t2-R1.T*t1)
tmp=Rr-K2.I*H*K1
d1=np.linalg.norm(tmp[0,:])/abs(tr[0,0])
n1=(tmp[0,:]/tr[0]*d1).T
height, width = ref_depth_gt.shape[:2]
x_grid, y_grid = np.meshgrid(np.arange(0, width), np.arange(0, height))
valid_points=region_mask==region_index
x, y = x_grid[valid_points], y_grid[valid_points]
p1 = np.matmul(np.linalg.inv(ref_intrinsics), np.vstack((x, y, np.ones_like(x))))
z1=-1*d1/np.matmul(np.array(n1.T),p1)
# p1=np.mat(np.vstack((x, y, np.ones_like(x))))
# z1=d1/(n1.T*K1.I*p1)
ref_depth_fit=ref_depth_gt.copy()
ref_depth_fit[valid_points]=z1.reshape(-1)




save_dir=os.path.join(scan_folder, 'images/edge_detect/single_depth_comp')
os.makedirs(save_dir, exist_ok=True)
plyfilename=os.path.join(save_dir,f'view_{ref_view}_frame_part.ply')
vertexs = []
vertex_colors = []
height, width = ref_depth_gt.shape[:2]
x_grid, y_grid = np.meshgrid(np.arange(0, width), np.arange(0, height))
valid_points=region_mask!=region_index
valid_points=region_mask!=-1
x, y = x_grid[valid_points], y_grid[valid_points]
img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
color = img[valid_points]
depth = ref_depth_fit[valid_points]
xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics), np.vstack((x, y, np.ones_like(x))) * depth)
xyz_world = np.matmul(np.linalg.inv(ref_extrinsics), np.vstack((xyz_ref, np.ones_like(x))))[:3]
vertexs.append(xyz_world.transpose((1, 0)))
vertex_colors.append((color).astype(np.uint8))

vertexs = np.concatenate(vertexs, axis=0) #* [N，3]
vertex_colors = np.concatenate(vertex_colors, axis=0) #* [N，3]
vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]) # u1 uint8
vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
for prop in vertexs.dtype.names:
    vertex_all[prop] = vertexs[prop]
for prop in vertex_colors.dtype.names:
    vertex_all[prop] = vertex_colors[prop]
el = PlyElement.describe(vertex_all, 'vertex')
PlyData([el]).write(plyfilename)
print("saving the final model to", plyfilename)

pdb.set_trace()
plt.show()
exit()
# draw the tracks
color = np.random.randint(0, 255, (len(p0), 3))
mask = np.zeros_like(img)
mask_opti_flow=np.zeros_like(line_mask)
for i, (new, old) in enumerate(zip(good_new, good_old)):
    # if err_valid[i]>5:
    #     continue
    a, b = new.ravel()
    # pdb.set_trace()
    c, d = old.ravel()
    mask = cv2.line(mask, (a, b), (c, d), color[i].tolist())
    img = cv2.circle(img, (a, b), 1, color[i].tolist(), -1)
    # pdb.set_trace()
    if int(b)<mask_opti_flow.shape[0] and int(a)<mask_opti_flow.shape[1]:
        mask_opti_flow[int(b),int(a)]=1
# img = cv2.add(img2, mask)
img=cv2.addWeighted(img2,0.7,mask,0.3,0)
plt.figure()
plt.imshow(img)
plt.figure()
plt.imshow(mask_opti_flow)
plt.show()
# saveimgpath=os.path.join(grad_mask_save_dir,'{:0>8}_opti_flow.jpg'.format(ref_view))
# cv2.imwrite(saveimgpath,img)