import enum
import cv2
import os
import pdb
import numpy as np
from numpy.core.numeric import zeros_like
from numpy.lib.function_base import append, piecewise
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import copy
import time
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

def row_edge_detect(img):
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    height,width=img_gray.shape
    w=9
    step_size=2
    mask=np.zeros_like(img_gray)
    rect=False
    invalid_width=20
    row_min_max=[invalid_width,height-invalid_width]  #* [20,460]
    col_min_max=[invalid_width,width-invalid_width]  #* [20,620]
    # for row in tqdm(range(row_min_max[0],row_min_max[1],step_size)):
    for row in (range(row_min_max[0],row_min_max[1],step_size)):
        part=img_gray[row:row+w,:].copy()
        #* 梯度
        part_blur=cv2.GaussianBlur(part,(5,5),20)
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
    # for col in tqdm(range(col_min_max[0],col_min_max[1],step_size)):
    for col in (range(col_min_max[0],col_min_max[1],step_size)):
        part=img_gray[:,col:col+w].copy().T
        #* 梯度
        part_blur=cv2.GaussianBlur(part,(5,5),20)
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
    rect=False
    invalid_width=20
    region_index=0
    region_dict={}
    row_min_max=[invalid_width,height-invalid_width]  #* [20,460]
    col_min_max=[invalid_width,width-invalid_width]  #* [20,620]
    for row in (range(row_min_max[0],row_min_max[1],1)):
        for col in range(col_min_max[0],col_min_max[1],1):
            if mask[row,col]>0 and mask[row+1,col]==0:
                if sum(mask[row+1:row+step_size,col])>0:
                    mask[row+1:row+step_size,col]=255
        
    return mask

scan_folder='/home/zhujun/MVS/data/scannet/scans_test/scene0707_00'
ref_view=0
src_view=143
imgpath=os.path.join(scan_folder,'images/edge_detect/{:0>8}.jpg'.format(ref_view))
imgpath2=os.path.join(scan_folder,'images/edge_detect/{:0>8}.jpg'.format(src_view))
img=cv2.imread(imgpath)
img2=cv2.imread(imgpath2)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
height,width=img_gray.shape
grad_mask_save_dir=os.path.join(scan_folder,'images/edge_detect','grad')

#! 轮廓检测
start = time.time()
mask,img=row_edge_detect(img)
print(time.time()-start)
start = time.time()
mask_test,img_test=test_edge_detect(img)
print(time.time()-start)
start = time.time()

#* 加入提取的直线
ref_lines_file_path=os.path.join(scan_folder, 'images/save_lines/{:0>8}_lines.txt'.format(ref_view))
ref_lines = np.loadtxt(ref_lines_file_path,delimiter=',')
line_mask=zeros_like(mask)
for i in range(ref_lines.shape[0]):
    ptStart = tuple(ref_lines[i,0:2].astype(np.int32))
    ptEnd = tuple(ref_lines[i,2:].astype(np.int32))
    cv2.line(line_mask, ptStart, ptEnd, 255,2)
line_mask_test=line_mask.copy()
line_mask[mask>0]=255
line_mask_test[mask_test>0]=255
print(time.time()-start)
start = time.time()

#* 轮廓连接
line_mask=mask_process(line_mask)
line_mask_test=mask_process(line_mask_test)
print(time.time()-start)
start = time.time()
region_mask,region_dict=region_detect(line_mask)
print(time.time()-start)
start = time.time()
plt.figure()
plt.imshow(line_mask)
plt.figure()
plt.imshow(line_mask_test)
plt.show()
exit()

#! 区域检测
region_mask,region_dict=region_detect(line_mask)
mask_color=cv2.applyColorMap(cv2.convertScaleAbs(region_mask.astype(np.float32)/region_mask.max()*255,alpha=1),cv2.COLORMAP_JET)
mask_color[region_mask==0]=[0,0,0]

plt.figure()
plt.imshow(region_mask)

region_index=7
img[region_mask==region_index]=[200,0,0]
#! 单应矩阵计算
mask_p0=np.array(region_dict[region_index]).reshape(-1,1,2).astype(np.float32)
p0=mask_p0.copy()
delete_index=[]
for i,p in  enumerate(p0):
    if p[0,0]>30 and p[0,1]<width-20:
        delete_index.append(i)
p0=p0[delete_index]
# pdb.set_trace()
p1, st, err = cv2.calcOpticalFlowPyrLK(img_gray, img_gray2, p0,None)
good_new = p1[st == 1]
good_old = p0[st == 1]
err_valid=err[st==1]
err=200
good_new_sparse=good_new[err_valid<err]
good_old_sparse=good_new[err_valid<err]
err_valid_sparse=good_new[err_valid<err]
sparse_pt=list(range(0,good_new_sparse.shape[0],1))
good_new_sparse=good_new[sparse_pt]
good_old_sparse=good_old[sparse_pt]
err_valid_sparse=err_valid[sparse_pt]

line_vector=good_new_sparse-good_old_sparse
line_vector_norm=np.linalg.norm(line_vector,axis=1)
line_vector_theta=np.arccos(line_vector[:,0]/line_vector_norm)
valid_lines=abs(line_vector_theta-np.mean(line_vector_theta))<10
good_new_sparse=good_new_sparse[valid_lines]
good_old_sparse=good_old_sparse[valid_lines]

# plt.figure()
# plt.subplot(211)
# plt.plot(line_vector_norm)
# plt.subplot(212)
# plt.plot(line_vector_theta)
# plt.draw()
savedir='/home/zhujun/MVS/data/scannet/scans_test/scene0707_00/images/edge_detect/optical_filter'
# img_name=os.path.join(savedir,f'line_vector_norm_err{err}.png')
# plt.savefig(img_name)
# plt.show()


good_new_kp = [cv2.KeyPoint(good_new_sparse[i][0], good_new_sparse[i][1], 1) for i in range(good_new_sparse.shape[0])]
good_old_kp = [cv2.KeyPoint(good_old_sparse[i][0], good_old_sparse[i][1], 1) for i in range(good_old_sparse.shape[0])]
matches=[cv2.DMatch(i,i,0) for i in range(len(good_new_sparse))]
mathcing=cv2.drawMatches(img,good_old_kp,img2,good_new_kp,matches,None)
plt.figure()
plt.imshow(mathcing)
plt.draw()
img_name=os.path.join(savedir,'mathcing.png')
plt.savefig(img_name)
plt.show()
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
# pdb.set_trace()
# plt.show()
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