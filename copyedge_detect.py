import cv2
import os
import pdb
import numpy as np
from numpy.lib.function_base import piecewise
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import copy
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
def region_seg(scan_folder, ref_view,prt_img=False):
    use_new_depth=False
    import cv2
    try:
        ref_img_orig, _, _ = read_img(os.path.join(scan_folder,'edge_detect/{:0>8}.jpg'.format(ref_view)),denoising=False)
    except:
        print('region_seg error')
        return False
    ref_img = cv2.cvtColor(ref_img_orig, cv2.COLOR_RGB2BGR) #! plt也是BGR
    # ref_lines_file_path=os.path.join(scan_folder, 'images/save_lines/{:0>8}_lines.txt'.format(ref_view))
    # ref_lines = np.loadtxt(ref_lines_file_path,delimiter=',')
    # ref_img_orig=plot_lines(ref_img_orig,ref_lines,change_color=False)
    ref_img=copy.deepcopy(ref_img_orig)
    ref_img_gray = cv2.cvtColor(ref_img,cv2.COLOR_RGB2GRAY)
    save_dir=os.path.join(scan_folder, "edge_detect/region_seg")
    mask_save_dir=os.path.join(scan_folder, "edge_detect/region_seg_mask")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)
    mask_filename=os.path.join(mask_save_dir,"{:0>8}_region_seg_mask.npy".format(ref_view))
    height,width,_=ref_img.shape
    patchsize=10
    mask=np.zeros((height,width),dtype=np.float32)
    
    if os.path.exists(mask_filename)  and 0:
        mask=np.load(mask_filename)
        # mask=np.cv2.imread(mask_filename)
    else:
        # start_time=time.time()
        seg_values={}
        for row in tqdm(range(patchsize,height-patchsize)):
            #! 基于相邻灰度差异的方式
            near_gray_gap=-np.ones((width,)).astype(np.float32) #*初值全为-1
            for col in range(patchsize,width-patchsize):
                patch_src_i=ref_img_gray[row,col-patchsize:col+patchsize]
                near_gray_gap[col]=abs(patch_src_i.max()-patch_src_i.min())
            near_gray_gap[near_gray_gap<0]=near_gray_gap.max()
            simi_pixel=np.where(near_gray_gap<0.04)[0] #! 0.05
            pixel_gap=np.ones_like(simi_pixel)*2
            pixel_gap[1:]=simi_pixel[1:]-simi_pixel[:-1]
            seg_start=np.where(pixel_gap>1)[0]
            pixel_gap=np.ones_like(simi_pixel)*2
            pixel_gap[:-1]=simi_pixel[1:]-simi_pixel[:-1]
            seg_end=np.where(pixel_gap>1)[0]
            seg_len=seg_end-seg_start
            seg_valid=seg_len>width*0.01
            seg_start_valid=seg_start[seg_valid] # simi_pixel的分段信息！！
            seg_end_valid=seg_end[seg_valid]
            for seg_i in range(len(seg_end_valid)):
                mask[row,simi_pixel[seg_start_valid[seg_i]:seg_end_valid[seg_i]+1]]=row*width+simi_pixel[seg_start_valid[seg_i]]
                seg_values[row*width+simi_pixel[seg_start_valid[seg_i]]]=len(simi_pixel[seg_start_valid[seg_i]:seg_end_valid[seg_i]+1])
        #* 中间结果
        # mask_color=cv2.applyColorMap(cv2.convertScaleAbs(mask/mask.max()*255,alpha=1),cv2.COLORMAP_JET) #! 彩色深度图
        # Image.fromarray(mask_color).save(os.path.join(save_dir, "{:0>8}_mask0.png".format(ref_view)))
        # print('行分割时间:',time.time()-start_time)
        for row in tqdm(range(patchsize,height-patchsize)):
            mask_row_i=mask[row,:]
            mask_row_i1=mask[row+1,:]
            simi_mask_index=np.logical_and(mask_row_i>0,mask_row_i1>0)
            mask_valid_seg=mask_row_i*simi_mask_index
            mask_valid_seg1=mask_row_i1*simi_mask_index
            for i, mask_value in enumerate(np.unique(mask_valid_seg)):
                if mask_value==0: continue
                mask_value_index=np.where(mask_valid_seg==mask_value)[0]
                if len(mask_value_index)<width*0.01:continue
                mask_value1=mask_valid_seg1[mask_value_index[0]]
                patch_src_i=ref_img[row,mask_value_index,:]
                patch_ref_i=ref_img[row+1,mask_value_index,:]
                ncc_errors=np.mean(
                    np.multiply((patch_src_i-np.mean(patch_src_i)),(patch_ref_i-np.mean(patch_ref_i))))/(np.finfo(np.float32).eps+np.std(patch_src_i)*np.std(patch_ref_i))
                if ncc_errors>0.8:
                    mask_row_i1[mask_row_i1==mask_value1]=mask_value
                    # if mask_value1==26718.0:print('ssssss',row,i)
                    seg_values[mask_value]=seg_values[mask_value1]+seg_values[mask_value]
                    seg_values[mask_value1]=0
        # print('初步分割时间:',time.time()-start_time)
        seg_values_filter={}
        for key in seg_values.keys():
            if seg_values[key]>0.005*width*height:
                seg_values_filter[key]=seg_values[key]
        # new_mask_values=list(range(40,len(seg_values_filter)+50))
        # new_mask_values=list(range(10,200))
        # shuffle(new_mask_values)
        new_mask=np.zeros_like(mask,dtype=np.uint8)
        for i,key in enumerate(seg_values_filter.keys()):
            # new_mask[mask==key]=new_mask_values[i]
            new_mask[mask==key]=i+1
        mask=new_mask
        # cv2.imwrite(mask_filename,mask)
        np.save(mask_filename,mask)
    if prt_img:
        mask_color=cv2.applyColorMap(cv2.convertScaleAbs(mask/mask.max()*255,alpha=1),cv2.COLORMAP_JET) #! 彩色深度图
        mask_color = cv2.cvtColor(mask_color,cv2.COLOR_BGR2RGB)
        mask_0=np.zeros_like(mask_color)
        mask_not0=(mask==1).astype(int)
        mask_0[:,:,0],mask_0[:,:,1],mask_0[:,:,2]=mask_not0,mask_not0,mask_not0
        mask_color=mask_color*mask_0
        
        img_cat=(ref_img_orig*255).astype(np.uint8).copy()
        img_cat[mask==1]=(img_cat[mask==1]*0.5+mask_color[mask==1]*0.5).astype(np.uint8)
        img_cat=cv2.hconcat([(ref_img_orig*255).astype(np.uint8),img_cat])
        Image.fromarray(img_cat).save(os.path.join(save_dir, "{:0>8}_cat.png".format(ref_view)))
    return True

def edge_seg(scan_folder, ref_view,prt_img=False):
    use_new_depth=False
    import cv2
    try:
        ref_img_orig, _, _ = read_img(os.path.join(scan_folder,'edge_detect/{:0>8}_lines.jpg'.format(ref_view)),denoising=False)
    except:
        print('region_seg error')
        return False
    # ref_img = cv2.cvtColor(ref_img_orig, cv2.COLOR_RGB2BGR) #! plt也是BGR
    # ref_lines_file_path=os.path.join(scan_folder, 'images/save_lines/{:0>8}_lines.txt'.format(ref_view))
    # ref_lines = np.loadtxt(ref_lines_file_path,delimiter=',')
    # ref_img_orig=plot_lines(ref_img_orig,ref_lines,change_color=False)
    ref_img=copy.deepcopy(ref_img_orig)
    ref_img_gray = cv2.cvtColor(ref_img,cv2.COLOR_RGB2GRAY)
    save_dir=os.path.join(scan_folder, "edge_detect/region_seg")
    mask_save_dir=os.path.join(scan_folder, "edge_detect/region_seg_mask")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)
    mask_filename=os.path.join(mask_save_dir,"{:0>8}_region_seg_mask.npy".format(ref_view))
    height,width,_=ref_img.shape
    patchsize=3
    mask=np.zeros((height,width),dtype=np.float32)
    drop_num=4
    if os.path.exists(mask_filename)  and 0:
        mask=np.load(mask_filename)
        # mask=np.cv2.imread(mask_filename)
    else:
        # start_time=time.time()
        seg_values={}
        end_index=height-patchsize
        # end_index=30
        for row in tqdm(range(patchsize,end_index)):
            near_gray_gap=-np.ones((width,)).astype(np.float32) #*初值全为-1
            for col in range(patchsize,width-patchsize):
                patch_src_i=ref_img_gray[row-1:row+2:,col-patchsize:col+patchsize]
                patch_src_i=np.sort(patch_src_i.reshape(-1))[drop_num:-drop_num]
                near_gray_gap[col]=abs(patch_src_i.max()-patch_src_i.min())
            near_gray_gap[near_gray_gap<0]=near_gray_gap.max()
            simi_pixel=np.where(near_gray_gap<0.05)[0] #! 0.05
            pixel_gap=np.ones_like(simi_pixel)*2
            pixel_gap[1:]=simi_pixel[1:]-simi_pixel[:-1]
            seg_start=np.where(pixel_gap>1)[0]
            pixel_gap=np.ones_like(simi_pixel)*2
            pixel_gap[:-1]=simi_pixel[1:]-simi_pixel[:-1]
            seg_end=np.where(pixel_gap>1)[0]
            seg_len=seg_end-seg_start
            seg_valid=seg_len>width*0.01
            seg_start_valid=seg_start[seg_valid] # simi_pixel的分段信息！！
            seg_end_valid=seg_end[seg_valid]
            for seg_i in range(len(seg_end_valid)):
                seg_gray=ref_img_gray[row,simi_pixel[seg_start_valid[seg_i]:seg_end_valid[seg_i]+1]]
                seg_gray=np.sort(seg_gray.reshape(-1))[2:-2]
                if (seg_gray.max()-seg_gray.min())<0.2:
                    mask[row,simi_pixel[seg_start_valid[seg_i]:seg_end_valid[seg_i]+1]]=row*width+simi_pixel[seg_start_valid[seg_i]]
                    seg_values[row*width+simi_pixel[seg_start_valid[seg_i]]]=len(simi_pixel[seg_start_valid[seg_i]:seg_end_valid[seg_i]+1])
        #* 中间结果
        # mask_color=cv2.applyColorMap(cv2.convertScaleAbs(mask/mask.max()*255,alpha=1),cv2.COLORMAP_JET) #! 彩色深度图
        # Image.fromarray(mask_color).save(os.path.join(save_dir, "{:0>8}_mask0.png".format(ref_view)))
        # print('行分割时间:',time.time()-start_time)
        for row in tqdm(range(patchsize,end_index)):
            mask_row_i=mask[row,:]
            mask_row_i1=mask[row+1,:]
            # simi_mask_index=np.logical_and(mask_row_i>0,mask_row_i1>0)
            simi_mask_index=mask_row_i1>0
            mask_valid_seg=mask_row_i*simi_mask_index
            mask_valid_seg1=mask_row_i1*simi_mask_index
            for i, mask_value1 in enumerate(np.unique(mask_valid_seg1)):
                if mask_value1==0: continue
                mask_value1_index=np.where(mask_valid_seg1==mask_value1)[0] #* 下一行的每个分段
                if len(mask_value1_index)<width*0.05:continue
                mask_valid_seg=mask_row_i[mask_value1_index] #* 上一行对应分段
                vals = np.unique(mask_valid_seg[mask_valid_seg>0])
                if len(vals)==0:
                    continue
                num_values=np.array([seg_values[v] for v in vals])
                mask_value=vals[num_values==max(num_values)] #* 以数量最多的为准
                mask_value=mask_value[0]
                # mask_value = vals[np.argmax(counts)] #* 得到上一段对应值
                mask_value_index=np.logical_and(mask_valid_seg1==mask_value1,mask_row_i==mask_value)
                if sum(mask_value_index)<width*0.05:continue
                # pdb.set_trace()
                patch_src_i=ref_img[row,mask_value_index,:]
                patch_ref_i=ref_img[row+1,mask_value_index,:]
                # pdb.set_trace()
                ncc_errors=np.mean(
                    np.multiply((patch_src_i-np.mean(patch_src_i)),(patch_ref_i-np.mean(patch_ref_i))))/(np.finfo(np.float32).eps+np.std(patch_src_i)*np.std(patch_ref_i))
                if ncc_errors>0.7:
                    mask_row_i1[mask_row_i1==mask_value1]=mask_value
                    # pdb.set_trace()
                    # if mask_value1==26718.0:print('ssssss',row,i)
                    seg_values[mask_value]=seg_values[mask_value1]+seg_values[mask_value]
                    seg_values[mask_value1]=0
        # print('初步分割时间:',time.time()-start_time)
        seg_values_filter={}
        for key in seg_values.keys():
            if seg_values[key]>0.005*width*height:
                seg_values_filter[key]=seg_values[key]
        # new_mask_values=list(range(40,len(seg_values_filter)+50))
        # new_mask_values=list(range(10,200))
        # shuffle(new_mask_values)
        new_mask=np.zeros_like(mask,dtype=np.uint8)
        for i,key in enumerate(seg_values_filter.keys()):
            # new_mask[mask==key]=new_mask_values[i]
            new_mask[mask==key]=i+1
        mask=new_mask
        # cv2.imwrite(mask_filename,mask)
        np.save(mask_filename,mask)
    if prt_img:
        mask_color=cv2.applyColorMap(cv2.convertScaleAbs(mask/mask.max()*255,alpha=1),cv2.COLORMAP_JET) #! 彩色深度图
        mask_color = cv2.cvtColor(mask_color,cv2.COLOR_BGR2RGB)
        mask_0=np.zeros_like(mask_color)
        valid_mask=mask!=0
        mask_not0=(valid_mask).astype(int)
        mask_0[:,:,0],mask_0[:,:,1],mask_0[:,:,2]=mask_not0,mask_not0,mask_not0
        mask_color=mask_color*mask_0
        
        img_cat=(ref_img_orig*255).astype(np.uint8).copy()
        img_cat[valid_mask]=(img_cat[valid_mask]*0.5+mask_color[valid_mask]*0.5).astype(np.uint8)
        img_cat=cv2.hconcat([(ref_img_orig*255).astype(np.uint8),img_cat])
        Image.fromarray(img_cat).save(os.path.join(save_dir, "{:0>8}_cat.png".format(ref_view)))
    return True


def norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))

scan_folder='/home/zhujun/MVS/data/scannet/scans_test/scene0707_00/images'
ref_view=0

# edge_seg(scan_folder, ref_view,prt_img=True)
imgpath=os.path.join(scan_folder,'edge_detect/{:0>8}.jpg'.format(ref_view))
img=cv2.imread(imgpath)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_gray_norm=img_gray-np.mean(img_gray)
img_grad=np.gradient(img_gray_norm)
img_grad=norm(img_grad)
# fig=plt.figure()
# ax=fig.add_subplot(1,2,1)
# ax.imshow(img,'gray')
# ax=fig.add_subplot(1,2,2)
# ax.imshow(img_grad,'gray')
# plt.show()
# pdb.set_trace()
# 创建CLAHE对象 clipLimit限制对比度，tileGridSize块的大小
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# img_gray = clahe.apply(img_gray)
# img_gray=cv2.GaussianBlur(img_gray,(3,3),10)
# dst2=cv2.addWeighted(img_gray,1.5,dst,-0.5,0)
# cv2.imshow("clahe",dst2)
w=16
thed=0.9
step_size=1
mask=np.zeros_like(img_gray)
rect=False
plt.figure()
drop_num=5
value_gaps=[]
for row in tqdm(range(20,460,step_size)):
    if row!=200:
        continue
    for col in range(20,620,1):
        if col!=360:
            rect=True
            continue
        part=img_gray[row:row+w,col:col+w].copy()
        # part_b,part_g,part_r=img[row:row+w,col:col+w,0],img[row:row+w,col:col+w,1],img[row:row+w,col:col+w,2]
        line=np.sort(part.reshape(-1))[drop_num:-drop_num];line_gap=line.max()-line.min()
        # plt.plot(line,label=f'gray_{line_gap}')
        value_gap=line.max()-line.min()
        value_gaps.append(value_gap)
        # part_median=np.median(part)
        # part[part<=part_median]=0
        # part[part>part_median]=1
        # # print(part)
        # for row_i in range(1,part.shape[0]-1):
        #     for col_i in range(1,part.shape[1]-1):
        #         if part[row_i,col_i]==1:
        #             if part[row_i-1,col_i]+part[row_i+1,col_i]+part[row_i,col_i-1]+part[row_i,col_i+1]<=1:
        #                 part[row_i,col_i]=0
        #         if part[row_i,col_i]==0:
        #             if part[row_i-1,col_i]+part[row_i+1,col_i]+part[row_i,col_i-1]+part[row_i,col_i+1]>=3:
        #                 part[row_i,col_i]=1
        # # print()
        # # print(part[1:-1,1:-1])
        # sum0=np.sum(part[1:-1,1:-1],0)
        # sum0[sum0<3]=0
        # sum0[sum0>12]=14
        # c_num_0=0 #* 连续零的个数
        # c_num_0_s_f=False #* 起始标志
        # c_num_0_s=[] #* 位置
        # c_num_0_s_tmp=0 #* 临时变量
        # for sum0_i,sum0_v in enumerate(sum0):
        #     if sum0_v==0:
        #         if not c_num_0_s_f:
        #             c_num_0_s_tmp=sum0_i
        #             c_num_0_s_f=True
        #         c_num_0+=1
        #     else:
        #         if c_num_0_s_f:
        #             if c_num_0>=4:
        #                 c_num_0_s.append(c_num_0_s_tmp)
        #                 c_num_0_s.append(sum0_i)
        #             c_num_0_s_f=False
        #             c_num_0=0

        # print(c_num_0_s)
        # if len(c_num_0_s)>1: print(value_gap)
        # if value_gap>20:
        #     for c_num_0_s_i in c_num_0_s:
        #         if c_num_0_s_i==0: continue
        #         for w_col in range(w):
        #             w_row=c_num_0_s_i+1
        #             cv2.circle(img, (col+w_row,row+w_col), 1, (0,0,0), 1)
        # print(sum0)
        # up_down=abs(np.sum(part[0:w//2,:]).astype(np.int64)-np.sum(part[w//2:,:]).astype(np.int64))
        # left_right=abs(np.sum(part[:,0:w//2]).astype(np.int64)-np.sum(part[:,w//2:]).astype(np.int64))
        # # left_up=np.sum(part[0:w//2,0:w//2]).astype(np.int64)
        # left_dw=np.sum(part[w//2:,0:w//2]).astype(np.int64)
        # right_up=np.sum(part[0:w//2,w//2:]).astype(np.int64)
        # right_dw=np.sum(part[w//2:,w//2:]).astype(np.int64)
        # mean4=np.mean([left_up,left_dw,right_up,right_dw])
        # part4=[abs(left_up-mean4)/w/w*4,abs(left_dw-mean4)/w/w*4,abs(right_up-mean4)/w/w*4,abs(right_dw-mean4)/w/w*4]
        # part2=[up_down/w/w*2,left_right/w/w*2]
        # # print(row,max(part2),'\t',max(part4))
        # # max_gap=max(max(part2),max(part4))
        # max_gap=max(part2)
        if rect:
            cv2.rectangle(img,(col,row),(col+w,row+w),(0,0,0))
            # print(row,max_gap)
            # print(part)
            print(value_gap)

        if value_gap>100: # max_gap>thed or 
            # print(row,max_gap)
            # part_orig=img_gray[row:row+w,col:col+w].copy()
            # print(part_orig.max()-part_orig.min())
            # print(part_orig)
            # cv2.rectangle(img,(col,row),(col+w,row+w),(255,0,0))
            # cv2.circle(img, (col+w//2,row+w//2), 1, (255,0,0), 1)
            mask_part=mask[row:row+w,col:col+w]
            for w_row in range(1,w-1):
                for w_col in range(1,w-1):
                    if part[w_row,w_col]==0:
                        part_sur=[part[w_col-1,w_row],part[w_col+1,w_row],part[w_col,w_row-1],part[w_col,w_row+1]]
                        if sum(part_sur)>0 and sum(part_sur)<3:
                            mask_part[w_row,w_col]=1
                            cv2.circle(img, (col+w_row,row+w_col), 1, (0,0,0), 1)

# print(part)
# cv2.rectangle(img,(col,row),(col+w,row+w),(255,0,0))
imgr=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# imgr[mask]=0
print(np.sum(mask))
saveimgpath=os.path.join(scan_folder,'edge_detect/{:0>8}_grad.jpg'.format(ref_view,w,int(thed*10)))
cv2.imwrite(saveimgpath,imgr)
cv2.imwrite('rect.jpg',imgr)
print("Saving to ",saveimgpath)
# cv2.imwrite('bin.jpg',mask)
# plt.subplot(211)
# plt.imshow(img[295-20:295+20,20:620])
# plt.subplot(212)
# plt.plot(value_gaps,label='gray')

# # plt.legend()
# # plt.show()
# # pdb.set_trace()