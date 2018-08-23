# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 09:56:10 2018

@author: Jameswang
"""
"""
代码包含code data images和date data images两种数据增强方式，参数如下：
    --code_images 是否生成code data images
    --date_images  是否生成date data images
    --image_years   date data images 的生成截止日期指定(范围为2018-image_years)
    --image_nums   code data images生成数量
    --noise_sigma  设置噪声系数
    --use_rotate  是否使用旋转
    --normal_mean  设置长度分布均值

用法如下:
    code data images 生成:
    python data_augementation.py --code_images --image_nums 100
    生成数据存放位置：data_argum/result_code_data
    date data images 生成：
    python data_augementation.py --date_images --image_years 2019
    生成数据存放位置：data_argum/result_date_data
    两种数据同时生成：
    python data_augementation.py --code_images --date_images
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import random
from PIL import Image
from skimage import img_as_float
from skimage.util import random_noise
import scipy.misc
import hashlib
import calendar
import glob
import re

def code_data_augem( image_nums, noise_sigma=0.155, use_rotate=True, normal_mean=285 ):
    
    #triming images
    #read the directory files
    imgs = []
    path = "code_trimed/"
    path_trim = "original_code_trim/"
    dir_ = "result_code_data/"
    valid_images = [".jpg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        imgs.append(Image.open(os.path.join(path,f)))
    
    # trimming images to same size
    pil_im = Image.open('original/blank.jpg')
    pil_im = pil_im.resize((28,50))
    box = (4,7,25,41)
    #plt.imshow(imgs[3])
    #plt.show()
    #trimd_imgs = []
    for i in range(len(imgs)):
        print(i)
        img =imgs[i].resize((21,34))
        pil_im.paste(img,box)
        pil_im.save(path_trim+'re_%d.jpg'%i)
        #trimd_imgs.append(pil_im)
        #plt.imshow(trimd_imgs[i])
        #plt.show()
        print("trimd_imgs ok")
    
    #read the trimmed images
    imgs_trimd = []
    valid_images = [".jpg"]
    for f in os.listdir(path_trim):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        imgs_trimd.append(Image.open(os.path.join(path_trim,f)))
        
    #generate data array 
    nums = [[]for i in range(10)]
    for tp in range(10):
        for i in range(image_nums):
            num = random.randint(0,9)
            nums[tp].append(num)
        for i in range(10):
            print("numbers[%d] of %d is: %d" %(tp,i,nums[tp].count(i)))
            
   # generate image list to directory
    tu_images = [[]for i in range (image_nums)]
    final_images = []
    img_name = [[]for i in range (image_nums)]
    for im_num in range(image_nums):
        for  rand_num in range(10):
            tu_images[im_num].append(imgs_trimd[nums[rand_num][im_num]])
            img_name[im_num].append(nums[rand_num][im_num])
        imgs_comb = np.hstack((np.asarray(i) for i in tu_images[im_num]))
        imgs_comb = Image.fromarray(imgs_comb)
        final_images.append(imgs_comb)
        #save_dir = os.path.join(dir_, str(im_num),'.jpg')
        #imgs_comb.save(dir_+'%d.jpg'%im_num)
        print("write image ok...")
       
   # random rotation
    if use_rotate:
        for im_num in range((int)(len(final_images)/2)):
           print("%50 images rotate")
           num = random.randint(0,99)
           final_images[num] = final_images[num].rotate(1)
       
   #长度随机正态分布 + 加入椒盐噪声
    sampleNo = image_nums;
   #mu = 285
    sigma = 10
    np.random.seed(0)
    normal_dis = np.random.normal(normal_mean, sigma, sampleNo )
    for im_num in range((len(final_images))):
        print("output length normal distribution")
        img_len = int(normal_dis[im_num])
        final_images[im_num] = final_images[im_num].resize((img_len, 50)) 
        #sigma = 0.155
        final_images[im_num] = img_as_float(final_images[im_num])
        final_images[im_num] = random_noise(final_images[im_num], var=noise_sigma**2)
       
   #数据存储
    for im_num in range(len(final_images)):
        #plt.imshow(final_images[im_num])
        #plt.show()
        #dir_=os.path.join(dir_+'%d.jpg'%im_num)
        #cv2.imwrite(dir_,final_images[im_num])
        one_img_num = '' ;
        for num_in_image in range(len(img_name[im_num])):
            one_img_num+=str(img_name[im_num][num_in_image])
        image = final_images[im_num]
        hash_digest = hashlib.md5(image.tostring()).hexdigest()
        name_str = one_img_num+'_'+str(image.shape[0])+'_'+str(image.shape[1])+'_'+hash_digest
        print(name_str)
        scipy.misc.imsave(dir_+name_str+'.jpg', image)
    print("code data has stored in:", dir_)
    
#code data argumentaion 
def year_parsing(year):
    first_num = int(year/100)
    second_num = int(year%100)
    return first_num, second_num

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def date_data_augem(image_years, noise_sigma=0.155, use_rotate=False, normal_mean = 255):
    #read the trimmed images
    imgs_trimd = []
    files= glob.glob(os.path.join('original_date_trim/', '*.jpg'))
    files = sorted(files, key = numericalSort)
    #files=sorted(files,key=lambda x: int(os.path.splitext(x)[0]))
    for f in files:
        imgs_trimd.append(Image.open(f))
        
    date_images = []
    data_nums = []
    for year in range(2018,image_years+1):
        for month in range(1,13):
            dayrange = calendar.monthrange(year,month)[1]
            for day in range(dayrange):
                waiting_images = []
                waiting_images.append(imgs_trimd[year_parsing(year)[0]-1])
                waiting_images.append(imgs_trimd[year_parsing(year)[1]-1])
                waiting_images.append(imgs_trimd[33])
                waiting_images.append(imgs_trimd[month-1])
                waiting_images.append(imgs_trimd[32])
                waiting_images.append(imgs_trimd[day])
                waiting_images.append(imgs_trimd[31])
                imgs_combd = np.hstack((np.asarray(i)for i in waiting_images))
                imgs_combd = Image.fromarray(imgs_combd)
                name_str = str(year)
                if month < 10:
                    name_str += '0'+str(month)
                else:
                    name_str +=str(month)
                if day < 9:
                    name_str += '0'+str(day+1)
                else:
                    name_str += str(day+1)
                data_nums.append(name_str)
                date_images.append(imgs_combd)
                print("image combined ok ...")
                
    dir_ = 'result_date_data/'
    sampleNo = len(date_images)*11
    mu = 255
    sigma_distr = 10
    np.random.seed(0)
    normal_dis = np.random.normal(mu, sigma_distr, sampleNo)
    #sigma_noise = 0.155
    for im_num in range(len(date_images)):
        one_img_num = data_nums[im_num]
        for i in range(10):
            height = int(normal_dis[im_num*9+i])
            img = date_images[im_num].resize((height,50))
            if use_rotate:
                img = img.rotate(1)
            img = img_as_float(img)
            img = random_noise(img, var=noise_sigma**2)
            hash_digest = hashlib.md5(img.tostring()).hexdigest()
            name_str = one_img_num+'_'+str(img.shape[0])+'_'+str(img.shape[1])+'_'+hash_digest
            scipy.misc.imsave(dir_+name_str+'.jpg', img)
            print(name_str)
            print('genarate number: %d'%(im_num*9+i+1))

def main(args):
    if args.code_images and args.date_images:
        code_data_augem(args.image_nums, args.noise_sigma, args.use_rotate, args.normal_mean)
        date_data_augem(args.image_years, args.noise_sigma, args.use_rotate, args.normal_mean)
    elif args.code_images:
        code_data_augem(args.image_nums, args.noise_sigma, args.use_rotate, args.normal_mean)
    elif args.date_images:
        date_data_augem(args.image_years, args.noise_sigma, args.use_rotate, args.normal_mean)
    else:
        print('请先指定参数类型！')
    

def parse_args():
    ''' parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_images', action = 'store_true', help = "assign code images")
    parser.add_argument('--date_images',action = 'store_true', help = "assign_date_images")
    parser.add_argument('--image_years',type = int, default = 2019, help="date images must assign years!")
    parser.add_argument('--image_nums', type = int, default = 100, help = 'image numbers')
    parser.add_argument('--noise_sigma', type = int, default = 0.155)
    parser.add_argument('--use_rotate', type = bool, default = False)
    parser.add_argument('--normal_mean', type = int, default = 285)
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
    