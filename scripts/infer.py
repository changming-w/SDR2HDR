#!/usr/bin/python3
# # -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import time
import datetime
import numpy as np
import cv2
import argparse
import tensorflow as tf
import warnings
from scipy.interpolate import CubicSpline

from save_lut_bin import save_lut_bin
from gainmap_utils import read_bin_as_np,ffmpeg_image2hdr,ffmpeg_image2sdr,ffmpeg_frame2hdr,rgb_to_Y,draw_plt,ffmpeg_frame2hdrNv
from gainmap_utils import write_heif
from gainmap_utils import postprocess_gainmap1dlut,postprocess_gainmap
# from gainmap_utils2 import ffmpeg_image2sdr,draw_plt,postprocess_gainmap
from tools.videoio_ffmpeg import VideoReader, VideoWriter, HDRVideoReader

warnings.filterwarnings("ignore")


#### tflite inference
class GMNet_infers():
    def __init__(self, model_path, is_quantity=False, use_gpu=False):
        self.model_path = model_path

        if not use_gpu:
            self.interpreter = tf.lite.Interpreter(model_path)
        else:
            # GPU delegate 加载，兼容 TF 1.14 / 2.8.x / 2.10+
            gpu_delegate = None
            # 方式1: TF 2.10+ 使用 GpuDelegateOptionsV2
            if hasattr(tf.lite.experimental, 'GpuDelegateOptionsV2'):
                gpu_delegate_options = tf.lite.experimental.GpuDelegateOptionsV2(
                    inference_priority1=tf.lite.experimental.GpuDelegateOptionsV2.Priority.SPEED,
                    inference_priority2=tf.lite.experimental.GpuDelegateOptionsV2.Priority.MIN_MEMORY_USAGE,
                    inference_priority3=tf.lite.experimental.GpuDelegateOptionsV2.Priority.AUTOMATIC
                )
                gpu_delegate = tf.lite.experimental.load_delegate(
                    'libtensorflowlite_gpu_delegate.dll',
                    gpu_delegate_options
                )
            # 方式2: TF 1.15 / 2.8.x 使用 load_delegate (无 options)
            elif hasattr(tf.lite.experimental, 'load_delegate'):
                gpu_delegate = tf.lite.experimental.load_delegate(
                    'libtensorflowlite_gpu_delegate.dll'
                )
            # 方式3: TF 1.14 无 load_delegate，用 ctypes 手动加载
            else:
                import ctypes
                lib_path = 'libtensorflowlite_gpu_delegate.dll'
                try:
                    lib = ctypes.pydll.LoadLibrary(lib_path)
                    # TF 1.14 的 Interpreter 构造函数接受 delegate_bodies
                    gpu_delegate = lib
                except OSError as e:
                    print(f'Warning: Failed to load GPU delegate from {lib_path}: {e}')
                    print('Falling back to CPU interpreter')

            if gpu_delegate is not None:
                self.interpreter = tf.lite.Interpreter(model_path, experimental_delegates=[gpu_delegate])
            else:
                print('GPU delegate unavailable, using CPU')
                self.interpreter = tf.lite.Interpreter(model_path)



        self.in_size = [self.interpreter.get_input_details()[0]['shape'][2],
                        self.interpreter.get_input_details()[0]['shape'][1]]  # [w,h]
        self.mean = [0.0, 0.0, 0.0]
        self.std = [1.0, 1.0, 1.0]
        self.is_quantity = is_quantity
        if 'uint8' in model_path:
            self.is_quantity = True
        self.mode = 'resize'
        self.padding = [0]*4

        print('Debug : interpreter == ', self.interpreter.get_input_details(), self.interpreter.get_output_details())

    def preprocess(self, in_img, mode='resize'):

        out_img = np.array(in_img).astype(np.float32)

        if mode == 'resize':
            out_img = cv2.resize(out_img, self.in_size, interpolation=cv2.INTER_CUBIC)
        elif mode == 'pad':
            scale = min(self.in_size[0]/out_img.shape[1], self.in_size[1]/out_img.shape[0])
            out_img = cv2.resize(out_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            self.padding = [(self.in_size[1]-out_img.shape[0])//2,
                            (self.in_size[0]-out_img.shape[1])//2,
                            (self.in_size[1]-out_img.shape[0])-(self.in_size[1]-out_img.shape[0])//2,
                            (self.in_size[0]-out_img.shape[1])-(self.in_size[0]-out_img.shape[1])//2]
            pad_img = np.pad(out_img, ((self.padding[0], self.padding[2]),
                             (self.padding[1], self.padding[3]), (0, 0)), 'constant', constant_values=0)
            out_img = pad_img

        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        out_img = np.array(out_img).astype(np.float32) / 255.0

        out_img = (out_img - np.array(self.mean)) / np.array(self.std)

        return out_img

    def postprocess(self, in_img, out_size=[], norm_max=False, mode='resize'):
        out_img = in_img
        if mode == 'resize':
            if len(out_size) == 2:
                out_img = cv2.resize(out_img, out_size, interpolation=cv2.INTER_LINEAR)
        elif mode == 'pad':
            out_img = out_img[self.padding[0]:out_img.shape[0]-self.padding[2],
                              self.padding[1]:out_img.shape[1]-self.padding[3], :]
            if len(out_size) == 2:
                out_img = cv2.resize(out_img, out_size, interpolation=cv2.INTER_LINEAR)
        if norm_max:
            tmp_img = (out_img-np.min(out_img))/(np.max(out_img)-np.min(out_img))
            out_img = tmp_img
        out_img = np.array(out_img*255).astype(np.uint8)

        return out_img

    def __call__(self, in_img):
        in_h, in_w, in_ch = in_img.shape
        out_img = self.preprocess(in_img, mode=self.mode)

        # out_img = np.transpose(out_img, (2, 0, 1))
        out_img = np.concatenate([out_img[:, :, ::-1], np.zeros_like(out_img[:, :, :1])], axis=-1)

        in_tensor = np.expand_dims(out_img, axis=0).astype(np.float32)
        # print('Debug : in_tensor shape == ', in_tensor.shape)
        if self.is_quantity:
            scale = self.interpreter.get_input_details()[0]['quantization'][0]
            zero_points = self.interpreter.get_input_details()[0]['quantization'][1]
            in_tensor = (in_tensor / scale + zero_points)
            in_tensor = np.array(in_tensor).astype(np.uint8)
        # in_tensor = np.array(in_tensor).astype(np.int8)

        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], in_tensor)
        self.interpreter.invoke()

        out_data = []
        for od in self.interpreter.get_output_details():
            out_tensor = self.interpreter.get_tensor(od['index'])
            # print('Debug : out_tensor shape == ', out_tensor.shape)
            if self.is_quantity:
                scale1 = od['quantization'][0]
                zero_points1 = od['quantization'][1]
                out_tensor = ((out_tensor - zero_points1) * scale1)
            out_img = np.squeeze(out_tensor, axis=0)

            if len(out_img)>10:
                out_img = self.postprocess(out_img, mode=self.mode, out_size=[in_w, in_h])

            out_data.append(out_img)

        return out_data


# def convert_gainmap_to_1dlut_tv(max_rgb_mini,in_gainmap,is_half=0,params={}):

#     # in_sdr_uint8 = np.clip(in_sdr*255,0,255).astype(np.uint8)
#     # max_rgb = np.max(in_sdr_uint8,axis=2)
#     # in_sdr_mini_uint8 = np.clip(in_sdr_mini*255,0,255).astype(np.uint8)
#     # max_rgb_mini = np.max(in_sdr_mini_uint8,axis=2)
#     max_rgb_mini = np.clip(max_rgb_mini,0,255).astype(np.uint8)

#     in_gh,in_gw = in_gainmap.shape[:2]
#     cur_gw = 0
#     cur_sw = 0

#     # # # # # limit by max
#     # last_gm = 0
#     # gm_1dlut_max = 0
#     # gm_1dlut = np.zeros([256]).astype(np.float32)
#     # for cur_luma in range(256):
#     #     # print(cur_luma,len(max_rgb_mini[max_rgb_mini==cur_luma]))
#     #     if len(max_rgb_mini[max_rgb_mini==cur_luma])>0:
#     #         gm_1dlut[cur_luma] = np.mean(in_gainmap[max_rgb_mini==cur_luma])
#     #         last_gm = gm_1dlut[cur_luma]
#     #         gm_1dlut_max = max(gm_1dlut_max,last_gm)
#     #     elif last_gm==0:
#     #         gm_1dlut[cur_luma] = 1.
#     #     else:
#     #         gm_1dlut[cur_luma] = last_gm
#     #         last_gm = gm_1dlut[cur_luma]
#     #         gm_1dlut_max = max(gm_1dlut_max,last_gm)

#     # # # # interplate vail lut_bin
#     gm_vali_interps_left = []
#     gm_vali_interps_right = []
#     gm_vali_len = 0
#     last_gm = 0
#     gm_1dlut = np.zeros([256]).astype(np.float32)
#     for cur_luma in range(256):
#         # print(cur_luma,len(max_rgb_mini[max_rgb_mini==cur_luma]))
#         if len(max_rgb_mini[max_rgb_mini==cur_luma])>0:
#             gm_1dlut[cur_luma] = np.mean(in_gainmap[max_rgb_mini==cur_luma])

#             if gm_vali_len==0:
#                 gm_vali_len = 1
#             gm_vali_interps_left.append([gm_vali_len,gm_1dlut[cur_luma]])
#             if gm_vali_len>1:
#                 gm_vali_len = 1
#             last_gm = gm_1dlut[cur_luma]
#             # last_gm = max(gm_1dlut[cur_luma],last_gm)
#         else:
#             gm_vali_interps_left.append([gm_vali_len,last_gm])
#             gm_vali_len +=1
#     if gm_1dlut[-1]==0:
#         gm_1dlut[-1]=np.max(gm_1dlut)
#     gm_vali_interps_left = np.array(gm_vali_interps_left)

#     last_gm = 0
#     gm_vali_len = 0
#     for cur_luma in gm_1dlut[::-1]:
#         if cur_luma>0:
#             gm_vali_len = 0
#             gm_vali_interps_right.append([gm_vali_len,cur_luma])
#             last_gm = cur_luma
#         else:
#             gm_vali_len +=1
#             gm_vali_interps_right.append([gm_vali_len,last_gm])
#     gm_vali_interps_right = np.array(gm_vali_interps_right)[::-1,:]

#     gm_1dlut = (gm_vali_interps_right[:,0]*gm_vali_interps_left[:,1]+gm_vali_interps_left[:,0]*gm_vali_interps_right[:,1])/(gm_vali_interps_left[:,0]+gm_vali_interps_right[:,0])
#     # print('Debug ： ',[i for i in np.concatenate([np.expand_dims(gm_1dlut,axis=1),gm_vali_interps_left,gm_vali_interps_right],axis=1)])
#     gm_1dlut_max = np.max(gm_1dlut)

#     # gm_1dlut = np.floor(gm_1dlut*4).astype(np.float32)/4.
#     # gm_1dlut_max = np.max(gm_1dlut)

#     return gm_1dlut,gm_1dlut_max

# def gm1dlut_to_edrlut_tv(gm_1dlut,qmax,headroom=4.,gm_1dlut_max=1.):
#     lut2d_gm_to_edr_bit = 2**10
#     # lut2d_gm_to_edr = []
#     # for i in range(256):
#     #     for j in range(lut2d_gm_to_edr_bit*4):
#     #         lut2d_gm_to_edr.append(gainmap_to_sdr(i/255.,-j/lut2d_gm_to_edr_bit,in_img_max_val = 1.0,norm_hdr=1.) *1023)
#     # lut2d_gm_to_edr = np.reshape(lut2d_gm_to_edr,(256,lut2d_gm_to_edr_bit*4)).astype(np.uint16)

#     lut_max_val = 1023
#     # gm_1dlut_max = max(np.power(2,headroom_max)+1,0)
#     # gm_1dlut_max = max(np.power(2, np.max(gm_1dlut)*headroom),1.0)
#     # gm_1dlut_12bit = np.floor(gm_1dlut*headroom/headroom_max*(2**10-1)) # gm_1dlut数值为10或12bit

#     last_lut_bin = 0
#     edr_1dlut = np.zeros_like(gm_1dlut).astype(np.float32)
#     for i in range(len(gm_1dlut)):
#         # gm_1dlut_cur = ((gm_1dlut_max-gm_1dlut[i])*qmax*lut2d_gm_to_edr_bit*headroom)
#         # gm_1dlut_cur = np.floor((gm_1dlut_max-gm_1dlut[i])*qmax*lut2d_gm_to_edr_bit*headroom) # gm_1dlut_cur范围为[0,4],整数2bit+小数10bit
#         # edr_1dlut[i] = gainmap_to_sdr(i/255.,-gm_1dlut_cur/lut2d_gm_to_edr_bit,in_img_max_val = 1.0,norm_hdr=1.) *lut_max_val
#         # # edr_1dlut[i] = lut2d_gm_to_edr[i,int(gm_1dlut_cur)]

#         edr_1dlut[i] = max(np.floor(i/255. * gm_1dlut[i]/gm_1dlut_max *lut_max_val),last_lut_bin)

#         last_lut_bin = max(last_lut_bin,edr_1dlut[i])
#         # print(i,gm_1dlut_cur,edr_1dlut_tmp,edr_1dlut[i])
#     edr_gain = 180./255*lut_max_val/edr_1dlut[180]

#     # print('Debug gm1dlut_to_edrlut_int : edr_1dlut_min edr_1dlut_max gm_1dlut_max headroom_rate edr_gain == ',np.min(edr_1dlut),np.max(edr_1dlut),np.max(gm_1dlut),headroom/gm_1dlut_max,edr_gain)
#     # edr_1dlut = np.clip(edr_1dlut,0,lut_max_val)
#     edr_1dlut = np.clip(edr_1dlut,0,2**31-1)
#     # edr_1dlut = np.floor(edr_1dlut).astype(np.int32)

#     return edr_1dlut,edr_gain

# def gm_post_tv(max_rgb_mini_float,gm_1dlut,qmax,headroom=4.):
    # lut2d_gm_to_edr_bit = 2**10
    # # lut2d_gm_to_edr = []
    # # for i in range(256):
    # #     for j in range(lut2d_gm_to_edr_bit*4):
    # #         lut2d_gm_to_edr.append(gainmap_to_sdr(i/255.,-j/lut2d_gm_to_edr_bit,in_img_max_val = 1.0,norm_hdr=1.) *1023)
    # # lut2d_gm_to_edr = np.reshape(lut2d_gm_to_edr,(256,lut2d_gm_to_edr_bit*4)).astype(np.uint16)

    # lut_max_val = 1023
    # gm_1dlut_max = np.max(gm_1dlut)
    # # gm_1dlut_max = max(np.power(2,headroom_max)+1,0)
    # # gm_1dlut_max = max(np.power(2, np.max(gm_1dlut)*headroom),1.0)
    # # gm_1dlut_12bit = np.floor(gm_1dlut*headroom/headroom_max*(2**10-1)) # gm_1dlut数值为10或12bit

    # last_lut_bin = 0
    # gm_tv = np.zeros_like(gm_1dlut).astype(np.float32)
    # # for i in range(len(gm_1dlut)):
    #     # edr_1dlut[i] = lut2d_gm_to_edr[i,int(gm_1dlut_cur)]
    # gm_1dlut_cur = ((gm_1dlut_max-gm_1dlut)*qmax*lut2d_gm_to_edr_bit*headroom)
    # # gm_1dlut_cur = np.floor((gm_1dlut_max-gm_1dlut)*qmax*lut2d_gm_to_edr_bit*headroom) # gm_1dlut_cur范围为[0,4],整数2bit+小数10bit
    # gm_tv = gainmap_to_sdr(max_rgb_mini_float/255.,-gm_1dlut_cur/lut2d_gm_to_edr_bit,in_img_max_val = 1.0,norm_hdr=1.)

    # gm_tv = np.where(max_rgb_mini_float>0,gm_tv*255./max_rgb_mini_float,0)

    #     # print(i,gm_1dlut_cur,edr_1dlut_tmp,edr_1dlut[i])
    # # edr_gain = 180./255*lut_max_val/edr_1dlut[180]

    # # print('Debug gm1dlut_to_edrlut_int : edr_1dlut_min edr_1dlut_max gm_1dlut_max headroom_rate edr_gain == ',np.min(edr_1dlut),np.max(edr_1dlut),np.max(gm_1dlut),headroom/gm_1dlut_max,edr_gain)
    # # edr_1dlut = np.clip(edr_1dlut,0,lut_max_val)
    # # edr_1dlut = np.clip(edr_1dlut,0,2**31-1)
    # # edr_1dlut = np.floor(edr_1dlut).astype(np.int32)

    # return gm_tv


def process_img(in_img,in_img_mini,model_infer):
    result={}

    # t_s = time.time()
    out_img = model_infer(in_img_mini)
    # t += time.time()-t_s

    gainmap_raw = out_img[0]
    Q_max = np.squeeze(out_img[1])
    # max_rgb = out_img[2]
    # max_rgb = np.max(in_img_mini,axis=-1)
    out_rgb = out_img[3]

    # gainmap_raw = out_img[0]
    # Q_max = 0
    # max_rgb = out_img[1]
    # out_rgb = out_img[2]
    # print('Debug : Q_max == ',Q_max)
    if Q_max<=1.:
        Q_max = Q_max*255

    ### gainmap for PC
    enhance_parmas={
        'mode': 'gainmap',
        'is_half': 0,
        # 'peak_light': 3000,
        # 'light_ranges': [1000,400],
        'peak': [8]*2, #8
        'gm_dark_range': [20,90,140,220],
        # 'gm_ymean_range': [45,60],
        'gm_dark_gain': [0]*4,
        # 'upscale_method': 'bilinear', # bilinear
        # '1dlut_method': 'mean_lut',
        # 'headroom_max': 3.6, # IMG_955_gainmap_headroom3.52
        'white_lum': 100,  # 203 100
        # 'dark_limit': 0.4, # gt_hdr10/sdr----0.2; hdr----0.4
        'dark_limit': -2,
        'show_debug': 1,
        'tonemap': 'hdr10_video', # dynLinear dynHable hdr10_image hdr10_video none
    }

    max_rgb = np.max(in_img,axis=2).astype(np.uint8)
    max_rgb_mini = np.max(in_img_mini,axis=2).astype(np.uint8)
    result = postprocess_gainmap(in_img,in_img_mini,gainmap_raw,Q_max,max_rgb,max_rgb_mini,enhance_parmas=enhance_parmas)

    #### gainmap1dlut for PC
    # enhance_parmas={
        # 'mode': 'gainmap',
    #     'is_half': 0,
    #     # 'peak_light': 3000,
    #     # 'light_ranges': [1000,400],
    #     'peak': [3]*2,
    #     'gm_dark_range': [20,90,140,220],
    #     # 'gm_ymean_range': [45,60],
    #     'gm_dark_gain': [0]*4,
    #     # 'upscale_method': 'bilinear',
    #     '1dlut_method': 'mean_lut',
    #     # 'headroom_max': 3.6, # IMG_955_gainmap_headroom3.52
    #     'white_lum': 203,  # 203 400
            # 'dark_limit': 0.2,
    #     'show_debug': 1,
    #    'tonemap': 'dynLinear',
    # }
    # max_rgb = np.max(in_img,axis=2).astype(np.uint8)
    # max_rgb_mini = np.max(in_img_mini,axis=2).astype(np.uint8)
    # result = postprocess_gainmap(in_img,in_img_mini,gainmap_raw,Q_max,max_rgb,max_rgb_mini,headroom = 4,white_lum=203,enhance_parmas=enhance_parmas)


    # ### gainmap1dlut for TV
    # enhance_parmas={
    #     'mode': 'gainmap1dlut',
    #     'is_half': 0,
    #     # 'peak_light': 3000,
    #     # 'light_ranges': [1000,400],
    #     'peak': [8]*2,
    #     'gm_dark_range': [20,90,140,220],
    #     # 'gm_ymean_range': [45,60],
    #     'gm_dark_gain': [0]*4,
    #     # 'upscale_method': 'bilinear',
    #     '1dlut_method': 'mean_lut',
    #     # 'headroom_max': 3.6, # IMG_955_gainmap_headroom3.52
    #     'white_lum': 100,  # 203 100
    #     'dark_limit': 0.2,
    #     'show_debug': 1,
    #     'tonemap': 'dynHable', # none dynHable
    #     'color_method': 'maxrgb', # 'yuv' 'hsv' 'maxrgb' 'yuv_comp'
    # }

    # if enhance_parmas['color_method'] in ['yuv','yuv_comp']:
    #     max_rgb = rgb_to_Y(in_img.astype(np.float32)).astype(np.uint8)
    #     max_rgb_mini = rgb_to_Y(in_img_mini.astype(np.float32)).astype(np.uint8)
    # else:
    #     max_rgb = np.max(in_img,axis=2).astype(np.uint8)
    #     max_rgb_mini = np.max(in_img_mini,axis=2).astype(np.uint8)
    # result = postprocess_gainmap1dlut(in_img,in_img_mini,max_rgb,max_rgb_mini,gainmap_raw,Q_max,enhance_parmas=enhance_parmas)

    result.update(enhance_parmas)
    return result

def process_imgs(in_root, model_path, out_root, use_gpu=0):
    gainmap_peak = np.power(2, 3)
    # if use_gpu_id < 0:
    #     device = torch.device('cpu')
    # else:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_paths = glob.glob(os.path.join(in_root,'*.*'))
    gainmap_pname_lists = [
        # # 'pattern_all255',
        # # 'grad_white_rect_8',
        # 'IMG_915',
        # 'IMG_927',
        # # 'IMG_900',
        # 'gaofancha_0001',
        # # 'gaofancha_0002',
        # # 'gaofancha_0003',
        # 'gaofancha_0004',
        # # 'chengshifengguangDR_t0000',
        # # 'chengshifengguangDR_t0015',
        # # 'IMG_955',
        # # 'heishenhuaSDR_00000006',
        # # 'IMG_3853_2160p',
        'Bulgaria4KHDR_00000009',
        'Bulgaria4KHDR_00000012',
        'Bulgaria4KHDR_00000017',
        # 'Bulgaria4KHDR_00000032',
        'Bulgaria4KHDR_00000042',
        'Bulgaria4KHDR_00000049',
        'Bulgaria4KHDR_00000060',
        # 'Bulgaria4KHDR_00000077',
        # 'Bulgaria4KHDR_00000078',
        # 'Bulgaria4KHDR_00000085',
        # 'Bulgaria4KHDR_00000109',
        # 'Bulgaria4KHDR_00000208',
        # 'Bulgaria4KHDR_00000145',
        'Morocco4KHDR_00000021',
        'Morocco4KHDR_00000030',
        # 'Morocco4KHDR_00000137',
        # 'Morocco4KHDR_00000177',
        # 'OPPO_THE_ORIGIN_4KUHD_60p_HDR_00000071',
        # 'Uzbekistan4KHDR_00000004',
        'Uzbekistan4KHDR_00000040',
        'Uzbekistan4KHDR_00000091',
        # 'Uzbekistan4KHDR_00000153',
        # 'Game_of_Thrones_S01E01_Winter_Is_Coming_2011_UHD_Bluray_2160p_10bit_DoVi_2Audio_TrueHDAtmos_7_1_x265beAst_p000_00000016',
        'Game_of_Thrones_S01E01_p000_00000030',
        'huangyelieren_2015_4K_p001_00000005',
        'HarryPotterandtheDeathlyHallowsPart2_2011_2160p_p001_00000240',
        'HarryPotterandtheDeathlyHallowsPart2_2011_2160p_p002_00000160',
        ]
    # in_paths=[i for i in in_paths if os.path.splitext(os.path.split(i)[-1])[0] in gainmap_pname_lists]
    # in_paths = in_paths[:3]
    print('Debug : found %d img in in_root' % len(in_paths))

    if not os.path.exists(out_root):
        os.makedirs(out_root)
    if not os.path.exists(out_root+'_hdr10'):
        os.makedirs(out_root+'_hdr10')

    model_infer = GMNet_infers(model_path)

    t_count = 0
    t = 0
    for iid, in_path in enumerate(in_paths):
        result={}
        in_path_name = os.path.splitext(os.path.split(in_path)[-1])[0]
        print('Debug : process == ',iid,in_path_name)

        in_img = cv2.imread(in_path)
        # in_img = cv2.resize(in_img,[in_img.shape[1]//2,in_img.shape[0]//2],interpolation=cv2.INTER_LINEAR)
        # in_img = read_bin_as_np(in_path)[:,:,::-1]
        if in_img is None:
            continue
        in_img = in_img[:,:,::-1]

        in_img_mini = cv2.resize(in_img,[384,216],interpolation=cv2.INTER_AREA)
        # if not t_count%(n_count_dict[in_path_name]//300+1)==0:
        #     continue


        result = process_img(in_img,in_img_mini,model_infer)

        # result['qmax'] = result['qmax']*np.max(result['gainmap'])/255.
        # result['gainmap'] = result['gainmap']/ np.max(result['gainmap'])*255.

        # print('Debug : ',out_img[0].shape,np.max(out_img[0]),out_img[1],out_img[2].shape,np.max(out_img[2]))
        out_path = os.path.join(out_root, in_path_name+'_gainmap_Qmax%f'%result['qmax']+'.png')
        cv2.imwrite(out_path, result['gainmap'])
        out_path = os.path.join(out_root, in_path_name+'_maxrgb.png')
        cv2.imwrite(out_path, result['maxrgb'])
        out_path = os.path.join(out_root, in_path_name+'_input.png')
        cv2.imwrite(out_path, in_img[:,:,::-1])
        out_path = os.path.join(out_root, in_path_name+'_gainmapout.png')
        cv2.imwrite(out_path, np.clip(result['gainmap_out'].astype(np.float32)*255./gainmap_peak,0,255).astype(np.uint8))

        if result['mode'] == 'gainmap':
            out_path = os.path.join(out_root, in_path_name+'_gainmapupscale.png')
            cv2.imwrite(out_path, np.clip(result['gainmap_upscale']*255,0,255).astype(np.uint8))

            if result['tonemap'] in ['hdr10_image']:
                out_path = os.path.join(out_root,in_path_name+'_BT2100PQ.png')
                cv2.imwrite(out_path, np.clip(result['bt2100_pq'][:,:,::-1],0,65535).astype(np.uint16))


                out_avif_path = os.path.join(out_root,in_path_name+'_avif.avif')
                write_heif(
                    out_avif_path,
                    result['bt2100_pq'],
                    format="AVIF",
                    quality=-1,
                    bitdepth=10,
                    yuv=None,
                )

                hdr10_video_path = os.path.join(out_root+'_hdr10',os.path.splitext(os.path.split(out_avif_path)[-1])[0]+'_hdr10.mp4')
                ffmpeg_image2hdr(out_avif_path, hdr10_video_path,t_len=3)
            else:
                out_path = os.path.join(out_root, in_path_name+'_apply'+'.png')
                cv2.imwrite(out_path, result['bt2100_pq'][:,:,::-1])
                hdr10_video_path = os.path.join(out_root+'_hdr10',os.path.splitext(os.path.split(out_path)[-1])[0]+'.mp4')
                ffmpeg_image2sdr(out_path, hdr10_video_path,t_len=3)

        elif result['mode'] == 'gainmap1dlut':
            out_path = os.path.join(out_root, in_path_name+'_applylut'+'.png')
            cv2.imwrite(out_path, result['gainmap_gmedr'][:,:,::-1])
            hdr10_video_path = os.path.join(out_root+'_hdr10',os.path.splitext(os.path.split(out_path)[-1])[0]+'.mp4')
            ffmpeg_image2sdr(out_path, hdr10_video_path,t_len=3)

            out_path = os.path.join(out_root,in_path_name+'_gm1dlut.png')
            draw_plt(np.arange(len(result['gm_1dlut'])),[result['gm_1dlut_min'],result['gm_1dlut_mean'],result['gm_1dlut_max'],result['gm_1dlut']],labels=['gm_1dlut_min','gm_1dlut_mean','gm_1dlut_max','gm_1dlut'],out_plt_path=out_path)
            print('Debug : gmedr_1dlut min max ==== ',result['headroom'],np.min(result['gmedr_1dlut']),np.max(result['gmedr_1dlut']))
            linear_1dlut = np.round(np.arange(256).astype(np.float32)/255.*1023)
            out_path = os.path.join(out_root,in_path_name+'_edrlut.png')
            draw_plt(np.arange(len(result['gmedr_1dlut'])),[result['gmedr_1dlut'],linear_1dlut],labels=['gmedr_1dlut','linear_1dlut'],out_plt_path=out_path)
            # out_path = os.path.join(out_root,in_path_name+'_gmedr1dlut.bin')
            # save_lut_bin(result['gmedr_1dlut'],out_path)


        t_count += 1

    # print('Debug : time avg cost == ', t/t_count)
    return 1


def process_vids(in_root, model_path, out_root, use_gpu=0, vid_ids =[]):
    gainmap_peak = np.power(2, 3)

    in_paths = glob.glob(os.path.join(in_root,'*.mp4'))
    gainmap_pname_lists = [
        'Bulgaria4KHDR',
        'Peru4KHDR',
        'Morocco4KHDR',
        'Uzbekistan4KHDR',
        'OPPO_THE_ORIGIN_4KUHD_60p_HDR',
        'Gravity_2013_2160p_BluRay_x265_10bit_HDR_TrueHD_7_1_Atmos_p000',
        'HarryPotterandtheDeathlyHallowsPart2_2011_2160p_p001',
        'HarryPotterandtheDeathlyHallowsPart2_2011_2160p_p002',
        ]
    if len(vid_ids) > 0:
        gainmap_pname_lists_select = [gainmap_pname_lists[i] for i in vid_ids]
    in_paths=[i for i in in_paths if os.path.splitext(os.path.split(i)[-1])[0] in gainmap_pname_lists_select]
    # in_paths = in_paths[:3]
    print('Debug : found %d img in in_root' % len(in_paths))

    model_infer = GMNet_infers(model_path, use_gpu=use_gpu)

    tonemap='hdr10_video'
    for iid, in_path in enumerate(in_paths):
        in_path_name = os.path.splitext(os.path.split(in_path)[-1])[0]
        if not os.path.exists(out_root):
            os.makedirs(out_root)
        if not os.path.exists(os.path.join(out_root+'_avif',in_path_name)):
            os.makedirs(os.path.join(out_root+'_avif',in_path_name))
        if not os.path.exists(out_root+'_hdr10'):
            os.makedirs(out_root+'_hdr10')

        print('Debug : process == ',iid,in_path_name)

        # hdr_reader = HDRVideoReader(in_path)
        sdr_reader = VideoReader(in_path)
        out_video_path = os.path.join(out_root, in_path_name+'_gainmap.mp4')
        writer = VideoWriter(out_video_path, height=2160, width=3840, fps=60)
        out_video_path = os.path.join(out_root, in_path_name+'_gainmapTM.mp4')
        tm_writer = VideoWriter(out_video_path, height=2160, width=3840, fps=60)

        t_count = 0
        hdr10_list=[]
        for idx in range(sdr_reader.get_nb_frames()):
            # hdr_img = hdr_reader.get_frame() #rgb
            # in_img = hdr_img/(2**16-1) #rgb
            sdr_img = sdr_reader.get_frame() #bgr
            if sdr_img is None:
                continue
            in_img = sdr_img[:,:,::-1]/(2**8-1) #rgb

            result={}

            # in_img = cv2.imread(in_path)[:,:,::-1]
            # in_img = cv2.resize(in_img,[in_img.shape[1]//2,in_img.shape[0]//2],interpolation=cv2.INTER_LINEAR)
            # in_img = read_bin_as_np(in_path)[:,:,::-1]
            # if in_img is None:
            #     continue

            # if idx<3600*15:
            #     # t_count += 1
            #     continue
            if idx>3600*3:
                break
            # if idx%3600>120:
            #     continue
            # if not t_count%(n_count_dict[in_path_name]//300+1)==0:
            #     continue

            in_img = in_img.astype(np.float32)*255
            in_img_mini = cv2.resize(in_img,[384,216],interpolation=cv2.INTER_AREA) # INTER_AREA for no flash


            result = process_img(in_img,in_img_mini,model_infer)

            # result['qmax'] = result['qmax']*np.max(result['gainmap'])/255.
            # result['gainmap'] = result['gainmap']/ np.max(result['gainmap'])*255.

            if t_count==1:
                print('Debug : result == ', result['tonemap'], result.keys())
                out_path = os.path.join(out_root, in_path_name+'_gainmap_Qmax%f'%result['qmax']+'.png')
                cv2.imwrite(out_path, result['gainmap'])
                out_path = os.path.join(out_root, in_path_name+'_maxrgb.png')
                cv2.imwrite(out_path, result['maxrgb'])
                out_path = os.path.join(out_root, in_path_name+'_input.png')
                cv2.imwrite(out_path, in_img[:,:,::-1])
                out_path = os.path.join(out_root, in_path_name+'_gainmapout.png')
                cv2.imwrite(out_path, np.clip(result['gainmap_out'].astype(np.float32)*255/gainmap_peak,0,255).astype(np.uint8))

                if result['mode'] == 'gainmap':
                    out_path = os.path.join(out_root, in_path_name+'_gainmapupscale.png')
                    cv2.imwrite(out_path, np.clip(result['gainmap_upscale']*255,0,255).astype(np.uint8))

            if result['tonemap'] in ['hdr10_video']:
                out_avif_path = os.path.join(out_root+'_avif',in_path_name,'%08d.avif'%t_count)
                write_heif(
                    out_avif_path,
                    result['bt2100_pq'],
                    format="AVIF",
                    quality=-1,
                    bitdepth=10,
                    yuv=None,
                )
                hdr10_list.append('file \'./%s/%s'%(in_path_name,'%08d.avif\''%t_count))
            elif result['tonemap'] in ['dynHable','dynLinear']:
                # out_path = os.path.join(out_root, in_path_name+'_tonemap.png')
                # cv2.imwrite(out_path, np.clip(result['bt2100_pq'].astype(np.float32),0,255).astype(np.uint8))
                print('Debug : bt2100_pq min max mean == ',result['bt2100_pq'].shape,np.min(result['bt2100_pq']),np.max(result['bt2100_pq']),np.mean(result['bt2100_pq']))
                tm_writer.write_frame(np.clip(result['bt2100_pq'][:,:,::-1],0,255).astype(np.uint8))
            else:
                pass

            # print('Debug : gainmap_out min max mean == ',result['gainmap_out'].shape,np.min(result['gainmap_out']),np.max(result['gainmap_out']),np.mean(result['gainmap_out']))
            writer.write_frame(np.clip(np.repeat(result['gainmap_out'].astype(np.float32),3,axis=-1)/gainmap_peak*255,0,255).astype(np.uint8))
            if t_count%1e2==0:
                print('Debug : process %d frames '%t_count, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
            t_count += 1


        print('Debug : tonemap == ',tonemap)
        if tonemap in ['hdr10_video']:
            hdr10_txt_path = os.path.join(out_root+'_avif',in_path_name+'.txt')
            np.savetxt(hdr10_txt_path,hdr10_list,fmt='%s')
            hdr10_video_path = os.path.join(out_root+'_hdr10',in_path_name+'.mp4')
            ffmpeg_frame2hdr(out_root+'_avif', hdr10_video_path, in_path_name+'.txt')

            # os.removedirs(os.path.join(out_root+'_avif',in_path_name))
        elif tonemap in ['dynHable','dynLinear']:
            tm_writer.close()

        # hdr_reader.close()
        sdr_reader.close()
        writer.close()

    return 1


def save_avif_to_hdr10(avif_root, out_hdr10_root,p_name='_p01'):

    in_paths = glob.glob(os.path.join(avif_root,'*'))
    gainmap_pname_lists = [
        'Bulgaria4KHDR_00000049',
        ]
    # in_paths=[i for i in in_paths if os.path.splitext(os.path.split(i)[-1])[0] in gainmap_pname_lists]
    # in_paths = in_paths[:3]
    print('Debug : found %d files in avif_root' % len(in_paths))

    if not os.path.exists(out_hdr10_root):
        os.makedirs(out_hdr10_root)

    for iid, avif_file_path in enumerate(in_paths):
        in_path_name = os.path.splitext(os.path.split(avif_file_path)[-1])[0]
        print('Debug : process == ',iid,avif_file_path)

        if not os.path.isdir(avif_file_path):
            continue

        avif_paths = sorted(glob.glob(os.path.join(avif_file_path,'*.avif')))
        print('Debug : found %d avif files in avif_paths' % len(avif_paths))

        if len(avif_paths)==0:
            print('Debug : found avif files == ',len(avif_paths),' in ',avif_file_path)
            continue

        hdr10_list = []
        for avif_path in avif_paths:
            hdr10_list.append('file \'./%s/%s'%(in_path_name,'%s\''%(os.path.split(avif_path)[-1])))


        hdr10_txt_path = os.path.join(avif_root,in_path_name+p_name+'.txt')
        np.savetxt(hdr10_txt_path,hdr10_list,fmt='%s')
        hdr10_video_path = os.path.join(out_hdr10_root,in_path_name+p_name+'_hdr10.mp4')
        ffmpeg_frame2hdr(avif_root, hdr10_video_path, in_path_name+p_name+'.txt')
        # ffmpeg_frame2hdrNv(avif_root, hdr10_video_path, in_path_name+p_name+'.txt')

    return 1


# def generate_bins():

#     # # # 全局lut表
#     # lut_linear256 = np.arange(0,256,1).astype(np.float32)/255.
#     # # bt709_eotf_lut = np.floor(np.where((0.0031308* 12.92) >= lut_linear256, lut_linear256 / 12.92, np.power((lut_linear256 + 0.055) / 1.055, 2.4))*65535)
#     # # lut_linear4095 = np.arange(0,4096,1).astype(np.float32)/4095.
#     # # gainmap_power2_lut = np.floor(np.power(2,lut_linear4095)*65535)
#     # L_gainmap=(lut_linear256)**(2.2)*255
#     # out_path = os.path.join(out_root,'lut1d_Lgainmap.txt')
#     # np.savetxt(out_path, L_gainmap,fmt='%d')


#     # # # # # gm_alpha_lut 表

#     # gm_alpha_pts =np.array([
#     #     [0,0],
#     #     [100,30],
#     #     [170,140],
#     #     [240,250],
#     #     [255,255],
#     # ]) # [y,gm_lut*4]
#     # gm_alpha_lut = np.round(np.interp(np.arange(256).astype(np.int32), gm_alpha_pts[:,0], gm_alpha_pts[:,1]*255)).astype(np.int32)

#     # out_path = os.path.join(out_root,'gm_alpha_lut.txt')
#     # np.savetxt(out_path, gm_alpha_lut,fmt='%d')
#     # out_path = out_path.replace('.txt','.bin')
#     # gm_alpha_lut.tofile(out_path)


#     # # # # headroom lut 表
#     # headroom = 6
#     # headroom_lut = np.zeros([256],dtype=np.int32)
#     # for hid in range(len(headroom_lut)):
#     #     headroom_lut[hid]=((headroom-2)*hid+2*255)

#     # head_pts =np.array([
#     #     [0,8],
#     #     [20,8],
#     #     [80,12],
#     #     [140,12],
#     #     [180,12],
#     #     [255,12],
#     # ]) # [y,headroom*4]
#     # headroom_lut = np.round(np.interp(np.arange(256).astype(np.int32), head_pts[:,0], head_pts[:,1]*255)).astype(np.int32)

#     # out_path = os.path.join(out_root,'headroom_lut.txt')
#     # np.savetxt(out_path, headroom_lut,fmt='%d')
#     # out_path = out_path.replace('.txt','.bin')
#     # headroom_lut.tofile(out_path)

#     # # # # headroom lut2 表
#     # peak=[5,4.5,3.5,3.5] #[5,3.5,3.5,3.5]
#     # gm_dark_range = [20,100,140,220]#[20,90,140,220]
#     # # gm_dark_gain = np.array([2,1.8,1,0]) #[2,1,1,0]
#     # gm_dark_gain = np.array([0.5]*4) #[2,1,1,0]

#     # head_k_pts =np.array([
#     #     [0,peak[0]],
#     #     [gm_dark_range[0],peak[0]],
#     #     [gm_dark_range[1],peak[1]],
#     #     [gm_dark_range[2],peak[2]],
#     #     [gm_dark_range[3],peak[3]],
#     #     [255,peak[-1]],
#     # ]) # [y,headroom]
#     # # headroom_lut_k = np.interp(np.arange(256).astype(np.int32), head_k_pts[:,0], head_k_pts[:,1]*255)
#     # headroom_lut_k = CubicSpline(head_k_pts[:,0], head_k_pts[:,1]*255)(np.arange(256).astype(np.int32))
#     # headroom_lut_k_new = (headroom_lut_k-np.min(headroom_lut_k))/(np.max(headroom_lut_k)-np.min(headroom_lut_k))*(np.max(head_k_pts[:,1]*255)-np.min(head_k_pts[:,1]*255))+np.min(head_k_pts[:,1]*255)
#     # headroom_lut_k= np.round(headroom_lut_k_new).astype(np.int32)

#     # head_b_pts =np.array([
#     #     [0,gm_dark_gain[0]],
#     #     [gm_dark_range[0],gm_dark_gain[0]],
#     #     [gm_dark_range[1],gm_dark_gain[1]],
#     #     [gm_dark_range[2],gm_dark_gain[2]],
#     #     [gm_dark_range[3],gm_dark_gain[3]],
#     #     [255,gm_dark_gain[-1]],
#     # ]) # [y,headroom]
#     # headroom_lut_b = np.interp(np.arange(256).astype(np.int32), head_b_pts[:,0], head_b_pts[:,1]*255)
#     # # headroom_lut_b = CubicSpline(head_b_pts[:,0], head_b_pts[:,1]*255)(np.arange(256).astype(np.int32))
#     # # headroom_lut_b_new = (headroom_lut_b-np.min(headroom_lut_b))/(np.max(headroom_lut_b)-np.min(headroom_lut_b))*(np.max(head_b_pts[:,1]*255)-np.min(head_b_pts[:,1]*255))+np.min(head_b_pts[:,1]*255)
#     # headroom_lut_b= np.round(headroom_lut_b).astype(np.int32)

#     # # out_path = os.path.join(out_root,'headroom_lut_k_D20GP5_D100P4p5_D140P3p5_D220P3p5_cubic.txt')
#     # # np.savetxt(out_path, headroom_lut_k,fmt='%d')
#     # # out_path = out_path.replace('.txt','.bin')
#     # # headroom_lut_k.tofile(out_path)

#     # out_path = os.path.join(out_root,'headroom_lut_b_G0p5.txt')
#     # np.savetxt(out_path, headroom_lut_b,fmt='%d')
#     # out_path = out_path.replace('.txt','.bin')
#     # headroom_lut_b.tofile(out_path)


#     # # # # # 全局gainmap_to_sdr的两维表
#     # headroom_cur = 16
#     # headroom = 16
#     # lut2d_gm_to_edr_bit = 2**10
#     # lut2d_gm_to_edr = []
#     # for i in range(256):
#     #     for j in range(lut2d_gm_to_edr_bit*headroom):
#     #         edr_1dlut_bin = gainmap_to_sdr(i/255.,-j*headroom_cur/(headroom*lut2d_gm_to_edr_bit-1),in_img_max_val = 1.0,norm_hdr=1.)*1023*64

#     #         # edr_1dlut_bin = ((edr_1dlut_bin/1023.)**2.2)**(1./2.6)
#     #         lut2d_gm_to_edr.append(edr_1dlut_bin)
#     # lut2d_gm_to_edr = np.reshape(lut2d_gm_to_edr,(256,lut2d_gm_to_edr_bit*headroom)).astype(np.uint16)
#     # out_path = os.path.join(out_root,'lut2d_gm_to_edr_h%02d_linear_16bit.txt'%(headroom_cur))
#     # np.savetxt(out_path, lut2d_gm_to_edr,fmt='%d')
#     # out_path = os.path.join(out_root,'lut2d_gm_to_edr_h%02d_linear_16bit.bin'%(headroom_cur))
#     # lut2d_gm_to_edr.tofile(out_path)


#     # # # # # lut gamma
#     # last_lut_gamma = 2.6
#     # last_lut = []
#     # for i in range(1024):
#     #     last_lut.append(np.clip(np.round((float(i)/1023.)**(last_lut_gamma/2.2)*1023),0,1023))
#     # last_lut = np.array(last_lut).astype(np.int32)
#     # out_path = os.path.join(out_root,'lut_last_gamma%02d.txt'%(last_lut_gamma*10))
#     # np.savetxt(out_path, last_lut,fmt='%d')
#     # out_path = os.path.join(out_root,'lut_last_gamma%02d.bin'%(last_lut_gamma*10))
#     # last_lut.tofile(out_path)


#     # # bin转image
#     # bin_path = r''
#     # out_path = os.path.splitext(bin_path)[0]+'_bin.png'
#     # out_np = read_bin_as_np(bin_path,[216,384,3],in_type=np.float32)[:,:,::-1]
#     # cv2.imwrite(out_path, np.array(out_np).astype(np.uint8))


#     # # # # # 生成特殊edrlut
#     # gm_edrlut = np.round(np.arange(256).astype(np.float32)/255.*1023)
#     # # gm_edrlut = gm_edrlut-160
#     # gm_edrlut = gm_edrlut*0.87

#     # gm_edrlut[-1] = 1023
#     # gm_edrlut = np.clip(gm_edrlut,0,1023).astype(np.int32)
#     # out_path = os.path.join(out_root,'test_gmedr1dlut.bin')
#     # save_lut_bin(gm_edrlut,out_path)
#     # np.savetxt(out_path.replace('.bin','.txt'), gm_edrlut,fmt='%d')


#     return 1



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_root", type=str, default='/media/huaxin/second_fix/yexianyi/gainmap/data/20260414_jiali_film/SDR_ffmpeg_hable/', help="")
    parser.add_argument("--out_root", type=str, default='./990000_G_fp32_film/test_img_tflite_out', help="")
    parser.add_argument("--model_path", type=str, default='/media/huaxin/tcl3/xulu/gainmap/proj/GMNet4/metric/ckpt/990000_G_fp32_film.tflite', help="")
    parser.add_argument("--use_gpu", type=int, default=0, help="")
    parser.add_argument("--vid_ids", nargs='+', type=int, default=[], help="")
    args = parser.parse_args()
    return args


def main():
    # # in_root = r"F:\localedr\data\raws\raws_mini"  # yxy_testdata
    # # in_root = r"F:\localedr\data\raws\dark_debug"  # yxy_testdata
    # in_root = r"F:\localedr\data\raws\film_jiali_jianying_test"  # yxy_testdata
    # # out_root = r"F:\localedr\demo\20260326_log_tmp"
    # # model_path = r"F:\localedr\proj\GMNet\checkpoints\train_syn_20251212_GMNet_m013_Cos_img256\convert_result_210000_G_v2HistL1Bias_imagedepthbig_branch4GainmapQmaxYRgb\210000_G_uint8.tflite"
    # # out_root = r"F:\localedr\demo\20260407_GMNetm013_branch4GainmapQmaxYRgb_uint8_gmypost9"
    # model_path = r"F:\localedr\proj\GMNet\checkpoints\20260420_GMNet_m014_4_s20_mixdata_jialifilm2_W100_sonyYingdi_W100\4C\610000_G_uint8.tflite"
    # out_root = r"F:\localedr\demo\20260515_GMm0144s20mixdata_jialifilm2_W100_610000_uint8_gmypost11_yuv_tmp"
    # # model_path = r"F:\localedr\proj\GMNet\checkpoints\20260316_GMNet_m014_4_s20\m014_4_s20_4c_y_w100\430000_G_uint8.tflite"
    # # out_root = r"F:\localedr\demo\20260330_GMm0144s20_430000_uint8_gmypost10_hsv"
    # # model_path = r"F:\localedr\proj\GMNet\checkpoints\train_syn_20260226_GMm013Linear4_Cos_img256\convert_result_460000_G_v2HistL1Bias_imagedepthbig_branch3GainmapMaxrgbRgb_tmp\460000_G_uint8.tflite"
    # use_gpu_id = 0

    args = parse_args()
    print('Debug : args == ', args)
    in_root = args.in_root
    out_root = args.out_root
    model_path = args.model_path
    use_gpu = args.use_gpu
    vid_ids = args.vid_ids

    if not os.path.exists(out_root):
        os.makedirs(out_root)

    #process_imgs(in_root, model_path, out_root, use_gpu=use_gpu)

    process_vids(in_root, model_path, out_root, use_gpu, vid_ids)


    # avif_root= '/media/huaxin/second_fix/yexianyi/gainmap/demo/20260708_s010_SDR_ffmpeg_hable_GMm0144s20s4_JialiFilmHable_peakLimit8_430000_gainmap_hdrvideo_upscaleSmooth_avif'
    # out_hdr10_root = avif_root.replace('avif', 'hdr10')
    # save_avif_to_hdr10(avif_root, out_hdr10_root,p_name='_p01')

    return 1


if __name__ == '__main__':
    main()
    print('over')
