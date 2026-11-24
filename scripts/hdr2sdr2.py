import os
import glob
# import numpy as np
# import cv2
# import time
# import struct
# from PIL import Image, ImageCms
# import shutil
import subprocess
import sys
# import json
# import re
import argparse



# FFMPEG_PATH = r'D:\ysoft\dev2\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe'
# FFMPEG_PATH = 'ffmpeg'  # 不能用裸 'ffmpeg'：conda 环境里的 4.4.2 精简版没有 zscale，退化会静默失败（输出 0 字节）
FFMPEG_PATH = '/usr/bin/ffmpeg'


def ffmpeg_segment_vid(input_path='../../vid2frameDataset/gif/vid.mp4 ', output_path='../../vid2frameDataset/gif/gif_%03d.mp4', seg_num=10):

    command = "%s -y -i %s -f segment -segment_time %d -c copy -reset_timestamps 1  %s " % \
        (FFMPEG_PATH, input_path, seg_num, output_path)

    print(command)
    subprocess.call(command, shell=True)
    return 0


def convert_standard_hdr10(input_root,out_path,max_nits=1000,time_len=5):
    # scale = 10000./max_nits
    # command = '%s -y -i %s -ss 15 -t 5 \
    #         -vf "zscale=transfer=linear, \
    #             format=gbrpf32le, \
    #             tonemap=linear:param=%f, \
    #             zscale=transfer=smpte2084:primaries=bt2020:range=full, \
    #             format=yuv420p10le" \
    #         -color_primaries bt2020  \
    #         -color_trc smpte2084 \
    #         -colorspace bt2020nc \
    #         -c:v libx265 \
    #         -x265-params "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:info=1:repeat-headers=1:aud=1:master-display=\'G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(100000000,50)\':max-cll=10000,400" \
    #         -flags +global_header \
    #         -crf 18 \
    #         -preset fast \
    #         %s '% \
    #     (FFMPEG_PATH, input_root,scale, out_path) 
    
    if time_len>0:
        command = '%s -y -i "%s" -ss 15 -t %d \
                -c:v libx265 -crf 16 -preset fast -movflags +faststart \
                "%s" '% \
            (FFMPEG_PATH, input_root,time_len, out_path) 
        # -c copy \
        
    else:
        # HDR 参考版：不重编码（-c copy 纯拷贝，零损失），保持源帧率/像素/色彩元数据原样；
        # 与退化版同一输入解码顺序，天然逐帧对齐。
        command = '%s -y -i "%s" -c copy \
                "%s" '% \
            (FFMPEG_PATH, input_root, out_path)
    
    print(command)
    subprocess.run(command, shell=True)

    return 1

def convert_standard_sdr(input_root,out_path,max_nits=1000,time_len=-1):
    '''
        only used in ffmpeg-8.0.1-full_build!
        tonemap = hable  reinhard  mobius 
    '''

    if time_len>0:
        command = '%s -y -i "%s" -ss 15 -t %d \
                -vf "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p" \
                -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
                -c:v libx265 -crf 16 -preset fast -movflags +faststart \
                -x265-params "aq-mode=3:aq-strength=1.0:psy-rd=2.0:psy-rdoq=1.0" \
                "%s" '% \
            (FFMPEG_PATH, input_root,time_len, out_path)
    else:
        command = '%s -y -i "%s" \
                -vf "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0.0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p" \
                -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
                -c:v libx265 -crf 16 -preset fast -movflags +faststart \
                -x265-params "aq-mode=3:aq-strength=1.0:psy-rd=2.0:psy-rdoq=1.0" \
                "%s" '% \
            (FFMPEG_PATH, input_root, out_path)

    # if time_len>0:
    #     command = '%s -y -i "%s" -ss 15 -t %d  -r 60 \
    #             -vf "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p,deband=1thr=0.02:2thr=0.02:3thr=0.02:range=16:direction=2:blur=false:coupling=true" \
    #             -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
    #             -c:v libx265 -crf 16 -pix_fmt yuv420p10le -preset fast -movflags +faststart \
    #             -x265-params "aq-mode=3:aq-strength=1.0:psy-rd=2.0:psy-rdoq=1.0" \
    #             "%s" '% \
    #         (FFMPEG_PATH, input_root,time_len, out_path) 
    # else:
    #     command = '%s -y -i "%s"  -r 60 \
    #             -vf "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0.0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p,deband=1thr=0.02:2thr=0.02:3thr=0.02:range=16:direction=2:blur=false:coupling=true" \
    #             -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
    #             -c:v libx265 -crf 16 -pix_fmt yuv420p10le -preset fast -movflags +faststart \
    #             -x265-params "aq-mode=3:aq-strength=1.0:psy-rd=2.0:psy-rdoq=1.0" \
    #             "%s" '% \
    #         (FFMPEG_PATH, input_root, out_path)     
    #     # -x265-params "aq-mode=3:aq-strength=1.0:psy-rd=2.0:psy-rdoq=1.0" \
    #     # -vf "deband=1thr=0.02:2thr=0.02:3thr=0.02:range=16:direction=2:blur=false:coupling=true" \
    #     # -vf "deband=1thr=0.03:2thr=0.03:3thr=0.03:range=32:direction=0:blur=false:coupling=true" \
    #     # -vf "deband=1thr=0.04:2thr=0.04:3thr=0.04:range=48:direction=0:blur=false:coupling=true" \
    #     # -vf "dither=ordered" \
        
    
    print(command)
    subprocess.run(command, shell=True)

    return 1


def main(in_root,sdr_root,out_root,gennerate_parmas={}):

    in_paths=glob.glob(os.path.join(in_root,'*.*'))
    # gainmap_pname_lists = [
    #     # 'pattern_all255',
    #     # 'grad_white_rect_8',
    #     'IMG_915_1080p',
    #     'IMG_927',
    #     'IMG_900',
    #     # 'chengshifengguangDR_t0000',
    #     # 'chengshifengguangDR_t0015',
    #     # 'IMG_955',
    #     # 'heishenhuaSDR_00000006',
    #     'IMG_3853_2160p',
    #     ]

    # in_paths=[i for i in in_paths if os.path.splitext(os.path.split(i)[-1])[0] in gainmap_pname_lists]
    in_paths=[i for i in in_paths if not os.path.splitext(i)[-1] == '.json']
    print('Debug : found img ==== ',len(in_paths))
    # G_infer = GMNet_infer(opt_path='./codes/options/config/test_syn.yml',scale=0)


    log = []
    duration_sum = 0
    peak_max = 0
    gm_min = 1e4
    for iid,in_path in enumerate(in_paths[:]):
        print('Debug : process == ',in_path,' with mode ',gennerate_parmas['mode'])
        in_name = os.path.splitext(os.path.split(in_path)[-1])[0]
        sdr_path = os.path.join(sdr_root,os.path.splitext(os.path.split(in_path)[-1])[0]+'.png')
        # if gennerate_parmas['mode']=='calc_gainmap' and not os.path.exists(sdr_path):
        #     print('Alarm : not found file == ',sdr_path)
        #     continue

        if not os.path.exists(out_root):
            os.makedirs(out_root)
        
        if gennerate_parmas['mode']=='seg_vid':
            out_path = os.path.join(out_root,in_name+'_%03d'+os.path.splitext(in_path)[-1])
            ffmpeg_segment_vid(in_path,out_path,seg_num=1200)

        elif gennerate_parmas['mode']=='ffmpeg_hable':
            # out_path = os.path.join(out_root,in_name+'.json')
            # metadata = get_meta_data(in_path,out_path)
            # print('Debug : ',metadata,' in ',in_name)
            # if metadata['max_luma']<=0:
            #     continue

            if not os.path.exists(out_root+'_hdr'):
                os.makedirs(out_root+'_hdr')
                os.makedirs(out_root+'_sdr')
                print('Debug : make database',out_root+'_hdr')

            out_path = os.path.join(out_root+'_hdr',in_name+'.mp4')
            convert_standard_hdr10(in_path,out_path,1000,time_len=-1)
            out_path = os.path.join(out_root+'_sdr',in_name+'.mp4')
            convert_standard_sdr(in_path,out_path,time_len=-1)

        else:
            print('Error : mode is no found == ',gennerate_parmas['mode'])
    

    return 1


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--vid_hdr_root", type=str, default='./test_img/HDR', help="")
#     parser.add_argument("--vid_sdr_root", type=str, default='./test_img/SDR', help="")
#     parser.add_argument("--out_root", type=str, default='./results', help="")
#     args = parser.parse_args()
#     return args

if __name__ == "__main__":

    if len(sys.argv) >= 3:
        # Linux 命令行用法：python hdr2sdr2.py <in_root> <out_root>
        # mode 固定 ffmpeg_hable：每个视频生成 _hdr（重编 60fps）+ _sdr（hable 退化）
        in_root, out_root = sys.argv[1], sys.argv[2]
        gennerate_parmas = {
            'mode': 'ffmpeg_hable',
            'tonemap_method': 'hable',
            'peak_limit': 3,
            'peak_limit_dyn': 0.99,
        }
        main(in_root, in_root, out_root, gennerate_parmas)
        print('over')
        sys.exit(0)

    in_root=r'F:\localedr\data\20260414_jiali_film\dark_flim'

    seg_root = r'F:\localedr\data\20260414_jiali_film\dark_flim_seg'
    # gennerate_parmas={
    #     'mode': ['rename','seg_vid','ext_hdr','ext_sdr','hdr10_bt2020_pq_to_sdr','calc_gainmap','ffmpeg_hable'][1],
    #     'tonemap_method':['aces','reinhardCv','youtube','hable','jianying'][-1],
    #     'peak_limit': 3,
    #     'peak_limit_dyn': 0.99,
    # }
    # main(in_root,in_root,seg_root,gennerate_parmas)


    out_root = r'F:\localedr\data\20260414_jiali_film\dark_flim_hable'
    gennerate_parmas={
        'mode': ['rename','seg_vid','ext_hdr','ext_sdr','hdr10_bt2020_pq_to_sdr','calc_gainmap','ffmpeg_hable'][-1],
        'tonemap_method':['aces','reinhardCv','youtube','hable','jianying'][-1],
        'peak_limit': 3,
        'peak_limit_dyn': 0.99,
    }

    main(seg_root,seg_root,out_root,gennerate_parmas)

    print('over')


