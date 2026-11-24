#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从全分辨率亮度开始构建6尺度金字塔，只增强亮区，暗区不动。"""
import argparse
from pathlib import Path
import cv2
import numpy as np

def smoothstep(x):
    x=np.clip(x,0.,1.);return x*x*(3.-2.*x)

def gamma_lut17(gamma):
    x=np.linspace(0,1,17,dtype=np.float32)
    return 255.*np.power(x,gamma)

def apply_lut17(y,lut):
    a=np.asarray(lut,np.float32)
    if a.size!=17:raise ValueError('LUT必须有17个值')
    t=np.clip(y,0,1)*16.;i=np.minimum(np.floor(t).astype(np.int32),15);f=t-i
    return (a[i]*(1-f)+a[i+1]*f)/255.

def yellow_green_gate(bgr,hue_min=25.,hue_max=95.,sat_min=15.,soft=12.):
    """基于HSV色相+饱和度的黄绿色软门控，返回[0,1]权重图。非黄绿色→0。"""
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    h=hsv[:,:,0].astype(np.float32)  # 0-180
    s=hsv[:,:,1].astype(np.float32)  # 0-255
    hue_gate=smoothstep((h-hue_min)/soft)*(1.-smoothstep((h-hue_max)/soft))
    sat_gate=smoothstep((s-sat_min)/soft)
    return (hue_gate*sat_gate).astype(np.float32)

def process_luma(y,levels=7,threshold=100.,soft=20.,
                 ltmo_contrast=(1.8,1.6,1.4,1.25,1.1,1.0,1.0),
                 mode='gamma',gamma=.62,ratio=1.0,lut=None,start_scale=8,color_gate=None):
    """
    mode='gamma': mapped = y^gamma (gamma<1提亮)
    mode='ratio': 每层有效比例 = 1 + (ltmo_contrast[i]-1)*ratio；ratio为全局倍率(1=原始,0=不变,>1更强)
    start_scale: L0起始分辨率的分母(1=原图,2=1/2,4=1/4,8=1/8,...)
    """
    h,w=y.shape
    if levels!=7:raise ValueError('当前实现固定为7个尺度')
    if len(ltmo_contrast)!=7:raise ValueError('ltmo_contrast必须有7个值')
    # L0为1/start_scale分辨率，后续逐级1/2下采样。
    y_l0=cv2.resize(y,(w//start_scale,h//start_scale),interpolation=cv2.INTER_AREA).astype(np.float32)
    pyr=[y_l0]
    for _ in range(6):pyr.append(cv2.pyrDown(pyr[-1]))
    # 每一级只提取阈值以上亮区，再上采样回原分辨率。
    high=[]
    for p in pyr:
        m=smoothstep((p*255.-threshold)/max(soft,1e-6)).astype(np.float32)
        high.append(cv2.resize(m,(w,h),interpolation=cv2.INTER_LINEAR))
    # 相邻尺度亮区相减，避免同一亮区在7层重复增强。
    scale_masks=[np.maximum(high[i]-high[i+1],0.) for i in range(6)]
    scale_masks.append(high[6])
    # 对每个scale_mask做高斯平滑，消除尺度间边界
    for i in range(len(scale_masks)):
        scale_masks[i]=cv2.GaussianBlur(scale_masks[i],(0,0),3.0)
    # 原始分辨率再次门控，暗区严格不增强。
    pixel_mask=smoothstep((y*255.-threshold)/max(soft,1e-6)).astype(np.float32)
    # 黄绿色门控：非黄绿色像素不增强
    if color_gate is not None:pixel_mask=pixel_mask*color_gate
    # 两种映射模式：ratio模式只对contrast>1的层级提亮
    if mode=='ratio':
        # ltmo_contrast决定哪些层级生效(>1)，ratio作为全局倍率缩放提亮量
        # 每层有效比例 = 1 + (c-1)*ratio；ratio=1即原始c，ratio=0不变
        out=y.copy().astype(np.float32)
        gain_total=np.zeros((h,w),np.float32)
        r=float(ratio)
        for c,m in zip(ltmo_contrast,scale_masks):
            c=float(c)
            if c>1.0:
                eff=1.0+(c-1.0)*r
                mapped=np.clip(y*eff,0,1)
                delta=m*pixel_mask*(mapped-y)
                out=np.clip(out+delta,0,1)
                gain_total+=m*pixel_mask
        dbg={'pyramid':pyr,'high':high,'scale_masks':scale_masks,'gain':gain_total,
             'pixel_mask':pixel_mask,'color_gate':color_gate,'output':out,'mode':mode}
        return out,dbg
    else:  # gamma模式：原有逻辑
        gain=np.zeros((h,w),np.float32)
        for c,m in zip(ltmo_contrast,scale_masks):gain+=float(c)*m
        gain*=pixel_mask
        table=gamma_lut17(gamma) if lut is None else np.asarray(lut,np.float32)
        mapped=apply_lut17(y,table)
        out=np.clip(y+gain*(mapped-y),0,1).astype(np.float32)
        dbg={'pyramid':pyr,'high':high,'scale_masks':scale_masks,'gain':gain,
             'pixel_mask':pixel_mask,'color_gate':color_gate,'mapped':mapped,'output':out,'mode':mode}
        return out,dbg

def process_one(img,a,out_path,dbg_dir=None):
    if img.dtype!=np.uint8:raise TypeError('当前简化版仅支持8-bit图像')
    if img.ndim==2:y=img.astype(np.float32)/255.;ycc=None;gate=None
    elif img.ndim==3 and img.shape[2]>=3:
        ycc=cv2.cvtColor(img[:,:,:3],cv2.COLOR_BGR2YCrCb);y=ycc[:,:,0].astype(np.float32)/255.
        gate=None if a.no_color_gate else yellow_green_gate(img[:,:,:3],a.hue_min,a.hue_max,a.sat_min,a.gate_soft)
    else:raise ValueError('不支持的图像格式')
    yo,d=process_luma(y,7,a.threshold,a.soft,a.ltmo_contrast,a.mode,a.gamma,a.ratio,a.lut,a.start_scale,color_gate=gate)
    if ycc is None:out=np.rint(yo*255).astype(np.uint8)
    else:
        ycc[:,:,0]=np.rint(yo*255).astype(np.uint8);out=cv2.cvtColor(ycc,cv2.COLOR_YCrCb2BGR)
    Path(out_path).parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(out_path),out):raise IOError(str(out_path))
    if dbg_dir:
        q=Path(dbg_dir);q.mkdir(parents=True,exist_ok=True)
        mx=max(max(a.ltmo_contrast),1e-6)
        cv2.imwrite(str(q/'gain.png'),np.rint(np.clip(d['gain']/mx,0,1)*255).astype(np.uint8))
        cv2.imwrite(str(q/'pixel_mask.png'),np.rint(d['pixel_mask']*255).astype(np.uint8))
        if d.get('color_gate') is not None:
            cv2.imwrite(str(q/'color_gate.png'),np.rint(d['color_gate']*255).astype(np.uint8))
        for i,(p,m) in enumerate(zip(d['pyramid'],d['scale_masks'])):
            cv2.imwrite(str(q/f'L{i}_pyramid.png'),np.rint(np.clip(p,0,1)*255).astype(np.uint8))
            cv2.imwrite(str(q/f'L{i}_mask.png'),np.rint(np.clip(m,0,1)*255).astype(np.uint8))

def main():
    ap=argparse.ArgumentParser(description='全分辨率起始的6尺度ltmo_contrast')
    ap.add_argument('input');ap.add_argument('output')
    ap.add_argument('--threshold',type=float,default=50.)
    ap.add_argument('--soft',type=float,default=20.)
    ap.add_argument('--ltmo-contrast',type=float,nargs=7,default=[1.3,1.4,1.2,1.2,1.4,1.1,1.0])
    ap.add_argument('--mode',choices=['gamma','ratio'],default='ratio',help='映射模式: gamma=gamma校正, ratio=按比例提升')
    ap.add_argument('--gamma',type=float,default=0.7)
    ap.add_argument('--ratio',type=float,default=1.0,help='ratio模式全局倍率: 每层有效比例=1+(ltmo-contrast[i]-1)×ratio(1.0=原始,0=不变,>1整体更强)')
    ap.add_argument('--lut',type=float,nargs=17,default=None)
    ap.add_argument('--start-scale',type=int,default=4,help='L0起始分辨率分母: 1=原图,2=1/2,4=1/4,8=1/8,...')
    ap.add_argument('--hue-min',type=float,default=25.,help='黄绿色门控色相下限(0-180)')
    ap.add_argument('--hue-max',type=float,default=95.,help='黄绿色门控色相上限(0-180)')
    ap.add_argument('--sat-min',type=float,default=15.,help='黄绿色门控饱和度下限(0-255)')
    ap.add_argument('--gate-soft',type=float,default=12.,help='色相/饱和度门控软过渡宽度')
    ap.add_argument('--no-color-gate',action='store_true',help='关闭黄绿色门控，对所有颜色生效')
    ap.add_argument('--debug-dir',default=None)
    a=ap.parse_args()
    input_path=Path(a.input)
    exts={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
    if input_path.is_dir():
        files=sorted(p for p in input_path.rglob('*') if p.is_file() and p.suffix.lower() in exts)
        if not files:raise SystemExit(f'没有找到图片: {a.input}')
        out_root=Path(a.output)
        print(f'共 {len(files)} 张图片')
        for i,f in enumerate(files):
            img=cv2.imread(str(f),cv2.IMREAD_UNCHANGED)
            if img is None:print(f'跳过（读取失败）: {f.name}');continue
            rel=f.relative_to(input_path)
            out_path=out_root/rel
            dbg_dir=(Path(a.debug_dir)/f.stem) if a.debug_dir else None
            try:
                process_one(img,a,out_path,dbg_dir)
            except Exception as e:
                print(f'错误 {f.name}: {e}');continue
            print(f'[{i+1}/{len(files)}] 已处理: {f.name}')
    else:
        img=cv2.imread(a.input,cv2.IMREAD_UNCHANGED)
        if img is None:raise FileNotFoundError(a.input)
        process_one(img,a,a.output,a.debug_dir)
    print('done')

if __name__=='__main__':main()
