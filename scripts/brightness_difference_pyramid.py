#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同分辨率多尺度亮度差分金字塔（Brightness Difference Pyramid）。
L1最大尺度->L6最小尺度；不使用pyrDown/pyrUp，所有频带和原图同尺寸。
依赖：pip install numpy opencv-python
"""
import argparse
from pathlib import Path
import cv2
import numpy as np

SCALES0=[32.,16.,8.,4.,2.,1.]
BG0=[1.12,1.43,1.21,1.53,1.15,1.25]
DG0=[0.48,0.42,1.,1.,1.,1.]
TH0=[2.0,0.,0.,0.,0.1,0.]
CL0=[40.,195.,196.,193.,196.,191.]
BMS0=0.0
BME0=0.206
PROTECT0=0.074
STRENGTH0=1.65

def vec(name,x,n):
    x=np.asarray(x,np.float32)
    if x.ndim!=1 or x.size!=n or not np.all(np.isfinite(x)):
        raise ValueError(f'{name}必须包含{n}个有效数值')
    return x

def smoothstep(x):
    x=np.clip(x,0.,1.)
    return x*x*(3.-2.*x)

def gate(a,t):
    if t<=0:return np.ones_like(a,np.float32)
    return smoothstep((a-t)/max(t,1e-12))

def build_difference_pyramid(y,scales=SCALES0):
    """scales按L1粗->L6细输入；返回bands内部顺序为细->粗。"""
    y=np.asarray(y,np.float32); s=np.asarray(scales,np.float32)
    if y.ndim!=2:raise ValueError('y必须是二维亮度图')
    if np.any(s<=0) or np.any(s[:-1]<=s[1:]):
        raise ValueError('scales必须为严格递减正数，例如32 16 8 4 2 1')
    fine_to_coarse=s[::-1]
    bases=[y]
    # 每个base都直接由原亮度图产生，sigma含义固定、便于寄存器映射。
    for sigma in fine_to_coarse:
        b=cv2.GaussianBlur(y,(0,0),float(sigma),float(sigma),
                           borderType=cv2.BORDER_REFLECT101)
        bases.append(b.astype(np.float32))
    bands=[bases[i]-bases[i+1] for i in range(len(s))]
    return bases,bands

def adjust_brightness_difference_pyramid(
        y,scales=SCALES0,bright_gains=BG0,dark_gains=DG0,
        thresholds_8bit=TH0,clips_8bit=CL0,
        bright_mask_start=0.12,bright_mask_end=0.35,
        protect=0.04,strength=1.0):
    """
    y: float32 [0,1]。
    bright_gains控制正亮度差；dark_gains控制负亮度差。
    bright_mask_start/end：暗区保护。y<=start不提亮正差分，y>=end完全生效。
    gain=1不变；bright>1提亮；dark>1压暗；dark<1提亮负差分。
    """
    y=np.asarray(y,np.float32); n=len(scales)
    sc=vec('scales',scales,n); bg=vec('bright_gains',bright_gains,n)
    dg=vec('dark_gains',dark_gains,n)
    th=vec('thresholds',thresholds_8bit,n)/255.
    cl=vec('clips',clips_8bit,n)/255.
    if np.any(bg<0) or np.any(dg<0) or np.any(th<0):
        raise ValueError('增益和阈值不能为负数')
    if not 0<=protect<0.5 or strength<0:
        raise ValueError('protect应为[0,0.5)，strength应>=0')
    if bright_mask_end>bright_mask_start:
        bright_region=smoothstep((y-bright_mask_start)/
                                 (bright_mask_end-bright_mask_start))
    else:bright_region=np.ones_like(y,np.float32)

    bases,bands=build_difference_pyramid(y,sc)
    adjusted=[]; bds=[]; dds=[]; deltas=[]
    for i,band in enumerate(bands):
        k=n-1-i  # bands细->粗，参数L1粗->L6细
        pos=np.maximum(band,0.); neg=np.minimum(band,0.)
        w=gate(np.abs(band),float(th[k]))
        bd=strength*(float(bg[k])-1.)*pos*w*bright_region
        dd=strength*(float(dg[k])-1.)*neg*w
        delta=bd+dd
        if protect>0:
            white_room=smoothstep((1.-y)/protect)
            black_room=smoothstep(y/protect)
            headroom=np.where(delta>=0,white_room,black_room)
            bd*=headroom; dd*=headroom; delta=bd+dd
        if cl[k]>0:
            scale=np.minimum(1.,float(cl[k])/np.maximum(np.abs(delta),1e-12))
            bd*=scale; dd*=scale; delta=bd+dd
        bd=bd.astype(np.float32); dd=dd.astype(np.float32)
        delta=(bd+dd).astype(np.float32)
        adjusted.append(band+delta); bds.append(bd); dds.append(dd)
        deltas.append(delta)
    # 恒等重建：Y = 最粗base + 所有差分频带之和。
    out=bases[-1].copy()
    for band in adjusted:out+=band
    out=np.clip(out,0,1).astype(np.float32)
    debug={'input_luma':y,'output_luma':out,'bases':bases,'bands':bands,
           'bright_deltas':bds,'dark_deltas':dds,'deltas':deltas,
           'bright_region_mask':bright_region}
    return out,debug


def normalize(img):
    if img.dtype==np.uint8:return img.astype(np.float32)/255.,img.dtype
    if img.dtype==np.uint16:return img.astype(np.float32)/65535.,img.dtype
    if np.issubdtype(img.dtype,np.floating):
        x=img.astype(np.float32)
        if not np.all(np.isfinite(x)) or x.min()<0 or x.max()>1:
            raise ValueError('浮点输入必须位于[0,1]')
        return x,img.dtype
    raise TypeError(f'不支持的数据类型:{img.dtype}')

def restore(x,dtype):
    x=np.clip(x,0,1)
    if dtype==np.uint8:return np.rint(x*255).astype(np.uint8)
    if dtype==np.uint16:return np.rint(x*65535).astype(np.uint16)
    return x.astype(dtype)

def process_image(img,**kwargs):
    """支持uint8/uint16灰度、BGR、BGRA；彩色输入只修改Y通道。"""
    x,dtype=normalize(img); alpha=None
    if x.ndim==2:y=x.copy(); ycc=None
    elif x.ndim==3 and x.shape[2] in (3,4):
        if x.shape[2]==4:alpha=x[:,:,3].copy(); bgr=x[:,:,:3]
        else:bgr=x
        ycc=cv2.cvtColor(bgr,cv2.COLOR_BGR2YCrCb); y=ycc[:,:,0].copy()
    else:raise ValueError(f'不支持的图像形状:{x.shape}')
    yo,debug=adjust_brightness_difference_pyramid(y,**kwargs)
    if ycc is None:out=yo
    else:
        ycc[:,:,0]=yo
        bgr=np.clip(cv2.cvtColor(ycc,cv2.COLOR_YCrCb2BGR),0,1)
        out=np.dstack([bgr,alpha]) if alpha is not None else bgr
    return restore(out,dtype),debug

def save_gray(path,x):
    x=np.rint(np.clip(x,0,1)*255).astype(np.uint8)
    if not cv2.imwrite(str(path),x):raise IOError(f'写入失败:{path}')

def save_signed(path,x):
    s=max(float(np.percentile(np.abs(x),99.5)),1e-8)
    save_gray(path,0.5+0.5*x/s)

def save_debug(folder,d):
    out=Path(folder); out.mkdir(parents=True,exist_ok=True)
    save_gray(out/'input_luma.png',d['input_luma'])
    save_gray(out/'output_luma.png',d['output_luma'])
    save_gray(out/'bright_region_mask.png',d['bright_region_mask'])
    save_signed(out/'output_delta.png',d['output_luma']-d['input_luma'])
    n=len(d['bands'])
    for i in range(n):
        lv=n-i
        save_signed(out/f'L{lv}_band.png',d['bands'][i])
        save_signed(out/f'L{lv}_bright_delta.png',d['bright_deltas'][i])
        save_signed(out/f'L{lv}_dark_delta.png',d['dark_deltas'][i])
        save_signed(out/f'L{lv}_total_delta.png',d['deltas'][i])

def main():
    p=argparse.ArgumentParser(description='同分辨率多尺度亮度差分金字塔调整')
    p.add_argument('input');p.add_argument('output')
    p.add_argument('--scales',type=float,nargs=6,default=SCALES0)
    p.add_argument('--bright-gains',type=float,nargs=6,default=BG0)
    p.add_argument('--dark-gains',type=float,nargs=6,default=DG0)
    p.add_argument('--thresholds',type=float,nargs=6,default=TH0)
    p.add_argument('--clips',type=float,nargs=6,default=CL0)
    p.add_argument('--bright-mask-start',type=float,default=BMS0)
    p.add_argument('--bright-mask-end',type=float,default=BME0)
    p.add_argument('--protect',type=float,default=PROTECT0)
    p.add_argument('--strength',type=float,default=STRENGTH0)
    p.add_argument('--debug-dir',default=None)
    a=p.parse_args();img=cv2.imread(a.input,cv2.IMREAD_UNCHANGED)
    if img is None:raise FileNotFoundError(f'读取失败:{a.input}')
    out,d=process_image(img,scales=a.scales,bright_gains=a.bright_gains,
        dark_gains=a.dark_gains,thresholds_8bit=a.thresholds,
        clips_8bit=a.clips,bright_mask_start=a.bright_mask_start,
        bright_mask_end=a.bright_mask_end,protect=a.protect,strength=a.strength)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(a.output,out):raise IOError(f'写入失败:{a.output}')
    if a.debug_dir:save_debug(a.debug_dir,d)
    print(f'完成:{a.input} -> {a.output}')
    print('尺度sigma(L1->L6):',a.scales)
    print('亮侧增益:',a.bright_gains);print('暗侧增益:',a.dark_gains)

if __name__=='__main__':main()
