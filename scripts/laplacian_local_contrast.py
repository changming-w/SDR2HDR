#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""六尺度拉普拉斯局部对比度增强：亮侧/暗侧独立控制。
依赖: pip install numpy opencv-python
参数顺序: L1最粗/最大范围 -> L6最细/最小范围。
"""
import argparse
from pathlib import Path
import cv2
import numpy as np

BG0=[0.96,1.26,1.0,1.52,1.2,1.0]
DG0=[0.98,0.98,1.0,1.0,1.0,1.0]
TH0=[0,0,0,0,0,0]
CL0=[255,255,255,255,255,255]
STRENGTH=1.16
PROTECT=0.005
GLOBAL_GAIN=0.977
GLOBAL_OFFSET=0.045

def vec(name,x,n):
    x=np.asarray(x,np.float32)
    if x.ndim!=1 or x.size!=n or not np.all(np.isfinite(x)):
        raise ValueError(f"{name}必须包含{n}个有效数值")
    return x

def smoothstep(x):
    x=np.clip(x,0.,1.)
    return x*x*(3.-2.*x)

def detail_gate(a,t):
    if t<=0:return np.ones_like(a,np.float32)
    return smoothstep((a-t)/max(t,1e-12))

def build_pyramid(y,levels=6):
    y=np.asarray(y,np.float32)
    if y.ndim!=2:raise ValueError("y必须是二维亮度图")
    if min(y.shape)<2**levels:
        raise ValueError(f"{levels}层金字塔要求短边至少{2**levels}")
    gs=[y]
    for _ in range(levels):gs.append(cv2.pyrDown(gs[-1]))
    ls=[]
    for i in range(levels):
        h,w=gs[i].shape
        ls.append(gs[i]-cv2.pyrUp(gs[i+1],dstsize=(w,h)))
    return gs,ls

def enhance_luma_laplacian(y,bright_gains=BG0,dark_gains=DG0,
        thresholds_8bit=TH0,clips_8bit=CL0,protect=0.03,strength=1.0,
        global_gain=1.0,global_offset=0.0):
    """y为float32 [0,1]。bright>1提亮正细节；dark>1压暗负细节；
    dark<1提亮负细节；增益=1不改变。
    global_gain: 全局亮度乘数（>1提亮，<1压暗）
    global_offset: 全局亮度偏移（正数提亮，负数压暗）"""
    y=np.asarray(y,np.float32); n=len(bright_gains)
    bg=vec("bright_gains",bright_gains,n)
    dg=vec("dark_gains",dark_gains,n)
    th=vec("thresholds",thresholds_8bit,n)/255.
    cl=vec("clips",clips_8bit,n)/255.
    if np.any(bg<0) or np.any(dg<0) or np.any(th<0):
        raise ValueError("增益和阈值不能为负")
    if not 0<=protect<0.5 or strength<0:
        raise ValueError("protect应为[0,0.5)，strength应>=0")
    gs,bands=build_pyramid(y,n)
    # 全局亮度调整（在最粗高斯层上操作）
    if global_gain != 1.0 or global_offset != 0.0:
        gs[-1] = np.clip(gs[-1] * global_gain + global_offset, 0, 1)
    enhanced=[]; bds=[]; dds=[]; deltas=[]
    for i,band in enumerate(bands):
        k=n-1-i                 # 内部细->粗，参数粗->细
        pos=np.maximum(band,0.); neg=np.minimum(band,0.)
        gate=detail_gate(np.abs(band),float(th[k]))
        bd=strength*(float(bg[k])-1.)*pos*gate
        dd=strength*(float(dg[k])-1.)*neg*gate
        delta=bd+dd
        if protect>0:
            local=gs[i]
            wr=smoothstep((1.-local)/protect)
            br=smoothstep(local/protect)
            weight=np.where(delta>=0,wr,br)
            bd*=weight; dd*=weight; delta=bd+dd
        if cl[k]>0:
            scale=np.minimum(1.,float(cl[k])/np.maximum(np.abs(delta),1e-12))
            bd*=scale; dd*=scale; delta=bd+dd
        bd=bd.astype(np.float32); dd=dd.astype(np.float32)
        delta=delta.astype(np.float32)
        enhanced.append(band+delta); bds.append(bd); dds.append(dd)
        deltas.append(delta)
    out=gs[-1].copy()
    for i in range(n-1,-1,-1):
        h,w=enhanced[i].shape
        out=cv2.pyrUp(out,dstsize=(w,h))+enhanced[i]
    out=np.clip(out,0,1).astype(np.float32)
    debug={"input_luma":y,"output_luma":out,"bands":bands,
           "bright_deltas":bds,"dark_deltas":dds,"deltas":deltas}
    return out,debug

def normalize(img):
    if img.dtype==np.uint8:return img.astype(np.float32)/255.,img.dtype
    if img.dtype==np.uint16:return img.astype(np.float32)/65535.,img.dtype
    if np.issubdtype(img.dtype,np.floating):
        x=img.astype(np.float32)
        if not np.all(np.isfinite(x)) or x.min()<0 or x.max()>1:
            raise ValueError("浮点输入必须位于[0,1]")
        return x,img.dtype
    raise TypeError(f"不支持的数据类型:{img.dtype}")

def restore(x,dtype):
    x=np.clip(x,0,1)
    if dtype==np.uint8:return np.rint(x*255).astype(np.uint8)
    if dtype==np.uint16:return np.rint(x*65535).astype(np.uint16)
    return x.astype(dtype)


def process_image(img,bright_gains=BG0,dark_gains=DG0,
        thresholds_8bit=TH0,clips_8bit=CL0,protect=0.03,strength=STRENGTH,
        global_gain=1.0,global_offset=0.0):
    """支持uint8/uint16灰度、BGR、BGRA。彩色图只修改Y通道。"""
    x,dtype=normalize(img); alpha=None
    if x.ndim==2:
        y=x.copy(); ycc=None
    elif x.ndim==3 and x.shape[2] in (3,4):
        if x.shape[2]==4:
            alpha=x[:,:,3].copy(); bgr=x[:,:,:3]
        else:bgr=x
        ycc=cv2.cvtColor(bgr,cv2.COLOR_BGR2YCrCb)
        y=ycc[:,:,0].copy()
    else:raise ValueError(f"不支持的图像形状:{x.shape}")
    yo,debug=enhance_luma_laplacian(y,bright_gains,dark_gains,
        thresholds_8bit,clips_8bit,protect,strength,global_gain,global_offset)
    if ycc is None:out=yo
    else:
        ycc[:,:,0]=yo
        bgr=np.clip(cv2.cvtColor(ycc,cv2.COLOR_YCrCb2BGR),0,1)
        out=np.dstack([bgr,alpha]) if alpha is not None else bgr
    return restore(out,dtype),debug

def save_gray(path,x):
    x=np.rint(np.clip(x,0,1)*255).astype(np.uint8)
    if not cv2.imwrite(str(path),x):raise IOError(f"写入失败:{path}")

def save_signed(path,x):
    s=max(float(np.percentile(np.abs(x),99.5)),1e-8)
    save_gray(path,0.5+0.5*x/s)

def save_debug(folder,d):
    out=Path(folder); out.mkdir(parents=True,exist_ok=True)
    save_gray(out/'input_luma.png',d['input_luma'])
    save_gray(out/'output_luma.png',d['output_luma'])
    save_signed(out/'output_delta.png',d['output_luma']-d['input_luma'])
    n=len(d['bands'])
    for i in range(n):
        lv=n-i
        save_signed(out/f'L{lv}_band.png',d['bands'][i])
        save_signed(out/f'L{lv}_bright_delta.png',d['bright_deltas'][i])
        save_signed(out/f'L{lv}_dark_delta.png',d['dark_deltas'][i])
        save_signed(out/f'L{lv}_total_delta.png',d['deltas'][i])

def main():
    p=argparse.ArgumentParser(description='六尺度拉普拉斯局部对比度增强（亮暗独立）')
    p.add_argument('input'); p.add_argument('output')
    p.add_argument('--bright-gains',type=float,nargs=6,default=BG0)
    p.add_argument('--dark-gains',type=float,nargs=6,default=DG0)
    p.add_argument('--gains',type=float,nargs=6,default=None,
                   help='兼容旧版:同时覆盖亮侧和暗侧')
    p.add_argument('--thresholds',type=float,nargs=6,default=TH0)
    p.add_argument('--clips',type=float,nargs=6,default=CL0)
    p.add_argument('--protect',type=float,default=PROTECT)
    p.add_argument('--strength',type=float,default=STRENGTH)
    p.add_argument('--global-gain',type=float,default=GLOBAL_GAIN,help='全局亮度乘数')
    p.add_argument('--global-offset',type=float,default=GLOBAL_OFFSET,help='全局亮度偏移')
    p.add_argument('--debug-dir',default=None)
    a=p.parse_args(); bg=a.bright_gains; dg=a.dark_gains
    if a.gains is not None:bg=dg=a.gains
    img=cv2.imread(a.input,cv2.IMREAD_UNCHANGED)
    if img is None:raise FileNotFoundError(f"读取失败:{a.input}")
    result,debug=process_image(img,bg,dg,a.thresholds,a.clips,
                               a.protect,a.strength,a.global_gain,a.global_offset)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(a.output,result):raise IOError(f"写入失败:{a.output}")
    if a.debug_dir:save_debug(a.debug_dir,debug)
    print(f"完成:{a.input} -> {a.output}")
    print("尺度:L1最粗/最大范围 -> L6最细/最小范围")
    print("亮侧增益:",bg); print("暗侧增益:",dg)

if __name__=='__main__':main()
