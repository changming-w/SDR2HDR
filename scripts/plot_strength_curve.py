from PIL import Image, ImageDraw
import numpy as np
exec(open('scripts/focus_brighten.py').read().split('def brighten_with_focus')[0])
y = np.linspace(0, 255, 512)
s = dynamic_strength(y, peak=1.0)
W, H = 800, 500
img = Image.new('RGB', (W, H), 'white')
draw = ImageDraw.Draw(img)
ml, mr, mt, mb = 80, 40, 40, 60
pw, ph = W - ml - mr, H - mt - mb
x2px = lambda v: int(ml + v/255.0*pw)
y2py = lambda v: int(mt + ph - v*ph)
for yv in [0.2,0.4,0.6,0.8,1.0]:
    py = y2py(yv)
    draw.line([(ml,py),(W-mr,py)], fill='#e0e0e0', width=1)
    draw.text((ml-20,py-6), f'{yv:.1f}', fill='#666')
for xv in [0,40,50,60,80,100,128,180,200,255]:
    px = x2px(xv)
    draw.line([(px,mt),(px,H-mb)], fill='#e0e0e0', width=1)
    draw.text((px-10,H-mb+5), str(xv), fill='#666')
draw.line([(ml,mt),(ml,H-mb),(W-mr,H-mb)], fill='#333', width=2)
draw.polygon([(W-mr-10,H-mb-6),(W-mr+5,H-mb),(W-mr-10,H-mb+6)], fill='#333')
draw.polygon([(ml-6,mt+10),(ml,mt-5),(ml+6,mt+10)], fill='#333')
pts = [(x2px(y[i]),y2py(s[i])) for i in range(len(y))]
pts += [(x2px(255),y2py(0)),(x2px(0),y2py(0))]
draw.polygon(pts, fill=(52,152,219,38))
for i in range(len(y)-1):
    draw.line([(x2px(y[i]),y2py(s[i])),(x2px(y[i+1]),y2py(s[i+1]))], fill='#2980b9', width=3)
for xv,yv in zip([0,40,50,60,80,100,128,180,200,255],[0,0.05,0.1,0.20,0.70,1.00,0.95,0.40,0.20,0.00]):
    px, py = x2px(xv), y2py(yv)
    draw.ellipse([px-5,py-5,px+5,py+5], fill='#e74c3c')
draw.text((W//2-200,5), 'Dynamic Strength Curve (Catmull-Rom Spline)', fill='#333')
draw.text((W//2-80,H-20), 'Y (luminance, 0-255)', fill='#555')
draw.text((10,mt+ph//2), 'Strength', fill='#555')
img.save('strength_curve_6seg.png')
print('Done: strength_curve_6seg.png')