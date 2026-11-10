"""Synthetic binary segmentation data. Not evidence of real-world annotation savings."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw

def main():
    root=Path('data/demo')
    if root.exists(): raise ValueError('Demo exists; use it rather than overwriting annotations')
    root.mkdir(parents=True); rng=np.random.default_rng(14); rows=[]
    for split,count in [('train',12),('validation',6),('test',6),('pool',12)]:
        for i in range(count):
            name=f'{split}-{i:03}'; im=Image.fromarray(rng.integers(0,40,(96,96,3),dtype='uint8')); mask=Image.new('L',(96,96),0)
            x,y=map(int,rng.integers(10,50,2)); box=(x,y,x+30,y+30)
            ImageDraw.Draw(im).ellipse(box,fill=(220,60,40)); ImageDraw.Draw(mask).ellipse(box,fill=255)
            im.save(root/(name+'.png')); r=dict(id=name,image=name+'.png',split=split,provenance='synthetic')
            if split!='pool': mask.save(root/(name+'-mask.png')); r['mask']=name+'-mask.png'
            rows.append(r)
    (root/'manifest.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__': main()
