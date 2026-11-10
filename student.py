"""Small binary segmentation student, reproducible training, and active-pool ranking."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from selection import select

class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,32,3,padding=1),nn.ReLU())
        self.head=nn.Conv2d(32,1,1)
    def forward(self,x): return self.head(self.features(x))

def rows(path):
    path=Path(path); records=json.loads(path.read_text(encoding='utf-8'))
    if not records: raise ValueError('Manifest is empty')
    ids=[r['id'] for r in records]
    if len(set(ids))!=len(ids): raise ValueError('Duplicate sample IDs')
    for r in records:
        if r.get('split') not in ('train','validation','test','pool'): raise ValueError('Invalid split')
        for key in ('image','mask'):
            if r.get(key): r[key]=str((path.parent/r[key]).resolve())
    hashes={}
    for r in records:
        h=hashlib.sha256(Path(r['image']).read_bytes()).hexdigest()
        if h in hashes and hashes[h]!=r['split']: raise ValueError('Image content overlaps splits')
        hashes[h]=r['split']
    return records

def image_tensor(path,size=96):
    with Image.open(path) as im: x=np.asarray(im.convert('RGB').resize((size,size)),dtype=np.float32)/255
    return torch.from_numpy(x.copy()).permute(2,0,1)

class Segments(Dataset):
    def __init__(self,records,size=96): self.records,self.size=records,size
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        r=self.records[i]; x=image_tensor(r['image'],self.size)
        with Image.open(r['mask']) as im:
            y=np.asarray(im.convert('L').resize((self.size,self.size),Image.Resampling.NEAREST))>0
        return x,torch.tensor(y,dtype=torch.float32)[None]

@torch.inference_mode()
def metrics(model,loader):
    model.eval(); intersection=union=0; loss=0; n=0
    for x,y in loader:
        logits=model(x); loss+=nn.functional.binary_cross_entropy_with_logits(logits,y).item()*len(x); n+=len(x)
        pred=logits>0; target=y.bool(); intersection+=(pred&target).sum().item(); union+=(pred|target).sum().item()
    return dict(iou=intersection/union if union else 1.,bce=loss/n)

def train(manifest,output,epochs=10,seed=42):
    if epochs<1: raise ValueError('epochs must be positive')
    output=Path(output)
    if output.exists(): raise ValueError('Choose a fresh run folder')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(2)
    records=rows(manifest)
    groups={s:[r for r in records if r['split']==s and r.get('mask')] for s in ('train','validation')}
    if not all(groups.values()): raise ValueError('Need labeled training and separate validation images')
    model=Student(); optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    loaders={s:DataLoader(Segments(rs),batch_size=4,shuffle=s=='train') for s,rs in groups.items()}
    output.mkdir(parents=True); history=[]; best=-1
    for epoch in range(epochs):
        model.train()
        for x,y in loaders['train']:
            optimizer.zero_grad(); loss=nn.functional.binary_cross_entropy_with_logits(model(x),y); loss.backward(); optimizer.step()
        result=dict(epoch=epoch+1,**metrics(model,loaders['validation'])); history.append(result)
        if result['iou']>best:
            best=result['iou']; torch.save(model.state_dict(),output/'student.pt')
        print(result,flush=True)
    (output/'history.json').write_text(json.dumps(dict(seed=seed,history=history,selection='best validation IoU',train_ids=[r['id'] for r in groups['train']]),indent=2))
    return history

def load_model(checkpoint):
    model=Student(); model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)); return model.eval()

@torch.inference_mode()
def predict(model,path): return model(image_tensor(path)[None]).sigmoid()[0,0].numpy()

@torch.inference_mode()
def rank(manifest,checkpoint,budget,strategy='combined',seed=42):
    pool=[r for r in rows(manifest) if r['split']=='pool']
    if not 0<=budget<=len(pool): raise ValueError('Budget exceeds pool size')
    if not budget:return []
    model=load_model(checkpoint); embeddings=[]; uncertainty=[]
    for r in pool:
        x=image_tensor(r['image'])[None]; features=model.features(x); p=model.head(features).sigmoid().clamp(1e-6,1-1e-6)
        embeddings.append(features.mean((2,3))[0].numpy()); uncertainty.append(float((-p*p.log()-(1-p)*(1-p).log()).mean()))
    if strategy=='random': chosen=np.random.default_rng(seed).choice(len(pool),budget,replace=False).tolist()
    elif strategy=='uncertainty': chosen=np.argsort(-np.array(uncertainty))[:budget].tolist()
    elif strategy in ('combined','diversity'): chosen=select(embeddings,uncertainty,budget,weight=.5 if strategy=='combined' else 0)
    else: raise ValueError('Unknown strategy')
    return [dict(id=pool[i]['id'],image=pool[i]['image'],uncertainty=uncertainty[i],strategy=strategy) for i in chosen]

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train'); t.add_argument('manifest'); t.add_argument('--output',required=True); t.add_argument('--epochs',type=int,default=10)
    r=sub.add_parser('rank'); r.add_argument('manifest'); r.add_argument('--checkpoint',required=True); r.add_argument('--budget',type=int,default=4); r.add_argument('--strategy',choices=['random','uncertainty','diversity','combined'],default='combined'); r.add_argument('--output',default='selection.json')
    e=sub.add_parser('evaluate'); e.add_argument('manifest'); e.add_argument('--checkpoint',required=True)
    a=vars(p.parse_args()); cmd=a.pop('command')
    if cmd=='train': train(**a)
    elif cmd=='rank':
        output=a.pop('output'); result=rank(**a); Path(output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    else:
        data=[r for r in rows(a['manifest']) if r['split']=='test' and r.get('mask')]
        if not data: raise ValueError('No labeled test samples')
        print(json.dumps(metrics(load_model(a['checkpoint']),DataLoader(Segments(data),batch_size=4)),indent=2))
if __name__=='__main__': main()
