import unittest
import torch
import numpy as np
from student import Student,metrics,Segments,rows
from torch.utils.data import DataLoader,TensorDataset

class Tests(unittest.TestCase):
    def test_shape_and_gradient(self):
        m=Student(); x=torch.rand(2,3,24,24); y=m(x)
        self.assertEqual(tuple(y.shape),(2,1,24,24)); y.mean().backward()
        self.assertIsNotNone(m.head.weight.grad)
    def test_metric(self):
        class Perfect(torch.nn.Module):
            def forward(self,x):return x[:,0:1]*20-10
        x=torch.ones(2,3,8,8); y=torch.ones(2,1,8,8)
        self.assertEqual(metrics(Perfect(),DataLoader(TensorDataset(x,y)))['iou'],1)
    def test_demo_split_integrity(self):
        from pathlib import Path
        if not Path('data/demo/manifest.json').exists():self.skipTest('Generate demo first')
        records=rows('data/demo/manifest.json')
        self.assertTrue(all(not r.get('mask') for r in records if r['split']=='pool'))
    def test_import_preserves_test_split(self):
        import tempfile,json,hashlib,sys
        from pathlib import Path
        from PIL import Image
        from unittest.mock import patch
        from import_reviews import main
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'reviews').mkdir()
            Image.new('RGB',(8,8),(255,0,0)).save(root/'pool.png')
            Image.new('RGB',(8,8),(0,255,0)).save(root/'test.png')
            Image.new('L',(8,8),255).save(root/'mask.png')
            manifest=[dict(id='pool',image='pool.png',split='pool'),dict(id='test',image='test.png',split='test',mask='mask.png')]
            (root/'manifest.json').write_text(json.dumps(manifest))
            audit=dict(id='pool',image_sha256=hashlib.sha256((root/'pool.png').read_bytes()).hexdigest(),mask=str(root/'mask.png'),source='human_uploaded_correction',review_seconds=2)
            (root/'reviews'/'audit.json').write_text(json.dumps(audit))
            with patch.object(sys,'argv',['import_reviews.py',str(root/'manifest.json'),'--reviews',str(root/'reviews'),'--output',str(root/'round2.json')]): main()
            result=json.loads((root/'round2.json').read_text())
            self.assertEqual(result[0]['split'],'train'); self.assertEqual(result[1]['split'],'test')
            self.assertEqual(json.loads((root/'manifest.json').read_text())[0]['split'],'pool')
if __name__=='__main__':unittest.main()
