import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from student import resolve_device,train,load_model,predict,rank


class DeviceTests(unittest.TestCase):
    def test_auto_and_explicit_cpu(self):
        with patch('torch.cuda.is_available',return_value=False):
            self.assertEqual(resolve_device('auto').type,'cpu')
            with self.assertRaisesRegex(ValueError,'CUDA requested'):resolve_device('cuda')
        self.assertEqual(resolve_device('cpu').type,'cpu')

    def exercise(self,device):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);records=[]
            for i,split in enumerate(('train','validation','pool')):
                image=root/(split+'.png');mask=root/(split+'-mask.png')
                Image.new('RGB',(24,24),(120+i*30,10,20)).save(image)
                record=dict(id=split,image=image.name,split=split)
                if split!='pool':Image.new('L',(24,24),255).save(mask);record['mask']=mask.name
                records.append(record)
            manifest=root/'manifest.json';manifest.write_text(json.dumps(records))
            train(manifest,root/'run',epochs=1,device=device)
            history=json.loads((root/'run/history.json').read_text())
            self.assertEqual(history['device'],device)
            checkpoint=root/'run/student.pt'
            model=load_model(checkpoint,device)
            self.assertEqual(next(model.parameters()).device.type,device)
            prediction=predict(model,root/'pool.png')
            self.assertEqual(prediction.shape,(96,96));self.assertTrue(np.isfinite(prediction).all())
            cpu=predict(load_model(checkpoint,'cpu'),root/'pool.png')
            np.testing.assert_allclose(prediction,cpu,atol=1e-3,rtol=1e-3)
            self.assertEqual(rank(manifest,checkpoint,1,device=device)[0]['id'],'pool')
            if device=='cuda':self.assertGreater(history['gpu_peak_allocated_bytes'],0)

    def test_cpu_training_prediction_and_ranking(self):self.exercise('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'Requires a CUDA runtime and visible GPU')
    def test_cuda_training_prediction_and_ranking(self):self.exercise('cuda')
