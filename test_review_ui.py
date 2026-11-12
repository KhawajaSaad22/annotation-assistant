import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest


class ReviewUITests(unittest.TestCase):
    def test_approval_resets_after_image_and_checkpoint_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); image=root/'image.png'; weights=root/'model.pt'; manifest=root/'manifest.json'
            Image.new('RGB',(8,8),'red').save(image); weights.write_bytes(b'checkpoint-one')
            manifest.write_text(json.dumps([dict(id='pool',image=str(image),split='pool')]))
            source=Path('app.py').read_text().replace("'data/demo/manifest.json'",repr(str(manifest)))
            source=source.replace("'runs/demo/student.pt'",repr(str(weights)))
            with patch('student.load_model',return_value='fake-model'),patch('student.predict',return_value=np.zeros((8,8))):
                app=AppTest.from_string(source).run(timeout=20)
                self.assertFalse(app.exception)
                app.checkbox[0].check().run()
                self.assertTrue(app.checkbox[0].value)
                Image.new('RGB',(8,8),'green').save(image)
                app.run()
                self.assertFalse(app.checkbox[0].value)
                app.checkbox[0].check().run()
                weights.write_bytes(b'checkpoint-two')
                app.run()
                self.assertFalse(app.checkbox[0].value)
                self.assertFalse(app.exception)
