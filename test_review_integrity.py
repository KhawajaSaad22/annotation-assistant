import tempfile
from pathlib import Path
import unittest
from PIL import Image
from review_integrity import digest,approval_token,validate_review


class ReviewTests(unittest.TestCase):
    def test_modified_mask_and_legacy_review_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); image=root/'image.png'; mask=root/'mask.png'
            Image.new('RGB',(4,4),'red').save(image)
            Image.new('L',(4,4),255).save(mask)
            audit=dict(schema_version=2,image_sha256=digest(image),mask=str(mask),
                       mask_sha256=digest(mask),checkpoint_sha256='checkpoint',
                       source='human_approved_student',review_seconds=1)
            validate_review(audit,image)
            Image.new('L',(4,4),0).save(mask)
            with self.assertRaisesRegex(ValueError,'mask changed'): validate_review(audit,image)
            audit['mask_sha256']=digest(mask); audit.pop('schema_version')
            with self.assertRaisesRegex(ValueError,'Legacy review'): validate_review(audit,image)

    def test_nonbinary_mask_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); image=root/'image.png'; mask=root/'mask.png'
            Image.new('RGB',(4,4),'red').save(image); Image.new('L',(4,4),128).save(mask)
            audit=dict(schema_version=2,image_sha256=digest(image),mask=str(mask),
                       mask_sha256=digest(mask),checkpoint_sha256='checkpoint',
                       source='human_uploaded_correction',review_seconds=1)
            with self.assertRaisesRegex(ValueError,'binary'): validate_review(audit,image)

    def test_approval_token_changes_with_each_review_input(self):
        args=['image','checkpoint',b'mask','human_approved_student']
        token=approval_token(*args)
        for i,value in enumerate(['new-image','new-checkpoint',b'new-mask','human_uploaded_correction']):
            changed=args.copy(); changed[i]=value
            self.assertNotEqual(token,approval_token(*changed))
