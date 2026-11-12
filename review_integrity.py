"""Bind human approval to exact image, checkpoint and accepted-mask contents."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def approval_token(image_digest, checkpoint_digest, mask_bytes, source):
    h=hashlib.sha256()
    for value in (image_digest.encode(),checkpoint_digest.encode(),mask_bytes,source.encode()):
        h.update(len(value).to_bytes(8,'big')); h.update(value)
    return h.hexdigest()


def validate_review(audit, image_path):
    if audit.get('schema_version')!=2 or not audit.get('mask_sha256'):
        raise ValueError('Legacy review: reopen the image and approve the mask again to create a version-2 review')
    if digest(image_path)!=audit['image_sha256']:
        raise ValueError('Reviewed image changed')
    if digest(audit['mask'])!=audit['mask_sha256']:
        raise ValueError('Approved mask changed; review it again')
    if not audit.get('checkpoint_sha256'):
        raise ValueError('Missing checkpoint identity')
    if audit.get('source') not in ('human_uploaded_correction','human_approved_student'):
        raise ValueError('Missing reviewed-label provenance')
    seconds=audit.get('review_seconds')
    if not isinstance(seconds,(int,float)) or not np.isfinite(seconds) or seconds<0:
        raise ValueError('Invalid review duration')
    with Image.open(image_path) as image, Image.open(audit['mask']) as mask:
        if image.size!=mask.size: raise ValueError('Mask dimensions do not match image')
        values=np.asarray(mask)
        if values.ndim!=2 or not np.isin(values,[0,255]).all():
            raise ValueError('Approved mask must be single-channel binary 0/255')
