"""Review a predicted mask, optionally upload a correction, and save an audit record."""
import hashlib
import io
import json
from pathlib import Path
import time
import uuid
import numpy as np
from PIL import Image
import streamlit as st
from student import rows,load_model,predict

st.set_page_config(page_title='Annotation assistant',layout='wide'); st.title('Review segmentation suggestions')
manifest=st.text_input('Manifest','data/demo/manifest.json'); checkpoint=st.text_input('Student checkpoint','runs/demo/student.pt')
@st.cache_resource
def model(path,mtime): return load_model(path)
try:
    pool=[r for r in rows(manifest) if r['split']=='pool']
    if not pool: st.info('No pool images.'); st.stop()
    item=st.selectbox('Image',[r['id'] for r in pool]); record=next(r for r in pool if r['id']==item)
    key=(item,checkpoint)
    if st.session_state.get('review_key')!=key: st.session_state.review_key=key; st.session_state.started=time.monotonic()
    m=model(checkpoint,Path(checkpoint).stat().st_mtime_ns); probability=predict(m,record['image'])
    with Image.open(record['image']) as image: original=image.convert('RGB').copy()
    mask=Image.fromarray(np.uint8(probability>=.5)*255).resize(original.size,Image.Resampling.NEAREST)
    st.image([original,mask],caption=['Image','Student suggestion'])
    correction=st.file_uploader('Optional corrected binary mask (PNG, matching original dimensions)',type=['png'])
    confirmed=st.checkbox('I reviewed this mask and approve it as a training label')
    if st.button('Save reviewed label',disabled=not confirmed):
        accepted=mask
        if correction:
            with Image.open(io.BytesIO(correction.getvalue())) as uploaded:
                if uploaded.size!=original.size: raise ValueError('Correction dimensions must match original image')
                accepted=uploaded.convert('L').point(lambda p:255 if p>0 else 0)
        output=Path('annotations'); output.mkdir(exist_ok=True); identity=uuid.uuid4().hex
        saved=output/(identity+'.png'); accepted.save(saved)
        audit=dict(id=item,image=record['image'],image_sha256=hashlib.sha256(Path(record['image']).read_bytes()).hexdigest(),
                   mask=str(saved.resolve()),checkpoint=str(Path(checkpoint).resolve()),
                   source='human_uploaded_correction' if correction else 'human_approved_student',
                   review_seconds=time.monotonic()-st.session_state.started,created_at=time.time())
        (output/(identity+'.json')).write_text(json.dumps(audit,indent=2))
        st.success('Saved reviewed label and provenance. Original manifest is unchanged.')
except Exception as exc: st.error(str(exc))
