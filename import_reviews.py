"""Create a new training round from reviewed labels without changing test/validation splits."""
import argparse
import json
from pathlib import Path
from student import rows
from review_integrity import validate_review

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('manifest'); p.add_argument('--reviews',default='annotations'); p.add_argument('--output',required=True); a=p.parse_args()
    output=Path(a.output)
    if output.exists(): p.error('Output exists; choose a new round manifest')
    records=rows(a.manifest); mapping={r['id']:r for r in records}; seen=set()
    for path in sorted(Path(a.reviews).glob('*.json')):
        audit=json.loads(path.read_text()); identity=audit['id']
        if identity in seen: raise ValueError('Multiple reviews for one image; choose a single reviewed record')
        seen.add(identity); record=mapping[identity]
        if record['split']!='pool': raise ValueError('Only pool samples can enter the training round')
        validate_review(audit,record['image'])
        record.update(split='train',mask=str(Path(audit['mask']).resolve()),provenance=audit['source'],
                      review_seconds=audit['review_seconds'],mask_sha256=audit['mask_sha256'],
                      checkpoint_sha256=audit['checkpoint_sha256'])
    if not seen: raise ValueError('No reviews to import')
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(records,indent=2))
    print(f'Created new training round with {len(seen)} reviewed labels')
if __name__=='__main__': main()
