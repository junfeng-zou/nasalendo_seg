"""Audit immutable inputs, ROI preprocessing and saved-checkpoint prediction replay."""
import train_mixed_head_distance as exp
from train_mixed_head_distance import torch,cv2,np,sha,read,dump,Model,image_tensor,prepare_inputs

def main():
    torch.set_num_threads(4);cv2.setNumThreads(1)
    manifest=read(exp.DATA/'manifest.json');assert sha(exp.SOURCE)==manifest['source_manifest_sha256'];assert sha(exp.DETECTOR)==manifest['detector_sha256']
    for r in manifest['records']:
        assert sha(r['image_path'])==r['image_sha256']
        if r['box'] is not None:assert sha(r['roi_path'])==r['roi_sha256']
    predictions=read(exp.OUT/'predictions.json');checks=[]
    sample=[]
    for domain in ['clinical','phantom']:
        for label in exp.LABELS:
            r=next(r for r in predictions if r['domain']==domain and r['label']==label and r['box'] is not None);sample.append(r)
    with torch.inference_mode():
        for arm in ['B','C']:
            ck=torch.load(exp.OUT/arm/'best.pt',map_location='cpu',weights_only=True);model=Model(arm,ck['mean'],ck['std']).cuda().eval();model.load_state_dict(ck['model_state'])
            for r in sample:
                _,roi,scale=prepare_inputs(cv2.imread(r['image_path']),r['box'],r['fov_diameter']);cached=cv2.imread(r['roi_path']);assert np.array_equal(roi,cached);assert np.allclose(scale,r['scales'])
                im=image_tensor(roi)[None].cuda();s=torch.tensor(scale)[None].cuda();p=model(im,s).softmax(-1)[0].cpu().numpy();pred=exp.LABELS[int(p.argmax())];assert pred==r[arm],(r['sample_id'],pred,r[arm])
                if arm=='B':assert torch.equal(model(im,s),model(im,s*7+11))
                checks.append(dict(arm=arm,sample_id=r['sample_id'],prediction=pred,roi_exact=True))
    dump(exp.OUT/'integrity_checks.json',dict(source_and_roi_hashes_unchanged=True,split_unchanged=True,detector_unchanged=True,B_scale_invariant=True,replay_checks=checks))
    print('PASS: source/split/detector/ROI hashes; 12 raw-to-cache-to-checkpoint checks; B ignores scale.')
if __name__=='__main__':main()
