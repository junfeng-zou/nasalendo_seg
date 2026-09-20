"""Input/weight integrity and raw-to-cache checkpoint replay for RGB+mask."""
import train_mixed_rgbmask_distance as x
from train_mixed_rgbmask_distance import torch,np,cv2,sha,read,dump,ROOT

def main():
    x.verify_inputs();torch.set_num_threads(4);cv2.setNumThreads(1)
    rows=read(x.DATA/'manifest.json')['records'];masks={r['sample_id']:r for r in read(x.MASKDATA/'manifest.json')['records']}
    for r in rows:
        assert sha(r['image_path'])==r['image_sha256'];m=masks[r['sample_id']];assert sha(m['mask_path'])==m['mask_sha256']
    ck=torch.load(x.OUT/'best.pt',map_location='cpu',weights_only=True);model=x.make_model(False).cuda().eval();model.load_state_dict(ck['model_state']);preds={r['sample_id']:r for r in x.csv.DictReader((x.OUT/'predictions.csv').open(encoding='utf-8-sig'))};checks=[]
    for domain in ['phantom','clinical']:
        for label in x.LABELS:
            r=next(r for r in rows if r['domain']==domain and r['split']=='val' and r['label']==label)
            raw=x.letterbox(cv2.imread(r['image_path']),384);assert np.array_equal(raw,cv2.imread(r['input_path']))
            mask=cv2.imread(masks[r['sample_id']]['mask_path'],0);assert set(np.unique(mask)).issubset({0,255})
            inp=torch.cat([x.image_tensor(raw),torch.tensor(mask.astype(np.float32)/255)[None]])[None].cuda()
            with torch.inference_mode():p=model(inp).softmax(-1)[0].cpu().numpy()
            expected=preds[r['sample_id']];delta=float(np.max(np.abs(p-np.array([float(expected['p_'+k]) for k in x.LABELS]))));assert delta<2e-4,delta
            assert x.LABELS[int(p.argmax())]==expected['prediction'];checks.append(dict(sample_id=r['sample_id'],max_probability_difference=delta))
    clinical=[r for r in rows if r['domain']=='clinical' and r['split']=='val'];labels=[];zero_preds=[]
    with torch.inference_mode():
        for inp,target,_ in x.DataLoader(x.CachedImages(clinical),batch_size=16):
            inp[:,3].zero_();p=model(inp.cuda()).argmax(-1).cpu().tolist();zero_preds.extend(p);labels.extend(target.tolist())
    cm=x.confusion_matrix(labels,zero_preds,3);metric=x.classification_metrics(cm,x.LABELS)
    dump(x.OUT/'integrity_checks.json',dict(original_RGB_checkpoint_unchanged=True,source_images_unchanged=True,split_unchanged=True,segmentation_weights_unchanged=True,checks=checks,trained_model_with_mask_zeroed_clinical_diagnostic=metric,note='Zeroed-mask inference is an input ablation on the trained four-channel model, not the separately trained RGB baseline.'))
    print('PASS',len(checks),'raw/cache/checkpoint replays; zero-mask clinical accuracy',metric['accuracy'])
if __name__=='__main__':main()
