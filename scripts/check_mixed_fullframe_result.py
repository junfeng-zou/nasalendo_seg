"""Verify checkpoint/raw-image inference against final predictions and plot learning curves."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
import timm
from distance_state_classifier.src.manual_head_abc import letterbox,image_tensor
from train_mixed_fullframe_distance import OUT,DATA,read,dump,sha,LABELS

@torch.inference_mode()
def main():
    assert (OUT/'completion.json').exists(),'Wait for completed training'
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    saved=torch.load(OUT/'best.pt',map_location='cpu',weights_only=True)
    model=timm.create_model('convnext_tiny',num_classes=3,drop_rate=saved['config']['dropout'],pretrained=False).cuda().eval();model.load_state_dict(saved['model_state'],strict=True)
    predictions=list(csv.DictReader((OUT/'predictions.csv').open(encoding='utf-8-sig')))
    checked=[]
    for domain in ['clinical','phantom']:
        for label in LABELS:
            r=next(r for r in predictions if r['domain']==domain and r['evaluation'].endswith('_val') and r['label']==label)
            image=letterbox(cv2.imread(r['image_path']),384);assert np.array_equal(image,cv2.imread(r['input_path']))
            probs=model(image_tensor(image)[None].cuda()).softmax(-1)[0].cpu().numpy();expected=np.array([float(r['p_'+k]) for k in LABELS])
            error=float(np.max(np.abs(probs-expected)));assert error<1e-3,(r['sample_id'],error)
            assert LABELS[int(probs.argmax())]==r['prediction']
            checked.append(dict(sample_id=r['sample_id'],max_probability_error=error))
    complete=read(OUT/'completion.json');assert sha(OUT/'best.pt')==complete['checkpoint_sha256'];assert sha(DATA/'manifest.json')==complete['split_sha256']
    dump(OUT/'inference_checks.json',dict(cases=checked,raw_image_matches_cached_input=True,weights_and_split_unchanged=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history=read(OUT/'history.json');epochs=[r['epoch'] for r in history]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].plot(epochs,[r['train_loss'] for r in history]);axes[0].set(xlabel='Epoch',ylabel='Training loss',title='Weighted cross entropy')
    for source in ['phantom','clinical']:
        axes[1].plot(epochs,[r['validation'][source+'_val']['macro_f1'] for r in history],label=source)
    axes[1].plot(epochs,[r['selection_score'] for r in history],label='Equal-domain mean',linestyle='--')
    axes[1].axvline(complete['best_epoch'],color='gray',linestyle=':');axes[1].set(xlabel='Epoch',ylabel='Validation Macro-F1',title='Fixed validation splits');axes[1].legend()
    fig.savefig(OUT/'training_curves.png',dpi=160);plt.close(fig)
    results=read(OUT/'results.json');fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,name in zip(axes,['clinical_val','phantom_val','phantom_test']):
        cm=np.array(results[name]['confusion_matrix']);ax.imshow(cm,cmap='Blues');ax.set(xticks=range(3),yticks=range(3),xticklabels=LABELS,yticklabels=LABELS,title=name,xlabel='Predicted',ylabel='True')
        for i in range(3):
            for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.savefig(OUT/'confusion_matrices.png',dpi=160);plt.close(fig)
    print(json.dumps(checked,indent=2))
if __name__=='__main__':main()
