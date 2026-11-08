import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from wasteops.taxonomy import CLASSES

FOLDER_TO_CLASS={"Cardboard":"cardboard","Food Organics":"food_organics","Glass":"glass","Metal":"metal","Miscellaneous Trash":"miscellaneous_trash","Paper":"paper","Plastic":"plastic","Textile Trash":"textile_trash","Vegetation":"vegetation"}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',default='data/RealWaste');ap.add_argument('--epochs',type=int,default=12);ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--lr',type=float,default=2e-4);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    root=Path(args.data_dir)
    if not root.exists():
        # Support the UCI archive's nested folder layout.
        matches=[p for p in Path('data').rglob('Cardboard') if p.is_dir()]
        if matches: root=matches[0].parent
        else: raise FileNotFoundError("Dataset not found. Run python scripts/download_realwaste.py first.")
    train_tf=transforms.Compose([transforms.Resize((232,232)),transforms.RandomResizedCrop(224,scale=(0.75,1.0)),transforms.RandomHorizontalFlip(),transforms.ColorJitter(0.15,0.15,0.15,0.05),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    eval_tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    ds=datasets.ImageFolder(root,transform=None)
    mapping={name:FOLDER_TO_CLASS.get(name) for name in ds.classes}
    if any(v is None for v in mapping.values()) or set(mapping.values())!=set(CLASSES): raise ValueError(f"Unexpected dataset classes: {ds.classes}")
    # Reorder ImageFolder targets to the stable API taxonomy.
    targets=np.array([ds.class_to_idx[next(k for k,v in mapping.items() if v==CLASSES[t])] for t in range(len(CLASSES))])
    y=np.array([CLASSES.index(mapping[ds.classes[t]]) for t in ds.targets])
    idx=np.arange(len(ds)); train_idx,tmp_idx=train_test_split(idx,test_size=.2,random_state=args.seed,stratify=y)
    val_idx,test_idx=train_test_split(tmp_idx,test_size=.5,random_state=args.seed,stratify=y[tmp_idx])
    class DatasetView(torch.utils.data.Dataset):
        def __init__(self,base,indices,tf):self.base=base;self.indices=indices;self.tf=tf
        def __len__(self):return len(self.indices)
        def __getitem__(self,i):
            path,_=self.base.samples[int(self.indices[i])]
            from PIL import Image
            image=Image.open(path).convert('RGB'); label=int(y[int(self.indices[i])]); return self.tf(image),label
    device=torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    train_loader=DataLoader(DatasetView(ds,train_idx,train_tf),batch_size=args.batch_size,shuffle=True,num_workers=args.workers)
    val_loader=DataLoader(DatasetView(ds,val_idx,eval_tf),batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
    test_loader=DataLoader(DatasetView(ds,test_idx,eval_tf),batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
    model=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3]=nn.Linear(model.classifier[3].in_features,len(CLASSES));model.to(device)
    weights=np.bincount(y[train_idx],minlength=len(CLASSES)); class_weights=torch.tensor(len(train_idx)/(len(CLASSES)*np.maximum(weights,1)),dtype=torch.float32,device=device)
    criterion=nn.CrossEntropyLoss(weight=class_weights); optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4); best=-1; Path('models').mkdir(exist_ok=True);Path('reports').mkdir(exist_ok=True)
    for epoch in range(args.epochs):
        model.train();loss_sum=0
        for x,t in train_loader:
            x,t=x.to(device),t.to(device);optimizer.zero_grad(set_to_none=True);loss=criterion(model(x),t);loss.backward();optimizer.step();loss_sum+=loss.item()*len(t)
        model.eval();preds=[];truth=[]
        with torch.inference_mode():
            for x,t in val_loader:
                p=model(x.to(device)).argmax(1).cpu().tolist();preds.extend(p);truth.extend(t.tolist())
        f1=f1_score(truth,preds,average='macro');print(f"epoch={epoch+1}/{args.epochs} train_loss={loss_sum/len(train_idx):.4f} val_macro_f1={f1:.4f}")
        if f1>best:
            best=f1;torch.save({"state_dict":model.cpu().state_dict(),"classes":CLASSES,"validation_macro_f1":best,"seed":args.seed},'models/wasteops_mobilenetv3.pt');model.to(device)
    payload=torch.load('models/wasteops_mobilenetv3.pt',map_location=device,weights_only=True);model.load_state_dict(payload['state_dict']);model.eval();preds=[];truth=[]
    with torch.inference_mode():
        for x,t in test_loader:preds.extend(model(x.to(device)).argmax(1).cpu().tolist());truth.extend(t.tolist())
    report=classification_report(truth,preds,target_names=CLASSES,output_dict=True,zero_division=0)
    summary={"dataset":"UCI RealWaste","split":{"train":len(train_idx),"validation":len(val_idx),"test":len(test_idx)},"test_accuracy":accuracy_score(truth,preds),"test_macro_f1":f1_score(truth,preds,average='macro'),"classification_report":report,"confusion_matrix":confusion_matrix(truth,preds).tolist(),"warning":"Image-level split; not a site-held-out generalization result."}
    Path('reports/metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in {'classification_report','confusion_matrix'}},indent=2))
if __name__=='__main__':main()
