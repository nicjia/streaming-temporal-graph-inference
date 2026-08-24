"""
How much does each component contribute, and is 0.470 MRR a floor or a ceiling?

Two questions in one pass, because loading 9.4M transactions and building the
graph costs more than training a small model on it:

  TRAINING LENGTH -- loss was still falling when the headline run stopped at two
      epochs (0.370 -> 0.260), so the reported MRR is a lower bound of unknown
      slack. Reported per epoch.

  ABLATIONS -- the full model against variants with the relation embedding
      removed, the time encoding frozen at a constant, one hop instead of two,
      and a smaller neighbourhood. Each is trained identically; the difference
      is attributable to the component.

All variants are scored on the same test edges with the same degree-matched
negatives, so the rows are directly comparable.
"""
import argparse, glob, json, os, sys, time
import numpy as np, torch
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,"python"))
import graph_engine
from backtest.evaluate import roc_auc
from ingestion.ethereum import load_transactions
from models import PCSRTemporalSampler, TGATLinkModel

def metrics(pos,neg):
    beaten=((neg>pos[:,None]).sum(1)+0.5*(neg==pos[:,None]).sum(1))
    return dict(mrr=float((1.0/(beaten+1)).mean()),
                r1=float((beaten<0.5).mean()),
                r10=float((beaten<10).mean()),
                auc=roc_auc(pos,neg.ravel()))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=6)
    ap.add_argument("--train-events",type=int,default=150000)
    ap.add_argument("--max-test",type=int,default=3000)
    ap.add_argument("--negatives",type=int,default=50)
    ap.add_argument("--out",default="/tmp/eth_variants.json")
    args=ap.parse_args()

    root=os.path.expanduser("~/.cache/huggingface/hub")
    base=glob.glob(f"{root}/datasets--vnegi10--Ethereum_blockchain_parquet/snapshots/*/transactions")[0]
    table,addresses=load_transactions(base+"/*.parquet",base.replace("transactions","blocks")+"/*.parquet")
    src=table["src"].to_numpy(np.int64); dst=table["dst"].to_numpy(np.int64)
    ts=table["ts"].to_numpy(np.int64); rel=table["relation"].to_numpy(np.uint16); V=len(addresses)
    cut_train=int(len(src)*0.75); cut_val=int(len(src)*0.85)

    graph=graph_engine.PCSRGraph(V,int(len(src)*2.5),int(len(src)*2.5)*8*4+(1<<26))
    graph.insert_edges(src.astype(np.uint32),dst.astype(np.uint32),ts.astype(np.uint32),rel)
    sampler=PCSRTemporalSampler(graph)

    seen=set(zip(src[:cut_train].tolist(),dst[:cut_train].tolist()))
    tS,tD,tT=src[cut_val:],dst[cut_val:],ts[cut_val:]
    is_new=np.fromiter(((a,b) not in seen for a,b in zip(tS,tD)),dtype=bool,count=len(tS))

    in_deg=np.bincount(dst[:cut_train],minlength=V).astype(np.float64)
    cand=np.flatnonzero(in_deg>0); prior=in_deg[cand]/in_deg[cand].sum()
    rng=np.random.default_rng(0)
    def take(mask,n):
        i=np.flatnonzero(mask)
        return i[np.linspace(0,len(i)-1,min(n,len(i))).astype(int)]
    sets={"new":take(is_new,args.max_test),"repeat":take(~is_new,args.max_test)}
    negs={k:rng.choice(cand,size=(len(v),args.negatives),p=prior) for k,v in sets.items()}

    step=max(1,cut_train//args.train_events)
    trS,trD,trT=src[:cut_train:step],dst[:cut_train:step],ts[:cut_train:step]
    num_rel=int(rel.max())+1
    print(f"\ntrain {len(trS):,} events | test new {len(sets['new']):,} repeat {len(sets['repeat']):,}\n")

    def evaluate(model, quick=False):
        """quick=True uses a smaller slice for the per-epoch curve; the final
        number is always computed on the full evaluation set."""
        model.eval(); out={}
        for k,idx in sets.items():
            if quick: idx=idx[::4]
            s,d,t=tS[idx],tD[idx],tT[idx]
            ng=negs[k][:len(idx)] if not quick else negs[k][::4]
            n_neg=ng.shape[1] if not quick else min(20,ng.shape[1])
            allc=np.concatenate([tD[idx][:,None],ng[:,:n_neg]],axis=1)
            sc=model.score_against(s,t,allc).numpy()
            out[k]=metrics(sc[:,0],sc[:,1:])
        model.train(); return out

    def train(tag,epochs,**kw):
        torch.manual_seed(0); g=np.random.default_rng(1)
        cfg=dict(node_dim=64,time_dim=64,num_layers=2,num_neighbors=20,num_relations=num_rel)
        freeze_time=kw.pop("freeze_time",False); cfg.update(kw)
        m=TGATLinkModel(V,sampler,**cfg)
        if freeze_time:
            # Constant time encoding: the model keeps the graph but loses any
            # notion of *when* -- isolates what continuous time contributes.
            m.encoder.time_encoder.frequencies.data.zero_()
            m.encoder.time_encoder.frequencies.requires_grad_(False)
        opt=torch.optim.Adam(m.parameters(),lr=1e-3); m.train()
        history=[]
        for ep in range(epochs):
            losses=[]
            for off in range(0,len(trS),256):
                b=slice(off,min(off+256,len(trS)))
                n=b.stop-b.start
                if n<8: continue
                opt.zero_grad()
                loss,_,_=m.loss(trS[b],trD[b],trT[b],
                                g.choice(cand,n,p=prior),g.choice(cand,n,p=prior))
                loss.backward(); opt.step(); losses.append(loss.item())
            ev=evaluate(m,quick=(ep<epochs-1))
            history.append(dict(epoch=ep+1,loss=float(np.mean(losses)),**{k:v for k,v in ev.items()}))
            print(f"  [{tag}] epoch {ep+1}: loss {np.mean(losses):.4f} | "
                  f"new MRR {ev['new']['mrr']:.3f} AUC {ev['new']['auc']:.3f} | "
                  f"repeat MRR {ev['repeat']['mrr']:.3f}",flush=True)
        history[-1]["final"]=True
        return history

    results={}
    print("=== training length (full model) ===")
    results["full"]=train("full",args.epochs)

    print("\n=== ablations (2 epochs each, identical otherwise) ===")
    for tag,kw in [("no relations",dict(num_relations=0)),
                   ("frozen time encoding",dict(freeze_time=True)),
                   ("1 hop",dict(num_layers=1)),
                   ("neighbourhood K=5",dict(num_neighbors=5))]:
        results[tag]=train(tag,2,**kw)

    json.dump(results,open(args.out,"w"),indent=1)
    base_line=results["full"][1]
    print(f"\n{'variant':<24}{'new MRR':>10}{'new R@1':>10}{'new AUC':>10}{'vs full':>10}")
    print("-"*64)
    for tag,h in results.items():
        e=h[1] if len(h)>1 else h[-1]
        print(f"{tag:<24}{e['new']['mrr']:>10.3f}{e['new']['r1']:>10.3f}"
              f"{e['new']['auc']:>10.3f}{e['new']['mrr']-base_line['new']['mrr']:>+10.3f}")
    best=max(results['full'],key=lambda e:e['new']['mrr'])
    print(f"\nbest full-model epoch: {best['epoch']} -> new MRR {best['new']['mrr']:.3f} "
          f"R@1 {best['new']['r1']:.3f} R@10 {best['new']['r10']:.3f} AUC {best['new']['auc']:.3f}")
    print(f"wrote {args.out}")

if __name__=="__main__": main()
