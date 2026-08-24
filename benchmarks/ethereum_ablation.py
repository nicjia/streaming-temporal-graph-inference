"""
What is the TGAT actually using to predict new links?

A single strong number invites the question "what drives it". These breakdowns
answer it without retraining, by reusing the checkpoint and slicing the same
test set:

  by history length -- does it need a long trace, or does it work cold?
  by transaction value -- does it hold on economically significant transfers,
      or only on dust?
  by relation type    -- is it a token-transfer effect or a general one?
"""
import argparse, glob, os, sys, time
import numpy as np, torch
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,"python"))
import graph_engine
from backtest.evaluate import roc_auc
from ingestion.ethereum import load_transactions, RELATION_NAMES
from models import PCSRTemporalSampler, TGATLinkModel

def metrics(pos, neg):
    beaten=((neg>pos[:,None]).sum(1)+0.5*(neg==pos[:,None]).sum(1))
    return float((1.0/(beaten+1)).mean()), float((beaten<0.5).mean()), roc_auc(pos,neg.ravel())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",default="/tmp/eth_tgat.pt")
    ap.add_argument("--negatives",type=int,default=50)
    ap.add_argument("--max-test",type=int,default=6000)
    args=ap.parse_args()

    root=os.path.expanduser("~/.cache/huggingface/hub")
    base=glob.glob(f"{root}/datasets--vnegi10--Ethereum_blockchain_parquet/snapshots/*/transactions")[0]
    table,addresses=load_transactions(base+"/*.parquet", base.replace("transactions","blocks")+"/*.parquet", quiet=True)
    src=table["src"].to_numpy(np.int64); dst=table["dst"].to_numpy(np.int64)
    ts=table["ts"].to_numpy(np.int64); rel=table["relation"].to_numpy(np.uint16)
    val=table["value"].to_numpy(); V=len(addresses)

    cut_train=int(len(src)*0.75); cut_val=int(len(src)*0.85)
    graph=graph_engine.PCSRGraph(V,int(len(src)*2.5),int(len(src)*2.5)*8*4+(1<<26))
    graph.insert_edges(src.astype(np.uint32),dst.astype(np.uint32),ts.astype(np.uint32),rel)
    sampler=PCSRTemporalSampler(graph)

    seen=set(zip(src[:cut_train].tolist(),dst[:cut_train].tolist()))
    tS,tD,tT,tR,tV=src[cut_val:],dst[cut_val:],ts[cut_val:],rel[cut_val:],val[cut_val:]
    is_new=np.fromiter(((a,b) not in seen for a,b in zip(tS,tD)),dtype=bool,count=len(tS))
    idx=np.flatnonzero(is_new)
    idx=idx[np.linspace(0,len(idx)-1,min(args.max_test,len(idx))).astype(int)]

    in_deg=np.bincount(dst[:cut_train],minlength=V).astype(np.float64)
    cand=np.flatnonzero(in_deg>0); prior=in_deg[cand]/in_deg[cand].sum()
    rng=np.random.default_rng(0)
    neg=rng.choice(cand,size=(len(idx),args.negatives),p=prior)

    model=TGATLinkModel(V,sampler,node_dim=64,time_dim=64,num_layers=2,
                        num_neighbors=20,num_relations=int(rel.max())+1)
    model.load_state_dict(torch.load(args.checkpoint)); model.eval()

    s,d,t = tS[idx],tD[idx],tT[idx]
    with torch.no_grad():
        pos=model.score(s,d,t).numpy()
        ng=np.stack([model.score(s,neg[:,k],t).numpy() for k in range(args.negatives)],1)

    # source-address history length at query time, straight from the engine
    hist=sampler.recency_features(s,t,20)[:,0]*5.0     # undo the /5 scaling -> log1p(count)
    counts=np.expm1(hist)

    def show(title, buckets):
        print(f"\n{title}")
        print(f"{'bucket':<26}{'n':>7}{'MRR':>8}{'R@1':>8}{'AUC':>8}")
        print("-"*57)
        for name,mask in buckets:
            if mask.sum()<50: continue
            m=metrics(pos[mask],ng[mask])
            print(f"{name:<26}{int(mask.sum()):>7}{m[0]:>8.3f}{m[1]:>8.3f}{m[2]:>8.3f}")

    q=np.quantile(counts,[.25,.5,.75])
    show("By source-address history length (prior transactions)",[
        (f"cold  (<{q[0]:.0f})",counts<q[0]),
        (f"low   ({q[0]:.0f}-{q[1]:.0f})",(counts>=q[0])&(counts<q[1])),
        (f"mid   ({q[1]:.0f}-{q[2]:.0f})",(counts>=q[1])&(counts<q[2])),
        (f"heavy (>={q[2]:.0f})",counts>=q[2])])

    v=tV[idx]; nz=v[v>0]
    show("By transaction value (ETH, wei/1e18)",[
        ("zero value (token/approve)",v==0),
        ("dust  (<0.01 ETH)",(v>0)&(v<1e16)),
        ("small (0.01-1 ETH)",(v>=1e16)&(v<1e18)),
        ("large (>=1 ETH)",v>=1e18)])

    r=tR[idx]
    show("By relation type",[(RELATION_NAMES.get(int(k),str(k)),r==k) for k in np.unique(r)])

    overall=metrics(pos,ng)
    print(f"\noverall new-link: MRR {overall[0]:.3f}  R@1 {overall[1]:.3f}  AUC {overall[2]:.3f}"
          f"   ({len(idx):,} edges)")

if __name__=="__main__": main()
