#!/usr/bin/env python3
"""Decompose reconstruction error vs depth-camera count on the dimpled sphere."""
import argparse, math, itertools
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def fib(n):
    phi = math.pi*(math.sqrt(5)-1); o=[]
    for i in range(n):
        y=1-(i/(n-1))*2; r=math.sqrt(max(0,1-y*y)); t=phi*i
        o.append((math.cos(t)*r, y, math.sin(t)*r))
    v=np.array(o); return v/np.linalg.norm(v,axis=1,keepdims=True)


def load(rec):
    V=np.fromfile(Path(rec)/"vertices.bin",np.float32).reshape(-1,3)
    F=np.fromfile(Path(rec)/"faces.bin",np.int64).reshape(-1,3)
    return trimesh.Trimesh(V,F,process=False)


def ang_to_axes(p, axes):
    d=p/np.clip(np.linalg.norm(p,axis=1,keepdims=True),1e-9,None)
    return np.degrees(np.arccos(np.clip((d@axes.T).max(1),-1,1)))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dir",required=True); ap.add_argument("--gt",required=True)
    ap.add_argument("--counts",default="1 2 4 8 12"); ap.add_argument("--pockets",type=int,default=12)
    ap.add_argument("--samples",type=int,default=150000)
    a=ap.parse_args()
    axes=fib(a.pockets); Ns=[int(x) for x in a.counts.split()]
    gt=trimesh.load(a.gt,process=False); gp=trimesh.sample.sample_surface(gt,a.samples,seed=0)[0]
    gp_ang=ang_to_axes(gp,axes)
    rows=[]
    for N in Ns:
        recon=load(Path(a.dir)/f"recon_d{N}/frame_00000")
        rp=trimesh.sample.sample_surface(recon,a.samples,seed=0)[0]
        best=None
        for s in itertools.product((1,-1),repeat=3):
            S=np.array(s,float); c=cKDTree(rp*S).query(gp)[0].mean()+cKDTree(gp).query(rp*S)[0].mean()
            if best is None or c<best[0]: best=(c,S)
        rpf=rp*best[1]; rp_ang=ang_to_axes(rpf,axes)
        acc=cKDTree(gp).query(rpf)[0]*1000; comp=cKDTree(rpf).query(gp)[0]*1000
        pk_g=gp_ang<18; pk_r=rp_ang<18
        rows.append(dict(N=N,
            glob=acc.mean()+comp.mean(), acc=acc.mean(), comp=comp.mean(),
            pk_comp=comp[pk_g].mean(), pk_acc=acc[pk_r].mean(),
            deep=(np.linalg.norm(rpf,axis=1)<0.45).mean()*100))
    gt_deep=(np.linalg.norm(gp,axis=1)<0.45).mean()*100
    N=[r["N"] for r in rows]
    fig,ax=plt.subplots(2,2,figsize=(12,9))
    ax[0,0].plot(N,[r["glob"] for r in rows],"o-",color="firebrick",lw=2); ax[0,0].set_title("Global Chamfer (headline)"); ax[0,0].set_ylabel("mm")
    ax[0,1].plot(N,[r["acc"] for r in rows],"o-",label="accuracy (recon→GT)",color="darkorange")
    ax[0,1].plot(N,[r["comp"] for r in rows],"s-",label="completeness (GT→recon)",color="seagreen")
    ax[0,1].legend(); ax[0,1].set_title("Chamfer decomposed"); ax[0,1].set_ylabel("mm")
    ax[1,0].plot(N,[r["pk_comp"] for r in rows],"s-",color="seagreen",lw=2)
    ax[1,0].set_title("Pocket-region completeness  (depth's real job ↓=better)"); ax[1,0].set_ylabel("mm")
    ax[1,1].plot(N,[r["deep"] for r in rows],"o-",color="navy",lw=2,label="recon deep-fraction")
    ax[1,1].axhline(gt_deep,ls="--",color="gray",label=f"GT = {gt_deep:.1f}%")
    ax[1,1].legend(); ax[1,1].set_title("Pocket volume recovered (% pts r<0.45)"); ax[1,1].set_ylabel("%")
    for x in ax.flat: x.set_xlabel("number of depth cameras"); x.grid(alpha=.3)
    fig.suptitle("Approach A (TSDF zero-crossing + soft veto): error vs depth-camera count",fontsize=13)
    fig.tight_layout()
    out=Path(a.dir)/"eval_panels.png"; fig.savefig(out,dpi=130)
    import csv
    with open(Path(a.dir)/"eval_metrics.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    print("saved",out)
    print(f"{'N':>3} {'global':>7} {'acc':>7} {'comp':>7} {'pk_comp':>8} {'pk_acc':>7} {'deep%':>6}")
    for r in rows: print(f"{r['N']:>3} {r['glob']:7.2f} {r['acc']:7.2f} {r['comp']:7.2f} {r['pk_comp']:8.2f} {r['pk_acc']:7.2f} {r['deep']:6.1f}")
    print(f"GT deep% = {gt_deep:.1f}")


if __name__=="__main__":
    main()
