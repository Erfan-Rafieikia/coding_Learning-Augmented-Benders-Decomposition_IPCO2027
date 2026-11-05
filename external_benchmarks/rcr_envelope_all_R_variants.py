"""Envelope routability (relatively complete recourse) for R networks with Rahmaniani
scenarios and a Normal(mu, 0.1 mu) Kothari-Luedtke-style comparison.
Usage: python rcr_envelope_all_R_variants.py r04 r05 ...  (prints lambda*; >=1 means RCR holds)
"""
import numpy as np, scipy.sparse as sp, sys, glob, os
from scipy.optimize import linprog
HERE=os.path.dirname(os.path.abspath(__file__)); RD=os.path.join(HERE,"sources","r_networks_crainic"); SD=os.path.join(HERE,"sources","r_scenarios_rahmaniani")
def readdow(f):
    L=[l for l in open(f).read().split("\n")]
    m,n,K=map(int,L[1].split()[:3]); arcs=[];u=[]
    for l in L[2:2+n]:
        a=l.split(); arcs.append((int(a[0])-1,int(a[1])-1)); u.append(float(a[3]))
    com=[]
    for l in L[2+n:2+n+K]:
        a=l.split(); com.append((int(a[0])-1,int(a[1])-1,float(a[2])))
    return m,arcs,np.array(u),com
def lam(m,arcs,u,com,vol):
    # max lambda s.t. route lambda*vol_q from o_q to d_q, sum_q y_aq <= u_a
    n=len(arcs);K=len(com);nv=n*K+1
    rows=[];cols=[];vals=[];beq=[]
    r=0
    for q,(o,d,_) in enumerate(com):
        for i in range(m):
            for a,(s,t) in enumerate(arcs):
                if s==i: rows.append(r);cols.append(q*n+a);vals.append(1)
                if t==i: rows.append(r);cols.append(q*n+a);vals.append(-1)
            rhs = vol[q] if i==o else (-vol[q] if i==d else 0)
            rows.append(r);cols.append(nv-1);vals.append(-rhs); beq.append(0); r+=1
    Aeq=sp.csr_matrix((vals,(rows,cols)),shape=(r,nv))
    rr=[];cc=[];vv=[]
    for a in range(n):
        for q in range(K): rr.append(a);cc.append(q*n+a);vv.append(1)
    Aub=sp.csr_matrix((vv,(rr,cc)),shape=(n,nv))
    c=np.zeros(nv);c[-1]=-1
    res=linprog(c,A_ub=Aub,b_ub=u,A_eq=Aeq,b_eq=np.zeros(r),bounds=[(0,None)]*nv,method="highs")
    return -res.fun if res.status==0 else float('nan')
out=[]
for cls in sys.argv[1:]:
    for v in range(1,10):
        f=f"{RD}/{cls}.{v}.dow"
        if not os.path.exists(f): continue
        m,arcs,u,com=readdow(f); nom=np.array([c[2] for c in com])
        row=[f"{cls}.{v}", "%.2f"%lam(m,arcs,u,com,nom)]
        for corr in ["0","0.8"]:
            sf=f"{SD}/{cls}-{corr}-1000"
            if os.path.exists(sf):
                S=np.loadtxt(sf,skiprows=1)[:,1:]
                for N in (100,500):
                    row.append("%.2f"%lam(m,arcs,u,com,S[:N].max(0)))
            else: row+=["-","-"]
        rng=np.random.default_rng(1)
        Kn=np.maximum(0,rng.normal(nom,0.1*nom,size=(500,len(nom)))).max(0)
        row.append("%.2f"%lam(m,arcs,u,com,Kn))
        print("\t".join(row),flush=True)
