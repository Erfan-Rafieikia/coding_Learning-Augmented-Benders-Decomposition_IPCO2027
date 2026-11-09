import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
r=pd.read_pickle('/tmp/ipco_work/metrics_all.pkl'); r=r[r.solved]
plt.rcParams.update({'font.size':7,'font.family':'serif','axes.linewidth':0.5,'pdf.fonttype':42,'mathtext.fontset':'cm'})
C={'LABD':'#2a78d6','RND':'#eb6834'}; MK={'LABD':'o','RND':'s'}
fig,axs=plt.subplots(1,2,figsize=(4.8,1.5))
shares=[5,15,25]; off={'LABD':-0.18,'RND':0.18}; rng=np.random.default_rng(0)
for ax,(col,lab) in zip(axs,[('Delta',r'$\Delta_a$ (%)'),('tau',r'$\tau_a$')]):
    for m in ['LABD','RND']:
        data=[(r[(r.M==m)&(r.share==s)][col].clip(lower=0) if col=='Delta' else r[(r.M==m)&(r.share==s)][col]).values for s in shares]
        pos=[i+off[m] for i in range(3)]
        ax.boxplot(data,positions=pos,widths=0.28,patch_artist=True,showfliers=False,whis=(0,100),
            medianprops=dict(color='#1f1f1d',lw=1.0),boxprops=dict(facecolor='white',edgecolor=C[m],lw=0.9),
            whiskerprops=dict(color=C[m],lw=0.7),capprops=dict(color=C[m],lw=0.7))
        for p,d in zip(pos,data):
            ax.scatter(p+rng.uniform(-0.06,0.06,len(d)),d,s=7,color=C[m],marker=MK[m],edgecolors='white',linewidths=0.3,zorder=3)
    ax.set_xticks(range(3)); ax.set_xticklabels(['5%','15%','25%']); ax.set_xlim(-0.6,2.6)
    ax.set_xlabel(r'exact-subproblem share $|\mathcal{M}_e|/|\mathcal{M}|$',labelpad=1)
    ax.set_title(lab+' by exact-subproblem share',fontsize=7,pad=3)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    ax.tick_params(width=0.4,length=2,pad=1.5); ax.grid(axis='y',lw=0.3,color='#e5e5e2'); ax.set_axisbelow(True)
axs[1].axhline(1,color='#8a8a85',lw=0.6,ls=(0,(3,2)),zorder=1)
axs[0].set_ylim(-0.08,1.7)
h=[Line2D([],[],color=C[m],marker=MK[m],lw=0.9,ms=3,label=m) for m in ['LABD','RND']]
fig.legend(handles=h,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(0.5,-0.02),handlelength=1.6,columnspacing=1.5)
fig.subplots_adjust(left=0.07,right=0.99,bottom=0.33,top=0.88,wspace=0.22)
fig.savefig('/tmp/ipco_work/fig_share.pdf'); fig.savefig('/tmp/ipco_work/fig_share.png',dpi=250)
