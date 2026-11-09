import pandas as pd, numpy as np
df=pd.read_pickle('/tmp/ipco_work/all.pkl')
roots={'ipco_full_pmedian_20261105_193605':['STANDARD_BD','RANDOM'],'ipco_cmp_pmedian_v0_20261106_171133':['PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN_V0'],
       'ipco_full100_pmedian_v0_20261107_211627':['STANDARD_BD','RANDOM','PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN_V0'],
       'ipco_full300_pmedian_v0_20261107_211634':['STANDARD_BD','RANDOM','PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN_V0']}
k=pd.concat([df[(df.root==r)&df.method.isin(m)] for r,m in roots.items()])
assert (k.n_logs>0).all(), k[k.n_logs==0]
k['M']=k.method.map({'STANDARD_BD':'BD','RANDOM':'RND','PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN_V0':'LABD'})
k['inst']=k.instance.str.split('_').str[0]; k['S']=k.instance.str.split('_S').str[1].astype(int)
bd=k[k.M=='BD'].set_index(['inst','S'])
r=k[k.M!='BD'].copy()
idx=list(zip(r.inst,r.S))
r['VBD']=[bd.loc[i,'original_Prob_Obj_Value'] for i in idx]; r['TBD']=[bd.loc[i,'solution_time_sec'] for i in idx]
r['solved']=[bd.loc[i,'stopping_reason']=='OPTIMAL' for i in idx]
r['V']=r.original_Prob_Obj_Value; r['Delta']=100*(r.V-r.VBD)/r.VBD.abs(); r['tau']=r.solution_time_sec/r.TBD
r['cert']=100*r.verified_final_relative_gap; r['share']=(r.ratio.astype(float)*100).round().astype(int)
r.to_pickle('/tmp/ipco_work/metrics_all.pkl'); k.to_pickle('/tmp/ipco_work/main_all.pkl')
print(bd[['stopping_reason','solution_time_sec','original_Prob_Obj_Value','verified_final_relative_gap']].round(3))
print(r.groupby(['M','S','stopping_reason']).size())
