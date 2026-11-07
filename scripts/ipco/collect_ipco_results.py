import glob,os,pandas as pd,sys
base=sys.argv[1]; rows=[]
for man in sorted(glob.glob(f'{base}/MASTER_RUNS/ipco_*/manifest.tsv')):
    root=os.path.dirname(man); stamp=os.path.basename(root)
    m=pd.read_csv(man,sep='\t',dtype=str)
    for _,t in m.iterrows():
        ds=sorted(glob.glob(f'{base}/SCFLP_class/RESULTS/labd_ipco_{stamp}/task_{t.task_id}_*/solution_log.xlsx'))
        rec=dict(root=stamp,task=int(t.task_id),instance=t.instance_label,method=t.method,ratio=t.selected_ratio,n_logs=len(ds))
        if ds:
            d=pd.read_excel(ds[-1])
            if len(d): rec.update(d.iloc[0].to_dict())
        rows.append(rec)
df=pd.DataFrame(rows); df.to_pickle('/tmp/ipco_work/all.pkl')
print(df[['root','task','instance','method','ratio','n_logs']].groupby(['root','method']).agg(n=('task','count'),logs=('n_logs','sum')))
