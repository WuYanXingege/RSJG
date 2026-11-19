"""Plot completed dev curves only. Python/matplotlib; single-seed descriptive data."""
import argparse,csv,json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--audit-library',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.audit_library));from audit_panel_alignment import require_matplotlib_panel_alignment
from report import archived
import io
old=list(csv.DictReader(io.StringIO(archived('f44652730d0d7660c0eefed2cefa6bf939f1037c','zara2/joint/EPOCH_METRICS.csv'))))
with (a.results/'PER_EPOCH_GROUPS.csv').open() as f:new=list(csv.DictReader(f))
with (a.results/'OLD_HIGH_COUNT_CURVE.csv').open() as f:dense=list(csv.DictReader(f))
rank=json.loads((a.results/'FINAL_RESULT.json').read_text())['new_rank1_epoch']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.right':False,'axes.spines.top':False})
fig,axes=plt.subplots(1,2,figsize=(7.2,2.8));fig.subplots_adjust(left=.09,right=.98,bottom=.2,top=.77,wspace=.35)
for ax,group,title in zip(axes,['all','N_ge_9'],['All development windows (n=590)','High-count windows (n=127)']):
 o=[(int(r['epoch']),json.loads(r['metrics'])['JFDE']) for r in old] if group=='all' else [(int(r['epoch']),float(r['H_JFDE'])) for r in dense]
 n=[(int(r['epoch']),float(r['JFDE'])) for r in new if r['group']==group]
 for values,name,color,chosen in [(o,'Coordinate proxy','#777777',26),(n,'Candidate-cost proxy','#0072B2',rank)]:
  x,y=zip(*values);ax.plot(x,y,label=name,color=color,lw=1.1);point=dict(values)[chosen];ax.scatter([chosen],[point],color=color,marker='*',s=65,zorder=4)
 ax.set(xlabel='Completed epoch',ylabel='Development H-JFDE (m)',title=title)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
require_matplotlib_panel_alignment(fig,json_out=str(a.results/'CURVES.alignment.json'),overlay_svg=str(a.results/'CURVES.alignment.svg'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
for ext in ['pdf','svg','png']:fig.savefig(a.results/f'CURVES.{ext}',dpi=200)
plt.close(fig)
