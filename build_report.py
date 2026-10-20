"""Build the two-page report and Thai recap from completed, actual run outputs."""
from pathlib import Path
import json
from xml.sax.saxutils import escape
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reports'
OUT.mkdir(exist_ok=True)
c = pd.read_csv(ROOT / 'results/classification/cv_summary.csv')
r = pd.read_csv(ROOT / 'results/regression/cv_summary.csv').set_index('variant')
cs = json.loads((ROOT / 'results/classification/locked_selection.json').read_text())['method']
rs = json.loads((ROOT / 'results/regression/selection.json').read_text())['selected_variant']
ch = pd.read_csv(ROOT / 'results/classification/historical_holdout_metrics.csv')
rh = pd.read_csv(ROOT / 'results/regression/holdout_metrics.csv').set_index('variant')
sub = pd.read_csv(ROOT / 'results/regression/holdout_subgroup_diagnostics.csv')
cbase = c.query("method == 'baseline' and threshold_mode == 'fixed_0.5'").iloc[0]
ctune = c.query("method == 'baseline' and threshold_mode == 'calibrated'").iloc[0]
cbest = c.query("method == @cs and threshold_mode == 'calibrated'").iloc[0]
rbase, rbest = r.loc['original'], r.loc[rs]
cbh = ch.query("method == 'baseline' and threshold_mode == 'fixed_0.5'").iloc[0]
csh = ch.query("method == @cs and threshold_mode == 'calibrated'").iloc[0]
rbh, rsh = rh.loc['original'], rh.loc[rs]
meta_c = json.loads((ROOT / 'models/classification/selected/metadata.json').read_text())
meta_r = json.loads((ROOT / 'models/regression' / rs / 'metadata.json').read_text())
tail = sub[(sub.variant == rs) & (sub.group == 'upper_5pct_expenditure') & (sub.value.astype(str).str.lower() == 'true')].iloc[0]
cnames = {'baseline':'Baseline + threshold', 'weight_3':'Weighted BCE: 3', 'weight_10':'Weighted BCE: 10', 'weight_ratio':'Weighted BCE: fit ratio', 'ros_0.10':'Random oversampling: 0.10', 'regularized':'Regularization + threshold'}
rnames = {'original':'Original FFN baseline', 'early_stopping':'Early stopping', 'dropout':'Dropout (0.30)', 'combined':'Combined regularization', 'huber':'Huber + early stopping', 'log_target':'Log target + early stopping'}
ctable = [['Classification / CV','Precision','Recall','F1 (SD)','AP']]
ctable.append(['Original FFN, cutoff 0.5', f'{cbase.precision_mean:.3f}', f'{cbase.recall_mean:.3f}', f'{cbase.f1_mean:.3f} ({cbase.f1_std:.3f})', f'{cbase.average_precision_mean:.3f}'])
for key in cnames:
    row=c.query("method == @key and threshold_mode == 'calibrated'").iloc[0]
    ctable.append([cnames[key],f'{row.precision_mean:.3f}',f'{row.recall_mean:.3f}',f'{row.f1_mean:.3f} ({row.f1_std:.3f})',f'{row.average_precision_mean:.3f}'])
rtable = [['Regression / CV','RMSE (SD)','MAE','R-squared']]
for key in rnames:
    row=r.loc[key]
    rtable.append([rnames[key],f'{row.rmse_mean:,.1f} ({row.rmse_std:,.1f})',f'{row.mae_mean:,.1f}',f'{row.r2_mean:.3f}'])
holdtable = [['Historical holdout diagnostic','Baseline','Selected']]
holdtable += [
    ['Classification F1 / AP',f'{cbh.f1:.3f} / {cbh.average_precision:.3f}',f'{csh.f1:.3f} / {csh.average_precision:.3f}'],
    ['Classification precision / recall',f'{cbh.precision:.3f} / {cbh.recall:.3f}',f'{csh.precision:.3f} / {csh.recall:.3f}'],
    ['Regression RMSE / MAE',f'{rbh.rmse:,.1f} / {rbh.mae:,.1f}',f'{rsh.rmse:,.1f} / {rsh.mae:,.1f}'],
    ['Regression R-squared',f'{rbh.r2:.3f}',f'{rsh.r2:.3f}']]
intro = ('This study uses feedforward neural networks (FFNs) to predict poverty status (pov_exp) and household expenditure (rspcexp23) from SES2023.csv. The objective is performance on held-out observations, not training accuracy. All reported values come from executed experiments; classification prioritizes F1, precision, recall and average precision (AP), while regression uses RMSE, MAE and R-squared.')
architecture = ('All candidates retain the original 32-64-32-1 network with ReLU hidden layers, Adam at learning rate 0.001, and a 50-epoch maximum. Batch sizes are 64 for classification and 32 for regression. Classification uses binary cross-entropy with logits and sigmoid scores. Weighted losses and oversampling alter minority emphasis; an independently selected threshold controls the precision-recall trade-off. Regression uses a standardized target with MSE, Huber loss, or standardized log1p expenditure. Dropout and early stopping aim to reduce memorization.')
methods = ('The unchanged dataset contains 49,664 rows and 769 poor cases (1.55%). Both targets are excluded from predictors. The union of previously evaluated test splits is quarantined; keeping identical predictor profiles together expands it to 18,878 rows, leaving 30,786 development rows (472 positives). Five stratified group folds use split seed 20261020 and model seeds 0 and 42. Within each outer training pool, classification separates fitting/stopping/calibration approximately 60/20/20; regression fits on the first and third subsets. Imputers and scalers learn only from fitting rows; oversampling occurs only there. Thresholds maximize calibration F1 over unique scores; outer labels select neither thresholds nor epochs. Reported SDs describe five fold means after averaging seeds. Selection requires positive paired gains in at least four folds and mean gain above its descriptive SE; regression also requires a 1% RMSE gain and no rural/region MAE increase above 10% (groups with at least 100 rows). This stability rule is not a significance test.')
classfinding = (f'The selected classification combination (dropout 0.20, weight decay 0.0001, patience 7) improved all five paired folds relative to threshold-tuned BCE. CV F1 rose from {cbase.f1_mean:.3f} at the original cutoff to {cbest.f1_mean:.3f}, and AP from {cbase.average_precision_mean:.3f} to {cbest.average_precision_mean:.3f}. Recall increased from {cbase.recall_mean:.3f} to {cbest.recall_mean:.3f}, but precision fell from {cbase.precision_mean:.3f} to {cbest.precision_mean:.3f}. The regularized model predicted no positives at cutoff 0.5, so threshold selection was essential. Oversampling and weights 3/10 did not reliably improve F1. The combination is supported as a whole; individual regularizers were not isolated.')
regfinding = (f'{rnames[rs]} was selected by the prespecified rule. CV RMSE decreased from {rbase.rmse_mean:,.1f} to {rbest.rmse_mean:,.1f} ({100*(1-rbest.rmse_mean/rbase.rmse_mean):.1f}%), MAE from {rbase.mae_mean:,.1f} to {rbest.mae_mean:,.1f}, and R-squared changed from {rbase.r2_mean:.3f} to {rbest.r2_mean:.3f}. All metrics and stopping decisions use original expenditure units. Huber and log-target RMSE differed by only 0.4, so there is no convincing separation between them; log-target had lower MAE. No extremes were removed or predictions clipped. Inverse log predictions were not corrected for conditional-mean bias.')
holdfinding = (f'Model choice was saved before final fitting and holdout prediction. The final classification cutoff was {meta_c["threshold"]:.5f}; the retained epoch was {meta_c["retained_epoch"]}. Compared with the 0.5 baseline, true positives increased from {int(cbh.tp)} to {int(csh.tp)}, but false positives increased from {int(cbh.fp)} to {int(csh.fp)}. This is a substantial precision cost, not an improvement on every criterion. The selected regression retained epoch {meta_r["selected_epoch"]}. In the upper expenditure tail (development 95th-percentile cutoff), holdout MAE was {tail.mae:,.1f} over {int(tail.n_unique_rows):,} observations, with mean residual {tail.mean_residual:,.1f} (prediction minus actual). Tail underprediction remains a major limitation. Lowest-quintile holdout MAE also increased from 1,589.5 to 1,910.8 (+20.2%), despite nearly unchanged CV MAE (1,617.5 to 1,621.4). Aggregate gains therefore do not extend to every group.')
limitations = ('The fixed-epoch classification baseline averaged 11,550 optimizer steps, oversampling 12,550, and the regularized candidate 2,911; compute budgets are not identical. Baseline training AP was 0.960 versus CV AP 0.095, compared with 0.278 versus 0.166 for regularization. This supports reduced overfitting, not universal superiority. CV guided configuration choice and is therefore selection-optimistic. The historical holdout was previously exposed and is not a pristine external test. Predictor profiles are not household IDs; survey weights, sampling clusters and future-year data are unavailable. Expenditure units follow the supplied column without an assumed currency/time definition. Rare positives and heavy expenditure tails constrain generalization.')
notes = ('SMOTENC was deferred because synthetic mixing requires additional checks for region exclusivity and household-share composition. Original files were preserved. Full tables, checkpoints, predictions, residual plots, seeds and execution logs accompany this report.')

styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyEC',fontName='Helvetica',fontSize=9.5,leading=12.5,spaceAfter=6,textColor=colors.HexColor('#263447')))
styles.add(ParagraphStyle(name='HeadEC',fontName='Helvetica-Bold',fontSize=11,leading=14,spaceBefore=8,spaceAfter=5,textColor=colors.HexColor('#123C55')))
styles.add(ParagraphStyle(name='SmallEC',fontName='Helvetica',fontSize=8,leading=10,spaceAfter=5,textColor=colors.HexColor('#46566B')))
styles.add(ParagraphStyle(name='TitleEC',fontName='Helvetica-Bold',fontSize=19,leading=23,spaceAfter=6,textColor=colors.HexColor('#123C55')))
styles.add(ParagraphStyle(name='CellEC',fontName='Helvetica',fontSize=8.2,leading=10))
def para(text, style='BodyEC'): return Paragraph(escape(text),styles[style])
def table(rows,widths):
    t=Table([[para(str(cell),'CellEC') for cell in row] for row in rows],colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E6F0F4')),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#87A7BA')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F5F8FA')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6)]))
    return t
story=[para('Feedforward Neural Networks','TitleEC'),para('EC521 | Poverty classification and expenditure regression | 20 October 2026','SmallEC')]
for heading,text in [('Introduction',intro),('Model Architecture and Conceptual Framework',architecture),('Methods and Data',methods)]:
    story += [para(heading,'HeadEC'),para(text)]
story += [para('Findings and Discussion','HeadEC'),para('Table 1. Development CV. All rows except the first use an independently selected threshold.','SmallEC'),table(ctable,[205,62,60,103,77]),Spacer(1,7),para(classfinding),PageBreak(),para('Findings and Discussion (continued)','HeadEC'),para('Table 2. Regression development CV; errors are in original rspcexp23 units.','SmallEC'),table(rtable,[205,132,85,85]),Spacer(1,7),para(regfinding),para('Table 3. Reused historical holdout (18,878 rows): diagnostic comparison only.','SmallEC'),table(holdtable,[235,136,136]),Spacer(1,7),para(holdfinding),para('Interpretation and limitations','HeadEC'),para(limitations),para(notes),para('Implementation references: scikit-learn, Common pitfalls and recommended practices (scikit-learn.org/stable/common_pitfalls.html); imbalanced-learn, SMOTENC API (imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html). Numerical source: included run outputs.','SmallEC')]
def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#C9D7DF'));canvas.line(44,37,A4[0]-44,37)
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#536578'))
    canvas.drawString(44,25,'EC521 | Reproducible FFN experiments');canvas.drawRightString(A4[0]-44,25,str(doc.page))
doc=SimpleDocTemplate(str(OUT/'EC521_Report.pdf'),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=38,bottomMargin=48,title='EC521 Feedforward Neural Networks',author='EC521 homework')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
def mdtable(rows):
    return '\n'.join(['| '+' | '.join(row)+' |' for row in [rows[0],['---']*len(rows[0]),*rows[1:]]])
md='# EC521: Feedforward Neural Networks\n\n20 October 2026\n\n'
for heading,text in [('Introduction',intro),('Model Architecture and Conceptual Framework',architecture),('Methods and Data',methods)]:md+=f'## {heading}\n\n{text}\n\n'
md+='## Findings and Discussion\n\n'+mdtable(ctable)+'\n\n'+classfinding+'\n\n'+mdtable(rtable)+'\n\n'+regfinding+'\n\n'+mdtable(holdtable)+'\n\n'+holdfinding+'\n\n'+limitations+'\n\n'+notes+'\n\nReferences: [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html); [SMOTENC documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html).\n'
(OUT/'EC521_Report.md').write_text(md)
# A combined machine-readable comparison with unused task-specific metrics blank.
combined=[]
for _,row in c.iterrows():combined.append(dict(task='classification',model=row.method,threshold_mode=row.threshold_mode,precision=row.precision_mean,recall=row.recall_mean,f1=row.f1_mean,ap=row.average_precision_mean,f1_fold_sd=row.f1_std))
for key,row in r.iterrows():combined.append(dict(task='regression',model=key,rmse=row.rmse_mean,mae=row.mae_mean,r2=row.r2_mean,rmse_fold_sd=row.rmse_std))
pd.DataFrame(combined).to_csv(ROOT/'results/model_comparison.csv',index=False)
thai=f'''# Recap งาน EC521: FFN Classification และ Regression

จัดทำจากผลรันจริงวันที่ 20 ตุลาคม 2026 เพื่อให้ตรวจสอบก่อนส่งอาจารย์

## ผลตรวจทรัพยากรและการรักษาต้นฉบับ

เข้าถึง GitHub Repository `NetJit03/FFN_Homework` ได้จริง และพบ `SES2023.csv` ในโครงการ อ่านโค้ดเดิมทั้งหมด 10 ไฟล์ รวมถึงไฟล์การทดลองเดิม 6 ตารางก่อนพัฒนา Environment `.venv` ใช้ Python 3.14.4, PyTorch 2.14.1, scikit-learn 1.9.1 และ imbalanced-learn 0.14.2 โดยทดสอบการคำนวณ gradient/อัปเดตน้ำหนักและ Oversampling จริงแล้ว งานใหม่อยู่ใน `ec521_submission/` ไม่แก้ไฟล์เดิม ไม่ commit หรือ push ทับ main และตรวจค่า SHA-256 ของไฟล์เดิมไว้

ข้อมูลมี 49,664 แถว ตัวแปรอิสระ 32 ตัว และเป้าหมาย 2 ตัว มีผู้มีสถานะยากจน 769 แถว (1.5484%) จึงไม่เลือกโมเดลจาก Accuracy เพียงอย่างเดียว ตัวแปร `electcon` มี Yes 49,634 แถวและค่าขาดหาย 30 แถว ค่าที่สังเกตได้จึงเป็นค่าคงที่ ส่วน `rspcexp23` มีมัธยฐาน 6,758 และค่าสูงสุด 360,795 หน่วยตามไฟล์ที่ให้มา ไม่สมมติหน่วยเงินหรือช่วงเวลาเพิ่มเอง

## แก้โค้ดตรงไหน และเพราะอะไร

1. **การแปลงและเติมข้อมูล:** แก้การแปลง Yes/No ให้เก็บผลกลับจริง โดยไฟล์ regression comparison เดิมเรียก `replace()` แต่ไม่ได้ assign ผลกลับ ทำให้ Yes ถูกแปลงเป็น missing ในขั้นถัดไป สคริปต์ใหม่ตรวจชนิดค่าอย่างชัดเจน และ fit Median Imputer/StandardScaler เฉพาะข้อมูลฝึก ไม่เติมค่าเป้าหมายที่หายไป ข้อผิดพลาดเรื่องเติมค่าก่อนแบ่งชุดในต้นฉบับเป็นความเสี่ยงของวิธีทำ แต่ข้อมูลครั้งนี้ขาดเฉพาะตัวแปรค่าคงที่ จึงไม่อ้างว่าแก้จุดนี้แล้วทำให้คะแนนดีขึ้นโดยตรง
2. **จัดการ Test Set ที่เคยดูผลแล้ว:** สร้างชุดทดสอบเดิมทั้งแบบ stratified และไม่ stratified จาก seed=1 และกัน union 17,867 แถวออกจากการเลือกโมเดลทั้งหมด ขยายให้รูปแบบตัวแปรที่เหมือนกันอยู่ฝั่งเดียวกัน ได้ชุดประเมินเดิมรวม 18,878 แถว เหลือชุดพัฒนา 30,786 แถว (จน 472 แถว) ชุดทดสอบเดิมมีคนจน 297 แถว ชุดนี้เป็นการวินิจฉัยจากข้อมูลที่เคยถูกใช้ในงานเดิม ไม่ใช่ external test ใหม่
3. **แยกหน้าที่ของข้อมูล:** ใช้ 5-fold StratifiedGroupKFold โดย group คือรูปแบบตัวแปรที่เหมือนกัน ไม่ใช่ household ID ภายใน fold ของ Classification แบ่ง fit/early-stopping/calibration ประมาณ 60/20/20% แยกจาก outer evaluation; Regression รวม fit+calibration เป็นชุดฝึกและคงชุด stopping แยกไว้ ทุกโมเดลภายในงานเดียวกันใช้ชุดแบ่งเหมือนกัน
4. **Threshold:** เลือก threshold จากคะแนนในชุด calibration โดยหา F1 สูงสุดจากค่าคะแนนจริง ไม่ใช้ outer evaluation เลือก threshold และไม่จำกัดช่วงเดิม 0.05–0.95 หากเสมอกันเลือก threshold สูงกว่า จึงแยกผล cutoff=0.5 กับผลปรับ threshold ให้เห็นอย่างชัดเจน
5. **การฝึกและความยุติธรรม:** คงสถาปัตยกรรมเดิม 32→64→32→1, ReLU, Adam 0.001, สูงสุด 50 epochs, batch 64 สำหรับ Classification และ 32 สำหรับ Regression ใช้ seeds 0 และ 42 บนทุก fold บันทึก epochs และ optimizer steps จริง เพราะ Oversampling ทำให้จำนวน steps เพิ่ม ส่วน Early Stopping ลด steps การเทียบนี้จึงไม่ใช่ equal-compute experiment
6. **การเลือกโมเดล:** กำหนดกติกาก่อนดูผลชุด holdout โดยต้องชนะอย่างน้อย 4 ใน 5 fold และกำไรเฉลี่ยมากกว่า SE เชิงพรรณนาของผลต่างแบบจับคู่ Regression ต้องลด RMSE อย่างน้อย 1% และไม่ทำให้ MAE ของกลุ่ม rural/region ที่มีอย่างน้อย 100 แถวแย่ลงเกิน 10% กติกานี้ช่วยดูความสม่ำเสมอ แต่ไม่ใช่การทดสอบนัยสำคัญทางสถิติ
7. **นำไปใช้กับข้อมูลใหม่:** บันทึกน้ำหนัก โมเดล preprocessing และ threshold/target transformation มีคำสั่งอ่าน CSV ที่มีแต่ตัวแปรอิสระเพื่อทำนายได้ ไม่จำเป็นต้องมี target และปฏิเสธค่าที่ไม่สามารถตีความเป็นตัวเลขแทนการแปลงเงียบ ๆ

## Classification: ผลที่ได้จริง

ทดลอง 6 วิธี × 5 fold × 2 seeds รวม 60 รอบฝึก และประเมินทั้ง threshold 0.5 และ threshold ที่เลือกจาก calibration ตารางแสดงค่าเฉลี่ยของ 5 fold หลังเฉลี่ย seeds ภายใน fold แล้ว SD เป็นความแปรปรวนข้าม fold ไม่ใช่ confidence interval

{mdtable(ctable)}

เลือก **{cnames[cs]}**: dropout=0.20, weight decay=0.0001 และ early stopping patience=7 โมเดลนี้ชนะ baseline ที่ปรับ threshold แล้วทั้ง 5 fold ค่า CV F1 เพิ่มจาก baseline cutoff 0.5 ที่ {cbase.f1_mean:.4f} เป็น {cbest.f1_mean:.4f}; AP เพิ่มจาก {cbase.average_precision_mean:.4f} เป็น {cbest.average_precision_mean:.4f} แต่ Precision ลดจาก {cbase.precision_mean:.4f} เป็น {cbest.precision_mean:.4f} ขณะที่ Recall เพิ่มจาก {cbase.recall_mean:.4f} เป็น {cbest.recall_mean:.4f}

เมื่อเทียบกับ **baseline ที่ปรับ threshold แล้วเหมือนกัน** F1 เพิ่มจาก {ctune.f1_mean:.4f} เป็น {cbest.f1_mean:.4f} จึงไม่ได้ให้ประโยชน์จาก threshold แก่โมเดลใหม่เพียงฝั่งเดียว Weighted BCE ตามอัตราส่วนคลาสช่วยได้บางส่วน แต่ weights=3/10 และ Random Oversampling 0.10 ไม่ได้ให้ผล F1 ดีขึ้นอย่างสม่ำเสมอ จึงไม่ขยายการค้นหาเพิ่มเติม

Baseline มี AP บนข้อมูลฝึกเฉลี่ย 0.9598 แต่ CV เพียง 0.0945 สะท้อนการจำข้อมูลฝึกมาก โมเดล regularized มี AP ฝึก 0.2779 และ CV 0.1659 ช่องว่างจึงลดลง อย่างไรก็ตาม ยังแยกไม่ได้ว่า Dropout, Weight Decay หรือ Early Stopping แต่ละอย่างช่วยมากเท่าใด เพราะทดลองเป็นชุดร่วมกัน หากใช้ threshold 0.5 โมเดล regularized ไม่ทำนายคลาสยากจนเลยใน CV และ F1=0 จึงต้องใช้ threshold ที่เลือกอย่างถูกวิธี

โมเดลสุดท้ายใช้ seed=0 ตามที่กำหนดไว้ threshold={meta_c['threshold']:.8f}, เก็บน้ำหนัก epoch {meta_c['retained_epoch']} หลังฝึกจริง {meta_c['epochs_executed']} epochs ในชุด holdout เดิม TP เพิ่มจาก {int(cbh.tp)} เป็น {int(csh.tp)} แต่ FP เพิ่มจาก {int(cbh.fp)} เป็น {int(csh.fp)} Precision ลดจาก {cbh.precision:.4f} เป็น {csh.precision:.4f}; Recall เพิ่มจาก {cbh.recall:.4f} เป็น {csh.recall:.4f} นี่คือ trade-off ที่ชัดเจน และคะแนนยังไม่สูงพอจะอ้างว่าเหมาะกับการตัดสินใจจริงโดยไม่มีการประเมินเพิ่ม

ไม่ทดลอง SMOTENC ในรอบนี้ เพราะต้องตรวจข้อจำกัดร่วมของ region one-hot และสัดส่วนอายุในครัวเรือนก่อนสร้างข้อมูลสังเคราะห์ Random Oversampling เก็บรูปแบบข้อมูลจริงไว้และเหมาะกับการเทียบในขอบเขตเวลานี้

## Regression: ผลที่ได้จริง

ทดลอง 6 วิธี × 5 fold × 2 seeds รวม 60 รอบฝึก ทุกวิธีแปลงค่าทำนายกลับหน่วยเดิมก่อนคำนวณ RMSE, MAE และ R² รวมถึงใช้ RMSE หน่วยเดิมสำหรับ Early Stopping

{mdtable(rtable)}

เลือก **{rnames[rs]}** ค่าเฉลี่ย CV RMSE ลดจาก {rbase.rmse_mean:,.2f} เป็น {rbest.rmse_mean:,.2f} หรือ {100*(1-rbest.rmse_mean/rbase.rmse_mean):.2f}% MAE เปลี่ยนจาก {rbase.mae_mean:,.2f} เป็น {rbest.mae_mean:,.2f} และ R² จาก {rbase.r2_mean:.4f} เป็น {rbest.r2_mean:.4f} การเลือกนี้ผ่านกติกา CV และตัวกรองความผิดพลาดในกลุ่ม rural/region ไม่ใช้คะแนน holdout เปลี่ยนตัวเลือก อย่างไรก็ตาม Huber กับ log-target มี RMSE ต่างกันเพียงประมาณ 0.37 หน่วย และ log-target มี MAE ต่ำกว่า จึงไม่มีหลักฐานน่าเชื่อถือว่า Huber เหนือกว่า log-target โดยทั่วไป การเลือก Huber เป็นผลจากกติกา RMSE ที่กำหนดไว้

มีการทดลอง Huber loss และ log1p target แล้ว โดย log inverse ไม่ได้แก้ bias ของ conditional mean อัตโนมัติ จึงไม่อ้างว่าการลด error ใน log-space จะดีที่สุดในหน่วยจริง โมเดลสุดท้ายที่เลือกเก็บ epoch {meta_r['selected_epoch']} จากการฝึกจริง {meta_r['epochs_run']} epochs

ตรวจ residuals ทั้งช่วงค่าเต็ม กลุ่มภูมิภาค ชนบท/เมือง quintiles และรายจ่ายหางบน ไม่ลบ outliers และไม่ clip ค่าทำนายเพื่อให้คะแนนดูดีขึ้น กลุ่มรายจ่ายสูงเกิน percentile 95 ของชุดพัฒนา มี holdout {int(tail.n_unique_rows):,} แถว และ MAE {tail.mae:,.2f}, mean residual {tail.mean_residual:,.2f} จึงยังทำนายรายจ่ายสูงต่ำกว่าจริงมาก แม้คะแนนรวมดีขึ้น นอกจากนี้ ใน holdout กลุ่มรายจ่าย quintile ต่ำสุด MAE แย่ลงจาก 1,589.54 เป็น 1,910.80 หรือประมาณ 20.21% โดยทำนายสูงกว่าจริงเฉลี่ย 1,866.50 หน่วย แม้ CV ของกลุ่มนี้แทบไม่เปลี่ยน (1,617.49 → 1,621.43) จึงต้องระบุชัดว่าคะแนนรวมดีขึ้นไม่ได้หมายความว่าทุกกลุ่มดีขึ้น และไม่เปลี่ยน hyperparameters ย้อนหลังจากผล holdout นี้ การผ่านกติกากลุ่มไม่ใช่หลักฐานความเป็นธรรมเชิงนโยบาย

## ผลชุด holdout เดิม: รายงานหลังล็อกตัวเลือกแล้วเท่านั้น

{mdtable(holdtable)}

ตัวเลขในส่วนนี้ห้ามเรียกว่า external test ใหม่ และไม่ได้ใช้เลือก hyperparameters หลังดูผล มีบันทึกการเลือกก่อนสร้างผล holdout พร้อม checkpoint และค่าทำนายรายแถวให้ตรวจกลับได้

## การรันซ้ำและข้อจำกัด

- ผ่านการรัน smoke test ด้วยข้อมูลจริงก่อนรันเต็ม ผล smoke เก็บแยกและไม่ใช้ในรายงาน
- ตรวจการคำนวณ metrics ใหม่จาก prediction รายแถว ตรวจ scaler/imputer จากข้อมูล fit ตรวจ threshold จาก calibration แบบอิสระ ตรวจการโหลด checkpoint กลับมา และตรวจรันซ้ำตัวแทนของ baseline/โมเดลที่เลือกด้วย seed เดิม ดูหลักฐานใน `checks/verification.json` และ `checks/reproducibility.json`
- CV ใช้ทั้งเปรียบเทียบและเลือก configuration คะแนนผู้ชนะจึงยังมี selection optimism ไม่ใช่ fully nested model-selection CV; fold ใช้ข้อมูลฝึกซ้อนกันและ seeds ใช้แถวประเมินชุดเดียวกัน จึงไม่ใช่ตัวอย่างอิสระ 10 ชุด
- ชุด holdout เคยเกี่ยวข้องกับการทดลองเดิมแล้ว ไม่มีข้อมูลใหม่ต่างปีหรือต่างแหล่ง จึงสรุปได้เฉพาะความสามารถภายในข้อมูลสำรวจชุดนี้
- ไม่ได้รับ household IDs, survey weights หรือ sampling clusters การ group ตาม predictors ช่วยลดการข้ามชุดของรูปแบบเหมือนกัน แต่ไม่แทนโครงสร้างการสุ่มตัวอย่างจริง และไม่ควรอ้างค่าประมาณระดับประชากร
- ผู้มีสถานะยากจนมีน้อย ทำให้ threshold/precision/recall แปรปรวน ขณะที่ Regression มีหางรายจ่ายสูงและ outlier มาก ค่า RMSE ข้าม fold จึงกระจายตัว
- ตัวเลขเดิมใน CSV ต้นฉบับไม่ได้นำมารวมเป็นผลใหม่ เพราะ preprocessing และชุดแบ่งต่างกัน Baseline ใหม่คือสถาปัตยกรรมเดิมภายใต้วิธีประเมินที่แก้ไขแล้ว

## รายการไฟล์สำหรับตรวจและส่ง

`classification.py`, `regression.py`, `common.py`, `requirements.txt`, `README.md`, ตาราง `results/`, กราฟ `figures/`, โมเดล `models/`, รายงานอังกฤษสองหน้า `reports/EC521_Report.pdf`, ฉบับแก้ไขข้อความ `.md`, Recap ฉบับนี้ และหลักฐานตรวจงานใน `checks/` มีไฟล์ ZIP รวมทั้ง dataset เพื่อรันซ้ำในเครื่องได้โดยไม่ต้องดาวน์โหลดจาก GitHub

ยังไม่มีการส่งให้อาจารย์หรืออัปโหลดข้อมูลไปที่อื่น ชุดงานนี้เตรียมให้เจ้าของงานตรวจสอบก่อนส่ง
'''
(OUT/'Recap_TH.md').write_text(thai)
print('Generated English PDF + Markdown, Thai recap, and combined comparison CSV.')
