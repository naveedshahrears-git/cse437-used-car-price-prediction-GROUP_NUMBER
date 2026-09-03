from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parent
NBS=['01_data_audit_and_eda.ipynb','02_preprocessing.ipynb','03_feature_engineering.ipynb','04_modeling_and_tuning.ipynb','05_evaluation_and_error_analysis.ipynb']
def main():
    if not (ROOT/'data/raw/vehicles.csv').exists():
        if subprocess.run([sys.executable,'src/download_data.py'],cwd=ROOT).returncode: return 1
    for nb in NBS:
        cmd=[sys.executable,'-m','jupyter','nbconvert','--to','notebook','--execute','--ExecutePreprocessor.timeout=1800','--inplace',str(ROOT/'notebooks'/nb)]
        print('RUNNING',nb); subprocess.run(cmd,cwd=ROOT,check=True)
    print('Done. Review results/, figures/, and report/report.md.'); return 0
if __name__=='__main__': raise SystemExit(main())
