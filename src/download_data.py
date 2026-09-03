from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]; DEST=ROOT/'data/raw/vehicles.csv'
def main():
    if DEST.exists() and DEST.stat().st_size>1_000_000:
        print('Dataset already present:',DEST); return 0
    try:
        import kagglehub
        cache=Path(kagglehub.dataset_download('austinreese/craigslist-carstrucks-data'))
        src=next(cache.rglob('vehicles.csv'))
        DEST.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,DEST)
        print('Saved:',DEST); return 0
    except Exception as e:
        print('Automatic download failed:',e)
        print('Download manually from https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data')
        print('Then place vehicles.csv at',DEST); return 1
if __name__=='__main__': raise SystemExit(main())
