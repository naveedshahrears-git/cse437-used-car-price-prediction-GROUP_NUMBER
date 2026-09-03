from pathlib import Path
import os, json
import numpy as np
import pandas as pd

RANDOM_STATE=437
ROOT=Path(__file__).resolve().parents[1]
RAW_PATH=ROOT/'data/raw/vehicles.csv'
PROCESSED_PATH=ROOT/'data/processed/vehicles_clean.csv'
MODEL_DATA_PATH=ROOT/'data/processed/model_data.csv'
RESULTS_DIR=ROOT/'results'; FIGURES_DIR=ROOT/'figures'; MODELS_DIR=ROOT/'models'
MODEL_MAX_ROWS=int(os.getenv('CSE437_MODEL_MAX_ROWS','120000'))
TUNING_MAX_ROWS=int(os.getenv('CSE437_TUNING_MAX_ROWS','60000'))
CORE_COLUMNS=['price','year','manufacturer','model','condition','cylinders','fuel','odometer','title_status','transmission','drive','size','type','paint_color','state','region','posting_date']
FINAL_FEATURES=['car_age','odometer','mileage_per_year','manufacturer','condition','cylinders','fuel','title_status','transmission','drive','type','paint_color','state','region']
NUMERIC_FEATURES=['car_age','odometer','mileage_per_year']
CATEGORICAL_FEATURES=[c for c in FINAL_FEATURES if c not in NUMERIC_FEATURES]

def ensure_dirs():
    for d in [PROCESSED_PATH.parent,RESULTS_DIR,FIGURES_DIR,MODELS_DIR]: d.mkdir(parents=True,exist_ok=True)

def read_raw(usecols=None):
    if not RAW_PATH.exists():
        raise FileNotFoundError(f'Missing {RAW_PATH}. Run: python src/download_data.py')
    return pd.read_csv(RAW_PATH,usecols=usecols,low_memory=False)

def normalize_text(s):
    return (s.astype('string').str.strip().str.lower().str.replace(r'\s+',' ',regex=True).replace({'':pd.NA,'nan':pd.NA,'none':pd.NA}))

def clean_vehicle_data(df):
    ensure_dirs(); before=len(df); report={'rows_raw':int(before)}
    present=[c for c in CORE_COLUMNS if c in df.columns]; df=df[present].copy(); report['columns_selected']=len(present)
    report['duplicate_rows_removed']=int(df.duplicated().sum()); df=df.drop_duplicates().copy()
    for c in ['price','year','odometer']: df[c]=pd.to_numeric(df[c],errors='coerce')
    p0=len(df); df=df[df.price.between(500,150000)].copy(); report['invalid_price_rows_removed']=int(p0-len(df))
    posting_year = pd.to_datetime(
        df['posting_date'], errors='coerce', utc=True
    ).dt.year
    fallback_year = int(posting_year.dropna().median()) if posting_year.notna().any() else 2021
    posting_year = posting_year.fillna(fallback_year)

    bad_year=(~df.year.between(1900,fallback_year+1)) & df.year.notna()
    report['invalid_year_values_set_missing']=int(bad_year.sum())
    df.loc[bad_year,'year']=np.nan

    bad_odo=(~df.odometer.between(0,500000)) & df.odometer.notna()
    report['invalid_odometer_values_set_missing']=int(bad_odo.sum())
    df.loc[bad_odo,'odometer']=np.nan

    for c in [x for x in present if x not in ['price','year','odometer','posting_date']]:
        df[c]=normalize_text(df[c])

    df['car_age']=(posting_year-df['year']).clip(lower=0)
    df['mileage_per_year']=df['odometer']/(df['car_age']+1)
    df.loc[df.mileage_per_year>100000,'mileage_per_year']=np.nan
    report.update(rows_clean=int(len(df)),rows_removed_total=int(before-len(df)),columns_clean=int(df.shape[1]))
    return df.reset_index(drop=True),report

def make_model_sample(df,max_rows=MODEL_MAX_ROWS):
    out=df[FINAL_FEATURES+['price']].copy()
    if len(out)>max_rows: out=out.sample(max_rows,random_state=RANDOM_STATE)
    return out.reset_index(drop=True)

def save_json(obj,path):
    def conv(x):
        if isinstance(x,np.integer): return int(x)
        if isinstance(x,np.floating): return float(x)
        if isinstance(x,np.ndarray): return x.tolist()
        raise TypeError
    Path(path).write_text(json.dumps(obj,indent=2,default=conv),encoding='utf-8')

def regression_metrics(y_true,y_pred):
    from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,mean_absolute_percentage_error
    return {'MAE':float(mean_absolute_error(y_true,y_pred)),'RMSE':float(mean_squared_error(y_true,y_pred)**0.5),'R2':float(r2_score(y_true,y_pred)),'MAPE_percent':float(mean_absolute_percentage_error(y_true,y_pred)*100)}
