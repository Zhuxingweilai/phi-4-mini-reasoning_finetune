import pandas as pd

fed = pd.read_csv("FEDFUNDS.csv", parse_dates=["observation_date"])
us_cpi = pd.read_csv("CPIAUCSL.csv", parse_dates=["observation_date"])
china_cpi = pd.read_csv("CHNCPIALLMINMEI.csv", parse_dates=["observation_date"])
dxy = pd.read_csv("DTWEXBGS.csv", parse_dates=["observation_date"])
usdcny = pd.read_csv("EXCHUS.csv", parse_dates=["observation_date"])

dxy_monthly = dxy.set_index("observation_date").resample("MS").mean().reset_index()

df = fed.merge(us_cpi, on="observation_date", how="inner") \
       .merge(china_cpi, on="observation_date", how="inner") \
       .merge(dxy_monthly, on="observation_date", how="inner") \
       .merge(usdcny, on="observation_date", how="inner")

df.columns = ["date", "fedfunds", "us_cpi", "china_cpi", "dxy", "usdcny"]

df = df.round({
    "fedfunds": 2,
    "us_cpi": 1,
    "china_cpi": 1,
    "dxy": 2,
    "usdcny": 4
})

df.to_csv("cleaned_macro_data.csv", index=False, encoding="utf-8-sig")