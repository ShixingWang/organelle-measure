# this script reads files under `data/growthrate`,
# visualize the growth curve, 
# and calculate the growth rates,
# and write to a single csv file.

import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.linear_model import LinearRegression

folder_i = Path("plots/growthrate")

subfolders = [
    "EYrainbow_glucose",
    "EYrainbow_glucose_largerBF",
    "EYrainbow_rapamycin_1stTry",
    "EYrainbow_rapamycin_CheckBistability",
    "EYrainbow_1nmpp1_1st",
    "EYrainbow_leucine_large",
    "EYrainbow_leucine"
]

fitter = LinearRegression(fit_intercept=False)
list_rates = []
for subfolder in subfolders:
    df = pd.read_csv(str(folder_i/f"{subfolder}.csv"))
    first = df.groupby("condition").first()['OD']
    df.set_index("condition",inplace=True)
    df["normalized"] = (df['OD']/first)
    df.reset_index(inplace=True)
    rates = []
    for cond in (conditions:= df['condition'].unique()):
        fitter.fit(
            df.loc[df["condition"].eq(cond),"minute"].to_numpy().reshape(-1,1)/60.0,
            np.log(df.loc[df["condition"].eq(cond),"normalized"].to_numpy())
        )
        rates.append(fitter.coef_[0])
    list_rates.append({
        "experiment": subfolder,
        "condition": conditions,
        "growth_rate": rates
    })
    fig = px.line(
        df,
        x="minute",y="normalized",
        color='condition'
    )
    fig.show()
df_rates = pd.concat(pd.DataFrame(rate) for rate in list_rates)
df_rates.to_csv(str(folder_i/"growth_rate.csv"),index=False)

# TODO: Bootstrap or Jackknife
# scipy.stat.bootstrap: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
# Jackknife resampling: https://en.wikipedia.org/wiki/Jackknife_resampling

# Measurements During Rebuttal
OD_rebuttal_individual = pd.read_csv("data/growthrate/rebuttal_OD_individual.csv")



OD_rebuttal_together = pd.read_csv("data/growthrate/rebuttal_OD_together.csv")

