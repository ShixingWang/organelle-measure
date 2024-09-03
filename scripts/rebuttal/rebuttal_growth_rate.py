# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.linear_model import LinearRegression

# %%
# plt.xlabel("Time / min")
# plt.ylabel("OD multiplication")
# plt.plot(
#     np.array([    0,  149,  247,  353,  479]),
#     np.array([0.213,0.214,0.203,0.210,0.213])/0.213,
#     label=r"0 glucose"
# )
# plt.plot(
#     np.array([    0,  132,  210,  306,  481]),
#     np.array([0.017,0.023,0.036,0.064,0.150])/0.017,
#     label=r"0.01% glucose"
# )
# plt.plot(
#     np.array([    0,  123,  240,  480]),
#     np.array([0.006,0.012,0.021,0.118])/0.006,
#     label=r"0.1% glucose"
# )
# plt.plot(
#     np.array([    0,   96,  212,  338,  454,  480]),
#     np.array([0.017,0.026,0.061,0.152,0.384,0.481])/0.017,
#     label=r"1% glucose"
# )
# plt.plot(
#     np.array([    0,   97,  216,  384,  480]),
#     np.array([0.008,0.015,0.034,0.118,0.233])/0.008,
#     label=r"2% glucose"
# )
# plt.plot(
#     np.array([    0,  158,  256,  389,  479]),
#     np.array([0.012,0.034,0.074,0.192,0.440])/0.012,
#     label=r"4% glucose"
# )
# plt.legend()

# # Try incorporate 2 database
# # Read CSV files
# od_glucose_large = pd.read_csv("data/growthrate/EYrainbow_glucose_largerBF.csv")
# od_glucose_small = pd.read_csv("data/growthrate/EYrainbow_glucose.csv")
# od_separate  = pd.read_csv("data/growthrate/rebuttal_OD_individual.csv")
# csv_together = pd.read_csv("data/growthrate/rebuttal_OD_together.csv")
# csv_together = csv_together.iloc[:-1,:]

# # Wash the data to have the same form
# od_glucose_large["condition"] = od_glucose_large["condition"].apply(lambda x: f"{x:g}")
# firsts_OD_large = od_glucose_large[["condition","OD"]].groupby("condition").first()
# od_glucose_large.set_index("condition",inplace=True)
# od_glucose_large["first_OD"] = firsts_OD_large
# od_glucose_large["normalized"] = od_glucose_large["OD"] / od_glucose_large["first_OD"]
# od_glucose_large.reset_index(inplace=True)

# od_glucose_small["condition"] = od_glucose_small["condition"].apply(lambda x: f"{x:g}")
# firsts_OD_small = od_glucose_small[["condition","OD"]].groupby("condition").first()
# od_glucose_small.set_index("condition",inplace=True)
# od_glucose_small["first_OD"] = firsts_OD_small
# od_glucose_small["normalized"] = od_glucose_small["OD"] / od_glucose_small["first_OD"]
# od_glucose_small.reset_index(inplace=True)

# od_separate["time"] = pd.to_datetime(od_separate["time"],format="%H:%M")
# firsts_time_separate = od_separate[["Experiment","time"]].groupby("Experiment").first()
# firsts_OD_separate = od_separate[["Experiment","OD"]].groupby("Experiment").first()
# od_separate.set_index("Experiment",inplace=True)
# od_separate["first_OD"]    = firsts_OD_separate
# od_separate["first_times"] = firsts_time_separate
# od_separate["normalized"] = od_separate["OD"] / od_separate["first_OD"]
# od_separate["minute"] = (od_separate["time"] - od_separate["first_times"]).dt.total_seconds()/60
# od_separate.reset_index(inplace=True)
# od_separate = od_separate[~od_separate["Experiment"].str.contains("first")]
# od_separate["condition"] = od_separate["Experiment"].apply(lambda x: x.partition("-")[2].replace("-","."))

# csv_together["Time"] = pd.to_datetime(csv_together["Time"],format="%H:%M")
# csv_together["minute"] = (csv_together.loc[:,"Time"] - csv_together.loc[0,"Time"]).dt.total_seconds()/60
# ods_together = []
# for group in csv_together.columns[1:-1]:
#     ods_together.append(pd.DataFrame({
#         "minute": csv_together["minute"],
#         "Experiment": group,
#         "OD": csv_together.loc[:,group]
#     }))
# od_together = pd.concat(ods_together,ignore_index=True)
# firsts_OD_together = od_together[["Experiment","OD"]].groupby("Experiment").first()
# od_together.set_index("Experiment",inplace=True)
# od_together["first_OD"] = firsts_OD_together
# od_together["normalized"] = od_together["OD"] / od_together["first_OD"]
# od_together.reset_index(inplace=True)
# od_together["condition"] = od_together["Experiment"].apply(lambda x: x.partition("-")[2].replace("-","."))

# linear = LinearRegression(fit_intercept=False)
# dfs_rate = []
# for r,replicate in enumerate([
#                             #   od_glucose_large,
#                             #   od_glucose_small,
#                               od_separate,
#                               od_together
#                  ]):
#     rates = []
#     internal_mean = []
#     internal_std  = []
#     for cond in (conditions:= replicate["condition"].unique()):
#         df_cond = replicate[replicate["condition"].eq(cond)]
#         cond_minute = df_cond.loc[:,"minute"].to_numpy()/60.0
#         cond_normal = df_cond.loc[:,"normalized"].to_numpy()
#         linear.fit(
#                     cond_minute.reshape(-1,1),
#             np.log2(cond_normal)
#         )
#         rates.append(linear.coef_[0])
        
#         n_measure = len(df_cond)
#         masked_rate = np.zeros(n_measure)
#         for i in range(n_measure):
#             mask_i = np.zeros(n_measure)
#             mask_i[i] = 1
#             mask_minute = np.ma.array(cond_minute,mask=mask_i)
#             mask_normal = np.ma.array(cond_normal,mask=mask_i)
#             linear.fit(
#                         mask_minute.reshape(-1,1),
#                 np.log2(mask_normal)
#             )
#             masked_rate[i] = linear.coef_[0]
#         internal_mean.append(masked_rate.mean())
#         internal_std.append(masked_rate.std())
#     dfs_rate.append(pd.DataFrame({
#         "replicate": r,
#         "condition": conditions,
#         "growth_rate": rates,
#         "internal_mean": internal_mean,
#         "internal_std": internal_std
#     }))
# df_rate = pd.concat(dfs_rate)

# tb_rate = df_rate[["condition","growth_rate"]].groupby(["condition"]).mean()
# tb_rate["std"] = df_rate[["condition","growth_rate"]].groupby(["condition"]).std()


# # Visualization
# list_colors = [1,2,3,4,0,5]
# percentages = {
#     "0":   0,
#     "0.5": 0.01,
#     "5":   0.1,
#     "50":  1,
#     "100": 2,
#     "200": 4,
# }
# plt.figure()
# for c,cond in enumerate(conditions):
#     plt.scatter(
#         od_together.loc[od_together["condition"].eq(cond),"minute"],
#         od_together.loc[od_together["condition"].eq(cond),"normalized"],
#         facecolor=sns.color_palette('tab10')[list_colors[c]],
#         marker="P"
#     )
# for c,cond in enumerate(conditions):
#     plt.scatter(
#         od_separate.loc[od_separate["condition"].eq(cond),"minute"],
#         od_separate.loc[od_separate["condition"].eq(cond),"normalized"],
#         facecolor=sns.color_palette('tab10')[list_colors[c]],
#         marker="X"
#     )
# for c,cond in enumerate(conditions):
#     plt.plot(
#         np.linspace(0,500,1000),
#         np.exp2(np.linspace(0,500,1000)/60.0*tb_rate.loc[cond,"growth_rate"]),
#         '--',c=sns.color_palette('tab10')[list_colors[c]],
#         label=f"fitted exponential curve, {percentages[cond]}% glucose"
#     )
# plt.scatter([],[],facecolor='k',marker="P",label="replicate 1")
# plt.scatter([],[],facecolor='k',marker="X",label="replicate 2")
# plt.legend()
# plt.xlabel("Time (min)")
# plt.ylabel("OD(600nm) Multiplication")
# plt.savefig(f"data/growthrate/rebuttal_replicate_curve.png",dpi=600)

# plt.figure()
# plt.errorbar(
#     np.arange(len(conditions)),
#     tb_rate.loc[conditions,"growth_rate"],
#     yerr=tb_rate.loc[conditions,"std"]/np.sqrt(2),
#     fmt="o",capsize=6
# )
# plt.xticks(
#     ticks=np.arange(len(conditions)),
#     labels=[f"{percentages[c]}%" for c in conditions]
# )
# plt.xlabel("Glucose Concentration")
# plt.ylabel(r"1 / Doubling Time (hour$^{-1}$)")
# plt.savefig("data/growthrate/rebuttal_growth_rate_errorbar.png",dpi=600)


# %% data from replicates
od_replicate = pd.read_csv("plots/growthrate/rebuttal_OD_replicate.csv")
od_replicate["time"] = pd.to_datetime(od_replicate["time"],format="%H:%M")
first_times = od_replicate[["group","replicate","time"]].groupby(["group","replicate"]).first()
first_ODs = od_replicate[["group","replicate","OD"]].groupby(["group","replicate"]).first()
od_replicate.set_index(["group","replicate"],inplace=True)
od_replicate["first_times"] = first_times
od_replicate["first_ODs"] = first_ODs
od_replicate["minute"] = (od_replicate["time"] - od_replicate["first_times"]).dt.total_seconds()/60.0
od_replicate["normalized"] = od_replicate["OD"] / od_replicate["first_ODs"]
od_replicate.reset_index(inplace=True)

# %%
linear = LinearRegression(fit_intercept=False)
means = {}
stds  = {}
for cond in (conditions:=od_replicate["group"].unique()):
    rates = []
    df_cond = od_replicate[od_replicate["group"].eq(cond)]
    for replicate in df_cond["replicate"].unique():
        cond_hour = df_cond.loc[
                                df_cond["replicate"].eq(replicate),
                                "minute"
                            ].to_numpy()/60.0
        cond_norm = df_cond.loc[
                                df_cond["replicate"].eq(replicate),
                                "normalized"
                            ].to_numpy()
        linear.fit(
                cond_hour.reshape(-1,1),
                np.log(cond_norm)
        )
        rates.append(linear.coef_[0])
    means[cond] = np.mean(rates)
    stds[cond]  = np.std(rates)

# %% bar plot
list_colors = [1,2,3,4,0,5]
plt.rcParams["figure.autolayout"]=True
plt.rcParams['font.size'] = '18'
plt.figure()
for c,cond in enumerate(conditions):
    plt.bar(
        [c], height = means[cond],
               yerr =  stds[cond], 
        capsize=15, color=plt.color_sequences['tab10'][list_colors[c]]
    )
plt.xticks(
    ticks=np.arange(len(conditions)),
    labels=["0","0.01%","0.1%","1%","2%","4%"]
)
plt.xlabel("Glucose concentration m/v")
plt.ylabel(r"Growth rate (hour$^{-1}$)")
plt.savefig("plots/growthrate/growthrate_replicate_errorbars.png",dpi=600)

# %% line plot
plt.rcParams['font.size'] = '16'
plt.figure()
for c,cond in enumerate(conditions):
    df_cond = od_replicate[od_replicate["group"].eq(cond)]
    for replicate in df_cond["replicate"].unique():
        df_rep = df_cond[df_cond["replicate"].eq(replicate)]
        plt.plot(
            df_rep["minute"], df_rep["normalized"],
            marker="o", alpha=0.75,
            color=plt.color_sequences['tab10'][list_colors[c]]
        )
for c,conc in enumerate(["0","0.01%","0.1%","1%","2%","4%"]):
    plt.plot(
        [],[],
        marker="o", alpha=0.75,
        color=plt.color_sequences['tab10'][list_colors[c]],
        label=f"{conc} glucose"
    )
plt.legend()
plt.xlabel("Time (min)")
plt.ylabel("OD Multiplication")
plt.savefig("plots/growthrate/growthrate_replicate_line.png",dpi=600)

# %% exponential curve
plt.rcParams['font.size'] = '16'
plt.figure()
for c,cond in enumerate(conditions):
    df_cond = od_replicate[od_replicate["group"].eq(cond)]
    plt.plot(
        np.arange(500), np.exp(np.arange(500)*means[cond]/60.0),
        alpha=0.75, color=plt.color_sequences['tab10'][list_colors[c]]
    )
    for replicate in df_cond["replicate"].unique():
        df_rep = df_cond[df_cond["replicate"].eq(replicate)]
        plt.scatter(
            df_rep["minute"], df_rep["normalized"],
            marker="o", alpha=0.75,
            color=plt.color_sequences['tab10'][list_colors[c]]
        )
# legend
for c,conc in enumerate(["0","0.01%","0.1%","1%","2%","4%"]):
    plt.plot(
        [],[],
        marker="o", alpha=0.75,
        color=plt.color_sequences['tab10'][list_colors[c]],
        label=f"{conc} glucose"
    )
plt.legend()
plt.xlabel("Time (min)")
plt.ylabel("OD Multiplication")
plt.savefig("plots/growthrate/growthrate_replicate_expcurve.png",dpi=600)

# %% [markdown]
# Only include date before 3 hours

# %%
od_replicate = od_replicate[od_replicate["minute"].lt(220)]
# %%
linear = LinearRegression(fit_intercept=False)
means = {}
stds  = {}
for cond in (conditions:=od_replicate["group"].unique()):
    rates = []
    df_cond = od_replicate[od_replicate["group"].eq(cond)]
    for replicate in df_cond["replicate"].unique():
        cond_hour = df_cond.loc[
                                df_cond["replicate"].eq(replicate),
                                "minute"
                            ].to_numpy()/60.0
        cond_norm = df_cond.loc[
                                df_cond["replicate"].eq(replicate),
                                "normalized"
                            ].to_numpy()
        linear.fit(
                cond_hour.reshape(-1,1),
                np.log(cond_norm)
        )
        rates.append(linear.coef_[0])
    means[cond] = np.mean(rates)
    stds[cond]  = np.std(rates)

# %% bar plot
list_colors = [1,2,3,4,0,5]
plt.rcParams["figure.autolayout"]=True
plt.rcParams['font.size'] = '18'
plt.figure()
for c,cond in enumerate(conditions):
    plt.bar(
        [c], height = means[cond],
               yerr =  stds[cond], 
        capsize=15, color=plt.color_sequences['tab10'][list_colors[c]]
    )
plt.xticks(
    ticks=np.arange(len(conditions)),
    labels=["0","0.01%","0.1%","1%","2%","4%"]
)
plt.xlabel("Glucose concentration m/v")
plt.ylabel(r"Growth rate (hour$^{-1}$)")
plt.savefig("plots/growthrate/3hours_growthrate_replicate_errorbars.png",dpi=600)

# %% line plot
plt.rcParams['font.size'] = '16'
plt.figure()
for c,cond in enumerate(conditions):
    df_cond = od_replicate[od_replicate["group"].eq(cond)]
    for replicate in df_cond["replicate"].unique():
        df_rep = df_cond[df_cond["replicate"].eq(replicate)]
        plt.plot(
            df_rep["minute"], df_rep["normalized"],
            marker="o", alpha=0.75,
            color=plt.color_sequences['tab10'][list_colors[c]]
        )
for c,conc in enumerate(["0","0.01%","0.1%","1%","2%","4%"]):
    plt.plot(
        [],[],
        marker="o", alpha=0.75,
        color=plt.color_sequences['tab10'][list_colors[c]],
        label=f"{conc} glucose"
    )
plt.legend()
plt.xlabel("Time (min)")
plt.ylabel("OD Multiplication")
plt.savefig("plots/growthrate/3hours_growthrate_replicate_line.png",dpi=600)

# %% exponential curve
plt.rcParams['font.size'] = '16'
plt.figure()
for c,cond in enumerate(conditions):
    df_cond = od_replicate[od_replicate["group"].eq(cond)]
    plt.plot(
        np.arange(220), np.exp(np.arange(220)*means[cond]/60.0),
        alpha=0.75, color=plt.color_sequences['tab10'][list_colors[c]]
    )
    for replicate in df_cond["replicate"].unique():
        df_rep = df_cond[df_cond["replicate"].eq(replicate)]
        plt.scatter(
            df_rep["minute"], df_rep["normalized"],
            marker="o", alpha=0.75,
            color=plt.color_sequences['tab10'][list_colors[c]]
        )
# legend
for c,conc in enumerate(["0","0.01%","0.1%","1%","2%","4%"]):
    plt.plot(
        [],[],
        marker="o", alpha=0.75,
        color=plt.color_sequences['tab10'][list_colors[c]],
        label=f"{conc} glucose"
    )
plt.legend()
plt.xlabel("Time (min)")
plt.ylabel("OD Multiplication")
plt.savefig("plots/growthrate/3hours_growthrate_replicate_expcurve.png",dpi=600)
# %%
