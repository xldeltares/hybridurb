# """
# Created on 01/09/2019
# @author: Xiaohan Li
# Updated: 19/09/2019
# Updated by Mees Radema, nov. 2021
# version: python 3
#
# Objective: finds the best model in term of category and model strcuture
#
# !NOTE! The catogeory method used here is different from the sortCat used in C:\Users\u0107727\Dropbox\MATLAB\python\FloodMAPPING\20190214_Logistic Regression_v3.1.py
# That one consideres the isolated nodes between categories to be the same as its successors and predesessors
# While here does not consider so.
# Therefore, the model strucure resulted is different.
#
# NOTE! v4.1 for Gent, calibration events from sythetic 40
# v4.2 for Antwerp, calibration events from design composite
#
# input:
#     - graph: sewer_graph_networkx.gpickle
#     - Calibration events .csv files
#
# Output:
# Test
# """ DO NOT REMOVE THE LINE COMMENTS in this segment! I don't have a CLUE why but removing them gives errors. Yeah.
import sys

sys.path.append("./FloodMAPPING")
from inspect import signature

import numpy as np
import scipy
import statsmodels.api as sm
from patsy.highlevel import dmatrices
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

# logistic regression
from sklearn.model_selection import StratifiedKFold, train_test_split

from floodMAPPING_funcs_p3 import *


#############################################################
#  Functions
############################################################
class py_solution:
    def sub_sets(self, sset):
        return self.subsetsRecur([], sorted(sset))

    def subsetsRecur(self, current, sset):
        if sset:
            return self.subsetsRecur(current, sset[1:]) + self.subsetsRecur(
                current + [sset[0]], sset[1:]
            )
        return [current]


# real calibration data
def readTablemq_v4_1(X, filename, version, tablepath):
    # read calibration table
    # !NOTE: the calculaiton method might be wrong, need more physical meaning
    table = pd.read_csv(tablepath + filename, index_col=[0])
    # table = pd.read_csv(path + r"Calibration events\\" + version +
    #                     "\Calibration_" + filename + '.csv', index_col=[0])
    # init
    d = table.copy()
    # cat
    d.loc[:, "OT4Cat"] = d.loc[:, "o_Tmo_yr"]
    # bottom up

    d.loc[:, "d"] = d.loc[:, "chambdepth_m"].values
    d.loc[:, "NS"] = (
        d.loc[:, "nc_ROmq_m3ds"].values * 300 / d.loc[:, "us_area_m2"].values * 1000.0
    )  # mm/5min
    # d.loc[:,'PS'] = d.loc[:,'nc_Qmq_m3ds'].values*300 / d.loc[:,'us_area_m2'].values * 1000.0 #mm/5min
    d.loc[:, "PS"] = d.loc[:, "nc_Qmq_m3ds"].values / d.loc[:, "Qo_m3ds"].values
    d.loc[:, "OT"] = d.loc[:, "o_Tmq_yr"]
    d.loc[:, "OT"] = d.loc[:, "OT"].round(1)
    d.loc[d.OT < 0.1, "OT"] = 0.01
    # top down
    d.loc[:, "UST"] = d.loc[:, "nc_ITCTmitc_yr"]
    d.loc[:, "UST"] = d.loc[:, "UST"].round(1)
    d.loc[d.UST <= 0.0, "UST"] = 0.01
    d.loc[:, "DST"] = d.loc[:, "o_Tmitc_yr"]
    d.loc[:, "DST"] = d.loc[:, "DST"].round(1)
    d.loc[d.DST < 0.1, "DST"] = 0.01
    d.loc[:, "IT"] = d.loc[:, "nc_ITTmitc_yr"]
    d.loc[:, "IT"] = d.loc[:, "IT"].round(1)
    d.loc[d.IT < 0.1, "IT"] = 0.01
    # post processing
    d = d[["OT4Cat", "NS", "PS", "OT", "d", "IT", "UST", "DST", "n_F"]]
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(how="any", axis=0)
    d = d.loc[d.NS > 0, :]
    d = d.loc[d.PS > 0, :]
    d = d.loc[d.d > 0, :]
    d = d.loc[d.IT > 0, :]
    d = d.loc[d.UST > 0, :]
    d["n_F"] = d.loc[:, "n_F"].astype(int)
    d["Event"] = filename
    d = d.dropna(how="any")
    return d


# real calibration data
def readTablemq_v4_1_2(X, keyword, version):
    # read calibration table
    # !NOTE: the calculaiton method might be wrong, need more physical meaning
    table = pd.read_csv(
        path + r"CalibrationEvents\\" + version + "\Calibration_" + keyword + ".csv",
        index_col=[0],
    )
    # init
    d = table.copy()
    Tc = pd.DataFrame.from_dict(dict(X.nodes(data="tc_min")), orient="index").rename(
        columns={0: "Tc"}
    )
    d = d.join(Tc)
    # cat
    d.loc[:, "OT4Cat"] = d.loc[:, "o_Tmo_yr"]
    # bottom up
    d.loc[:, "d"] = d.loc[:, "chambdepth_m"].values
    d.loc[:, "NS"] = (
        d.loc[:, "nc_ROmq_m3ds"].values * 300 / d.loc[:, "us_area_m2"].values * 1000.0
    )  # mm/5min
    d.loc[:, "PS"] = d.loc[:, "nc_Qmq_m3ds"].values / d.loc[:, "Qo_m3ds"].values
    d.loc[:, "OT"] = d.loc[:, "o_Tmq_yr"]
    d.loc[:, "OT"] = d.loc[:, "OT"].round(1)
    d.loc[d.OT < 0.1, "OT"] = 0.01
    # top down
    i = d.loc[:, "nc_Qmq_m3ds"] / d.loc[:, "us_area_m2"] * 1000 * 60 * 60  # mm/hr
    T_func_h = T_func()
    d.loc[:, "UST"] = T_func_h.calc(np.vstack([i.values, d.loc[:, "Tc"].values]).T)
    d.loc[:, "DST"] = d.loc[:, "OT"]
    d.loc[:, "IT"] = d.loc[:, "NS"]
    # post processing
    d = d[["OT4Cat", "NS", "PS", "OT", "d", "IT", "UST", "DST", "n_F"]]
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(how="any", axis=0)
    d = d.loc[d.NS > 0, :]
    d = d.loc[d.PS > 0, :]
    d = d.loc[d.d > 0, :]
    d = d.loc[d.IT > 0, :]
    d = d.loc[d.UST > 0, :]
    d["n_F"] = d.loc[:, "n_F"].astype(int)
    d["Event"] = keyword
    d = d.dropna(how="any")
    return d


# box cox tranformation
def bcTransfrom(lams, d):
    for v in lams.keys():
        d.loc[:, v] = stats.boxcox(d.loc[:, v], lams[v])
    return d


# k-fold
def kfoldValidation(equ, d):
    # d is after box-cox transfromation
    y_0, X_0 = dmatrices(equ, d.loc[d.n_F == 0, :], return_type="dataframe")
    y_1, X_1 = dmatrices(equ, d.loc[d.n_F == 1, :], return_type="dataframe")
    cv = StratifiedKFold(n_splits=10)
    aps = []
    fig, ax = plt.subplots()
    thres = np.linspace(0, 1, 10000)
    ps = np.empty((10, 10000))
    rs = np.empty((10, 10000))
    count = 0
    for sub0, sub1 in zip(cv.split(X_0, y_0), cv.split(X_1, y_1)):
        y_train = pd.concat([y_0.iloc[sub0[0], :], y_1.iloc[sub1[0], :]])
        X_train = pd.concat([X_0.iloc[sub0[0], :], X_1.iloc[sub1[0], :]])
        y_test = pd.concat([y_0.iloc[sub0[1], :], y_1.iloc[sub1[1], :]])
        X_test = pd.concat([X_0.iloc[sub0[1], :], X_1.iloc[sub1[1], :]])
        count += 1
        logit = sm.Logit(y_train, X_train)
        m = logit.fit()
        y_prob = m.predict(X_test)
        y_pred = np.where(y_prob > 0.5, 1, 0)
        precison, recall, thresholds = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        aps.append(ap)
        p = np.interp(thres, thresholds, precison[:-1])
        ps[count - 1] = p
        r = np.interp(thres, thresholds, recall[:-1])
        rs[count - 1] = r
        if count == 1:
            # im1 = plt.scatter(r, p, c = thres, s = 2, cmap = 'jet', vmax = 1, vmin = 0, label = '10-fold validation')
            plt.plot(r, p, lw=1, alpha=0.3, label="10-fold validation", zorder=2)
        else:
            # im1 = plt.scatter(r, p, c = thres, s = 2, cmap = 'jet', vmax = 1, vmin = 0)
            plt.plot(r, p, lw=1, alpha=0.3, zorder=2)
    df = pd.DataFrame(
        data=np.array([np.mean(rs, axis=0), np.mean(ps, axis=0), thres]).T,
        columns=["r", "p", "tres"],
    )
    df.to_csv(path + "CalibrationEvents\\" + version + f"\\PRC_{equ}.csv")
    ploty = np.insert(
        np.mean(ps, axis=0), 0, np.linspace(0, np.min(np.mean(ps, axis=0)), 1000)
    )
    plotx = np.insert(
        np.mean(rs, axis=0),
        0,
        np.ones(
            1000,
        ),
    )
    plotz = np.insert(
        thres,
        0,
        np.zeros(
            1000,
        ),
    )
    im2 = plt.scatter(
        plotx,
        ploty,
        s=5,
        c=plotz,
        cmap="Spectral_r",
        label=f"mAP (AP = %.2f $\pm$ %.2f)" % (np.mean(aps), np.std(aps)),
        zorder=4,
    )
    plt.fill_between(
        np.mean(rs, axis=0),
        np.mean(ps, axis=0) - np.std(ps, axis=0),
        np.mean(ps, axis=0) + np.std(ps, axis=0),
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
        zorder=1,
    )
    clb = plt.colorbar()
    clb.ax.set_ylabel("Prediction thresholds")
    f_scores = np.linspace(0.2, 0.8, num=4)
    for i, f_score in enumerate(f_scores):
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        if i == 0:
            (l,) = plt.plot(
                x[y >= 0],
                y[y >= 0],
                color="gray",
                alpha=0.5,
                label="iso-f1 curves",
                zorder=0,
            )
        else:
            (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.5)
        plt.annotate(
            "f1={0:0.1f}".format(f_score),
            xy=(f_score, f_score),
            xycoords="data",
            color="gray",
            rotation=-45,
            ha="center",
            zorder=0,
        )
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve (PRC)")
    plt.legend(loc="lower right", scatterpoints=3)
    plt.savefig(path + "CalibrationEvents\\" + version + f"\\PRC_{equ}.png", dpi=300)
    plt.close()
    print("P = %.2f" % (np.mean(aps)))
    return aps


# fit model with call calibration
def modelCalibration(equ, d, plot=True):
    y_train, X_train = dmatrices(equ, d, return_type="dataframe")
    logit = sm.Logit(y_train, X_train)
    model = logit.fit(method="bfgs")
    y_prob = model.predict(X_train)
    y_pred = np.where(y_prob > 0.5, 1, 0)
    fpr, tpr, _ = roc_curve(y_train, y_prob)
    roc_auc = auc(fpr, tpr)
    precison, recall, thresholds = precision_recall_curve(y_train, y_prob)
    ap = average_precision_score(y_train, y_prob)
    if plot:
        fig, ax = plt.subplots()
        im = plt.scatter(
            recall[:-1],
            precison[:-1],
            c=thresholds,
            s=5,
            cmap="jet",
            vmax=1,
            vmin=0,
            label="Average precision (%.2f)" % ap,
        )
        clb = plt.colorbar()
        clb.ax.set_ylabel("Prediction thresholds")
        f_scores = np.linspace(0.2, 0.8, num=4)
        for i, f_score in enumerate(f_scores):
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            if i == 0:
                (l,) = plt.plot(
                    x[y >= 0], y[y >= 0], color="gray", alpha=0.5, label="iso-f1 curves"
                )
            else:
                (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.5)
            plt.annotate(
                "f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02), color="gray"
            )
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curve (PRC)")
        plt.legend(loc="lower left", scatterpoints=10)
        plt.savefig(path + "CalibrationEvents\\" + version + "\\PRC.png", dpi=300)
        plt.close()
    return model


# validate new events
def eventValidation(model, equ, d, plot=True):
    # d is after box-cox tranformation
    total_events = d.groupby("Event")["n_F"].sum().sort_values().index
    f1s = []
    ps = []
    rs = []
    baccs = []
    rfs = []
    pfs = []
    for event in total_events:
        dd = d.loc[d.Event == event, :]
        y_test, X_test = dmatrices(equ, dd, return_type="dataframe")
        y_pred = model.predict(X_test)
        # 1. predict logstic regression
        y_test = dd.loc[:, "n_F"]
        y_pred_ = pd.cut(y_pred, [-1, 0.5, 1], labels=[0, 1]).astype(float)
        # prediction score
        y_pred_ = y_pred_.groupby(y_pred_.index).max()
        y_test_ = y_test.groupby(y_test.index).max()
        y_handle = pd.concat([y_test_, y_pred_], axis=1).rename(
            columns={"n_F": "y_test", 0: "y_pred"}
        )
        y_test_ = y_handle["y_test"]
        y_pred_ = y_handle["y_pred"]
        p = precision_score(y_test_, y_pred_)
        r = recall_score(y_test_, y_pred_)
        f1 = f1_score(y_test_, y_pred_)
        bacc = balanced_accuracy_score(y_test_, y_pred_)
        f1s.append(f1)
        ps.append(p)
        rs.append(r)
        baccs.append(bacc)
        rfs.append(y_test_.sum())
        pfs.append(y_pred_.sum())
        if plot:
            TP = set(y_test_[y_test_ == 1].index) & set(y_pred_[y_pred_ == 1].index)
            FP = set(y_test_[y_test_ == 0].index) & set(y_pred_[y_pred_ == 1].index)
            FN = set(y_test_[y_test_ == 1].index) & set(y_pred_[y_pred_ == 0].index)
            TN = set(y_test_[y_test_ == 0].index) & set(y_pred_[y_pred_ == 0].index)
            fig, ax = plt.subplots(figsize=(5, 5.5))
            if len(TN) + len(TN) > 0:
                nx.draw_networkx_nodes(
                    X.nodes,
                    pos,
                    node_size=1,
                    node_color="k",
                    label="TN (%.1f%%)" % (100 * len(TN) / (len(FP) + len(TN))),
                    ax=ax,
                )
                nx.draw_networkx_nodes(
                    FP,
                    pos,
                    node_size=3,
                    node_color="orange",
                    label="FP (%.1f%%)" % (100 * len(FP) / (len(FP) + len(TN))),
                    ax=ax,
                )
            if len(TP) + len(FN) > 0:
                nx.draw_networkx_nodes(
                    FN,
                    pos,
                    node_size=3,
                    node_color="red",
                    label="FN (%.1f%%)" % (100 * len(FN) / (len(TP) + len(FN))),
                    ax=ax,
                )
                nx.draw_networkx_nodes(
                    TP,
                    pos,
                    node_size=3,
                    node_color="green",
                    label="TP (%.1f%%)" % (100 * len(TP) / (len(TP) + len(FN))),
                    ax=ax,
                )

            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            catchment = path.split("\\")[-3]
            plt.title(
                f"{catchment} - Event: %s \n P = %.2f, R = %.2f, F1 = %.2f, BAcc = %.2f"
                % (event, p, r, f1, bacc)
            )
            plt.legend(scatterpoints=3, loc="best")
            plt.tight_layout()
            plt.savefig(
                path + "CalibrationEvents\\" + version + "\\" + event + f"_{equ}.png",
                dpi=300,
            )
            y_pred.to_csv(
                path + "CalibrationEvents\\" + version + "\\" + event + f"_{equ}.csv"
            )
            plt.close()
    print("F1 for new events: %.2f" % np.mean(f1s))
    scores = pd.DataFrame(
        data=np.array([ps, rs, f1s, baccs, rfs, pfs]).T,
        columns=["p", "r", "f1", "bacc", "rf", "pf"],
        index=total_events,
    )
    scores.to_csv(
        path + "CalibrationEvents\\" + version + "\\" + "scores_" + equ + ".csv"
    )
    return f1s


#############################################################
#  Korte Woerden
############################################################

path = "J:\\LocalWorkspace\\hybridtest\\KorteWoerden_Stratiform\\"
tablepath_cal = (
    "J:\\LocalWorkspace\\hybridtest\\data_preprocessing\\eventcsv_raingen_diffuse_cal\\"
)
tablepath_val = (
    "J:\\LocalWorkspace\\hybridtest\\data_preprocessing\\eventcsv_raingen_val\\"
)
version = "v4.1"
X = read_gpickle(
    "J:\\LocalWorkspace\\hybridtest\\sewer\\sewer_graph_networkx_v4.1.gpickle"
)
pos_dict = dict(X.nodes(data="geo"))
pos = {k: pos_dict[k][:2] for k in pos_dict}
cal_eventlist = os.listdir(tablepath_cal)
val_eventlist = os.listdir(tablepath_val)
# event_list = os.listdir(tablepath)
lams = {}

# calibration events
# Events = [i for i in range(0,3)]
Events = [i for i in range(0, len(os.listdir(tablepath_cal)))]
d_C = pd.concat(
    [
        readTablemq_v4_1(X, cal_eventlist[keyword], version, tablepath_cal)
        for keyword in Events
    ]
)
# d_C = pd.concat([readTablemq_v4_1(X, event_list[keyword], version, tablepath) for keyword in Events])
d = d_C.copy()
# for v in ['NS', 'PS', 'OT','d', 'UST','DST', 'IT']:  d.loc[:,v], lams[v] = stats.boxcox(d.loc[:,v])
# ^Right now UST is the x
for v in ["NS", "PS", "OT", "d", "DST", "IT"]:
    d.loc[:, v], lams[v] = stats.boxcox(d.loc[:, v])
# validation events
# V_Events = [i for i in range(10,20)]
V_Events = [i for i in range(0, len(os.listdir(tablepath_val)))]
# V_Events = V_Events + ['T2', 'T5', 'T20', 'T100', '30May']
d_V = pd.concat(
    [
        readTablemq_v4_1(X, val_eventlist[keyword], version, tablepath_val)
        for keyword in V_Events
    ]
)
d_v = bcTransfrom(lams, d_V.copy())

#############################################################
#  Selection of Variables
############################################################
# test1  test all categories
doit = True
if doit == True:
    thres = []
    aps = []
    APs = []
    f1s = []
    F1s = []
    thresholds = [2, 5, 10, 20, 50]
    sub_thresholds = py_solution().sub_sets(thresholds)[1:]
    sub_thresholds = [s for s in sub_thresholds if len(s) >= 2]
    for sub in sub_thresholds:
        thres.append("Cat:" + ";".join([str(s) for s in sub]))
        group = pd.cut(
            d.loc[d.n_F > 0, "OT4Cat"].values, [-1] + sub, right=True, labels=False
        )
        Cat = pd.DataFrame(
            group[np.newaxis].T, columns=["Cat"], index=d.loc[d.n_F > 0, "OT4Cat"].index
        )
        Cat = Cat.groupby(Cat.index).apply(np.nanmin).to_frame(name="Cat")
        new_d = d[["n_F", "NS", "PS", "OT", "d"]].join(Cat)
        new_d.Cat = new_d.Cat.fillna(np.nanmax(group) + 1.0)
        equ = "n_F ~ C(Cat)+NS+OT+PS+d"
        # aps_ = kfoldValidation(equ, new_d); aps.append(np.mean(aps_));APs.append(aps_)
        model = modelCalibration(equ, new_d)
        new_d_v = d_v[["n_F", "NS", "PS", "OT", "d", "Event"]].join(Cat)
        new_d_v.Cat = new_d_v.Cat.fillna(np.nanmax(group) + 1.0)
        f1s_ = eventValidation(model, equ, new_d_v)
        f1s.append(np.mean(f1s_))
        F1s.append(f1s_)
    plt.figure()
    df__ = pd.DataFrame(F1s)
    df__.index = thres
    df__.T.boxplot()
    ax = plt.gca()
    ax.set_xticklabels(thres, rotation=90)
    plt.tight_layout()
# plt.show()

# modify d based on the previous test1
sub = [2, 20]
group = pd.cut(d.loc[d.n_F > 0, "OT4Cat"].values, [-1] + sub, right=True, labels=False)
Cat = pd.DataFrame(
    group[np.newaxis].T, columns=["Cat"], index=d.loc[d.n_F > 0, "OT4Cat"].index
)
Cat = Cat.groupby(Cat.index).apply(np.nanmin).to_frame(name="Cat")
new_d = d[["n_F", "NS", "PS", "OT", "d", "UST", "DST", "IT"]].join(Cat)
new_d.Cat = new_d.Cat.fillna(np.nanmax(group) + 1.0)
new_d_v = d_v[["n_F", "NS", "PS", "OT", "d", "UST", "DST", "IT", "Event"]].join(Cat)
new_d_v.Cat = new_d_v.Cat.fillna(np.nanmax(group) + 1.0)
d = new_d
d_v = new_d_v

# test2 test all possible subsets
doit = True
if doit == True:
    equs = []
    aps = []
    APs = []
    f1s = []
    F1s = []
    # old
    variables = ["d", "NS", "PS", "OT", "C(Cat)"]
    sub_variables = py_solution().sub_sets(variables)[1:]
    for sub in sub_variables:
        equ = "n_F ~ " + "+".join(sub)
        equs.append(equ)
        try:
            aps_ = kfoldValidation(equ, d)
            aps.append(np.mean(aps_))
            APs.append(aps_)
        except:
            print("Erorr with KFold validation - d-NS-PS-OT")
        model = modelCalibration(equ, d)
        f1s_ = eventValidation(model, equ, d_v)
        f1s.append(np.mean(f1s_))
        F1s.append(f1s_)
    # new
    variables = ["IT", "UST", "DST", "C(Cat)"]
    sub_variables = py_solution().sub_sets(variables)[1:]
    for sub in sub_variables:
        equ = "n_F ~ " + "+".join(sub)
        equs.append(equ)
        try:
            aps_ = kfoldValidation(equ, d)
            aps.append(np.mean(aps_))
            APs.append(aps_)
        except:
            print("Error with KFold validation - IT-UST-DST")
        model = modelCalibration(equ, d)
        f1s_ = eventValidation(model, equ, d_v)
        f1s.append(np.mean(f1s_))
        F1s.append(f1s_)

    plt.figure()
    df_ = pd.DataFrame(F1s)
    df_.index = equs
    df_.T.boxplot()
    ax = plt.gca()
    ax.set_xticklabels(equs, rotation=90)
    plt.tight_layout()
    Scores = [np.array(F1s), np.array(APs)]

############################################################
#  calibrate model based on more events - d
############################################################

d = d_C.copy()
for v in ["NS", "PS", "OT", "d", "UST", "DST", "IT"]:
    d.loc[:, v], lams[v] = stats.boxcox(d.loc[:, v])

d_v = bcTransfrom(lams, d_V.copy())

# save lams
pickle.dump(
    lams,
    open(
        path
        + "CalibrationEvents\\"
        + version
        + "\\sewer_"
        + version
        + "_Calibrated_lambda.pickle",
        "wb",
    ),
    protocol=2,
)

equs = []
aps = []
APs = []
f1s = []
F1s = []

# groups
sub = [2, 5, 20, 100]
group = pd.cut(d.loc[d.n_F > 0, "OT4Cat"].values, [-1] + sub, right=True, labels=False)
Cat = pd.DataFrame(
    group[np.newaxis].T, columns=["Cat"], index=d.loc[d.n_F > 0, "OT4Cat"].index
)
Cat = Cat.groupby(Cat.index).apply(np.nanmin).to_frame(name="Cat")

# save groups
for n in X.nodes:
    if n in Cat.index and ~np.isnan(Cat.loc[n, "Cat"]):
        X.nodes[n]["Cat"] = Cat.loc[n, "Cat"]
    else:
        X.nodes[n]["Cat"] = np.nanmax(Cat) + 1.0

cat = pd.DataFrame.from_dict(dict(X.nodes(data="Cat")), orient="index")
cat = np.nanmax(cat) - cat
cat = cat.astype(int).rename(columns={0: "Cat"})
cat.to_csv(path + "CalibrationEvents\\" + version + "\\sewer_" + version + "_Cat.csv")

for n in X.nodes:
    X.nodes[n]["Cat"] = cat.loc[n, "Cat"]

nx.write_gpickle(
    X,
    path
    + "CalibrationEvents\\"
    + version
    + "\\sewer_"
    + version
    + "_Calibrated.gpickle",
)


# equ 1
new_d = d[["n_F", "NS", "PS", "OT", "d"]].join(cat)
new_d.Cat = new_d.Cat.fillna(0)
equ = "n_F ~ C(Cat)+NS+PS+OT+d"
equs.append(equ)
# aps_ = kfoldValidation(equ, new_d); aps.append(np.mean(aps_));APs.append(aps_)
model = modelCalibration(equ, new_d)
model.summary()
new_d_v = d_v[["n_F", "NS", "PS", "OT", "d", "Event"]].join(cat)
if np.nanmax(Cat) > 0:
    new_d_v.Cat = new_d.Cat.fillna(0)
f1s_ = eventValidation(model, equ, new_d_v, plot=False)
f1s.append(np.mean(f1s_))
F1s.append(f1s_)
# save calibraiton data and model
new_d.to_csv(path + "CalibrationEvents\\" + version + f"\Calibration_{equ}.csv")
pickle.dump(
    model,
    open(
        path
        + "CalibrationEvents\\"
        + version
        + "\\sewer_"
        + version
        + "_"
        + equ
        + "_Calibrated_Statsmodel.pickle",
        "wb",
    ),
    protocol=2,
)

# equ op2
new_d = d[["n_F", "NS", "PS", "OT", "d"]].join(cat)
new_d.Cat = new_d.Cat.fillna(0)
equ = "n_F ~ C(Cat)+NS+PS+OT+d"
equs.append(equ)
# aps_ = kfoldValidation(equ, new_d); aps.append(np.mean(aps_));APs.append(aps_)
model = modelCalibration(equ, new_d)
model.summary()
new_d_v = d_v[["n_F", "NS", "PS", "OT", "d", "Event"]].join(cat)
if np.nanmax(Cat) > 0:
    new_d_v.Cat = new_d.Cat.fillna(0)
f1s_ = eventValidation(model, equ, new_d_v, plot=False)
f1s.append(np.mean(f1s_))
F1s.append(f1s_)
# save calibraiton data and modeloom
new_d.to_csv(path + "CalibrationEvents\\" + version + f"\Calibration_{equ}.csv")
pickle.dump(
    model,
    open(
        path
        + "CalibrationEvents\\"
        + version
        + "\\sewer_"
        + version
        + "_"
        + equ
        + "_Calibrated_Statsmodel.pickle",
        "wb",
    ),
    protocol=2,
)

# equ op3
# new_d = d[['n_F','NS','PS','OT']].join(cat); new_d.Cat = new_d.Cat.fillna(0)
# equ = 'n_F ~ C(Cat)+NS+PS+OT';  equs.append(equ)
# #aps_ = kfoldValidation(equ, new_d); aps.append(np.mean(aps_));APs.append(aps_)
# model = modelCalibration(equ, new_d); model.summary()
# new_d_v = d_v[['n_F','NS','PS','OT', 'd', 'Event']].join(cat); new_d_v.Cat = new_d.Cat.fillna(0)
# f1s_ = eventValidation(model, equ, new_d_v, plot = False ); f1s.append(np.mean(f1s_));F1s.append(f1s_)
# # save calibraiton data and modeloom
# new_d.to_csv(path + 'CalibrationEvents\\' + version + f'\Calibration_{equ}.csv')
# pickle.dump(model, open(path + "CalibrationEvents\\" + version +  "\\sewer_" + version +  "_" + equ + "_Calibrated_Statsmodel.pickle", "wb"), protocol=2 )

# equ op4
new_d = d[["n_F", "OT", "d"]].join(cat)
new_d.Cat = new_d.Cat.fillna(0)
equ = "n_F ~ C(Cat)+OT+d"
equs.append(equ)
# aps_ = kfoldValidation(equ, new_d); aps.append(np.mean(aps_));APs.append(aps_)
model = modelCalibration(equ, new_d)
model.summary()
new_d_v = d_v[["n_F", "OT", "d", "Event"]].join(cat)
if np.nanmax(Cat) > 0:
    new_d_v.Cat = new_d.Cat.fillna(0)
f1s_ = eventValidation(model, equ, new_d_v, plot=False)
f1s.append(np.mean(f1s_))
F1s.append(f1s_)
# save calibraiton data and modeloom
new_d.to_csv(path + "CalibrationEvents\\" + version + f"\Calibration_{equ}.csv")
pickle.dump(
    model,
    open(
        path
        + "CalibrationEvents\\"
        + version
        + "\\sewer_"
        + version
        + "_"
        + equ
        + "_Calibrated_Statsmodel.pickle",
        "wb",
    ),
    protocol=2,
)

# equ op6
new_d = d[["n_F", "NS", "OT", "PS"]].join(cat)
new_d.Cat = new_d.Cat.fillna(0)
equ = "n_F ~ C(Cat)+NS+OT+PS"
equs.append(equ)
# aps_ = kfoldValidation(equ, new_d); aps.append(np.mean(aps_));APs.append(aps_)
model = modelCalibration(equ, new_d)
model.summary()
new_d_v = d_v[["n_F", "NS", "OT", "PS", "Event"]].join(cat)
if np.nanmax(Cat) > 0:
    new_d_v.Cat = new_d.Cat.fillna(0)
f1s_ = eventValidation(model, equ, new_d_v, plot=False)
f1s.append(np.mean(f1s_))
F1s.append(f1s_)
# save calibraiton data and modeloom
new_d.to_csv(path + "CalibrationEvents\\" + version + f"\Calibration_{equ}.csv")
pickle.dump(
    model,
    open(
        path
        + "CalibrationEvents\\"
        + version
        + "\\sewer_"
        + version
        + "_"
        + equ
        + "_Calibrated_Statsmodel.pickle",
        "wb",
    ),
    protocol=2,
)

# equ op7
new_d = d[["n_F", "NS", "PS"]].join(cat)
new_d.Cat = new_d.Cat.fillna(0)
equ = "n_F ~ C(Cat)+NS+PS"
equs.append(equ)
# aps_ = kfoldValidation(equ, new_d); aps.append(np.mean(aps_));APs.append(aps_)
model = modelCalibration(equ, new_d)
model.summary()
new_d_v = d_v[["n_F", "NS", "PS", "Event"]].join(cat)
if np.nanmax(Cat) > 0:
    new_d_v.Cat = new_d.Cat.fillna(0)
f1s_ = eventValidation(model, equ, new_d_v, plot=False)
f1s.append(np.mean(f1s_))
F1s.append(f1s_)
# save calibraiton data and modeloom
new_d.to_csv(path + "CalibrationEvents\\" + version + f"\Calibration_{equ}.csv")
pickle.dump(
    model,
    open(
        path
        + "CalibrationEvents\\"
        + version
        + "\\sewer_"
        + version
        + "_"
        + equ
        + "_Calibrated_Statsmodel.pickle",
        "wb",
    ),
    protocol=2,
)

# equ op8
new_d = d[["n_F", "NS", "PS", "d"]].join(cat)
new_d.Cat = new_d.Cat.fillna(0)
equ = "n_F ~ C(Cat)+NS+PS+d"
equs.append(equ)
# aps_ = kfoldValidation(equ, new_d); aps.append(np.mean(aps_));APs.append(aps_)
model = modelCalibration(equ, new_d)
model.summary()
new_d_v = d_v[["n_F", "NS", "PS", "d", "Event"]].join(cat)
if np.nanmax(Cat) > 0:
    new_d_v.Cat = new_d.Cat.fillna(0)
f1s_ = eventValidation(model, equ, new_d_v, plot=False)
f1s.append(np.mean(f1s_))
F1s.append(f1s_)
# save calibraiton data and modeloom
new_d.to_csv(path + "CalibrationEvents\\" + version + f"\Calibration_{equ}.csv")
pickle.dump(
    model,
    open(
        path
        + "CalibrationEvents\\"
        + version
        + "\\sewer_"
        + version
        + "_"
        + equ
        + "_Calibrated_Statsmodel.pickle",
        "wb",
    ),
    protocol=2,
)


# equ2
new_d = d[["n_F", "IT", "UST", "DST", "d"]].join(cat)
new_d.Cat = new_d.Cat.fillna(0)
equ = "n_F ~ C(Cat)+IT+UST+DST"
equs.append(equ)
# aps_ = kfoldValidation(equ, new_d); aps.append(np.mean(aps_));APs.append(aps_)
model = modelCalibration(equ, new_d)
model.summary()
new_d_v = d_v[["n_F", "IT", "UST", "DST", "d", "Event"]].join(cat)
if np.nanmax(Cat) > 0:
    new_d_v.Cat = new_d.Cat.fillna(0)
f1s_ = eventValidation(model, equ, new_d_v, plot=False)
f1s.append(np.mean(f1s_))
F1s.append(f1s_)
# save calibraiton data and model
new_d.to_csv(path + "CalibrationEvents\\" + version + f"\Calibration_{equ}.csv")
pickle.dump(
    model,
    open(
        path
        + "CalibrationEvents\\"
        + version
        + "\\sewer_"
        + version
        + "_"
        + equ
        + "_Calibrated_Statsmodel.pickle",
        "wb",
    ),
    protocol=2,
)


for e, f in zip(equs, F1s):
    plt.plot(f, label=e)

plt.legend()
plt.grid(which="both")
plt.show()


###################################################
#     VIF scores
###################################################


# VIF scores
doit = True
if doit == True:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif = pd.DataFrame()
    X_train = d.loc[:, ["IT", "DST", "UST"]]
    vif["VIF Factor"] = [
        variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])
    ]
    vif["features"] = X_train.columns
    print(vif)


# VIF scores
doit = True
if doit == True:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif = pd.DataFrame()
    X_train = d.loc[:, ["PS", "NS", "OT", "d"]]
    vif["VIF Factor"] = [
        variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])
    ]
    vif["features"] = X_train.columns
    print(vif)
