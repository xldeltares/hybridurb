"""
UPDATE MARKER
Most recent modification on 20190731 (after revision)
(adapted to python3)

@author: Xiaohan Li

version: python 3

packages:

functions:
frange(start, stop, step)
I_idf(T, tc)
T_idf(intensity,duration)
wetPipe(condshape, condwidth, condheight, sedimentdepth = None)
V_CW(D,Sf,Ks)
Re_R(v, D)
f_DW(Re, Ks, D)
Hf_DW(f, v, L, D)

createSubgraph(G, largest_wcc)

breakLoop(X)
reverseOutfall(X, levels = None)
splitTree(X, weight_var)
recoverLeaf(X2, X1)
addOutfall(X)
----- DAG -----
calcCost(X)
multi2uniDiGraph(X)
makeDAG(X)
calcDAGTc(X, method = 'CW_cap')
calcDAGStaticFI_reduced(X)
calcDAGStaticFI(X)
fillNegCapacity_nodes(X, param = 'Qcap_m3ds')
----- non DAG -----
calcConcentrationTime(X)
calcStaticFI_reduced(X)
calcStaticFI_path(X)

linearReservior(R_df, A)
calcQ(X, sim_s_RR)
calcOT(X, sim_n_RR)
calcF(X, sim_n_FV)
summarizeEvent(X, sim_n_RO, sim_nc_RO, sim_nc_Q, sim_n_F, sim_n_FV, sim_o_T = None)

calcWithinGroupsVariance(variable, groupvariable)
calcBetweenGroupsVariance(variable, groupvariable)
calcSeparations(variables, groupvariable)
calcWithinGroupsCovariance(variable1, variable2, groupvariable)
calcBetweenGroupsCovariance(variable1, variable2, groupvariable)

drawLoop(cycle, graph)
drawGraphTree(X)

ensure_dir(directory)
readOutfallLevels(outfall_level_filename, hours)
"""


########################################################
# python version selection
########################################################

import sys

python_version = sys.version_info[0]
import collections
import copy
import datetime

# tools
import os
import pickle
import sys
import time
from collections import defaultdict
from operator import itemgetter
from random import randint

import geopandas as gpd
import matplotlib._pylab_helpers

# ploting
# import graphviz
# import pygraphviz
import matplotlib.pyplot as plt

########################################################
# Packages
########################################################
# cores
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

# computations
import sympy
from matplotlib.cm import get_cmap
from networkx.drawing.nx_agraph import graphviz_layout

# gis
from osgeo import gdal, ogr, osr
from scipy.signal import convolve
from sklearn.preprocessing import scale

########################################################
# functions
########################################################


########################################################
# computation tools
########################################################


def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


def I_idf(T, tc):
    # solve intensity using numerical solver
    # tc in minutes, T in jears
    # D in days
    T = float(T)
    D = tc / 60.0 / 24.0
    beta_a = 10.0 ** (-0.05 - 0.58 * np.log10(D))
    beta_b = 10.0 ** (-0.55 - 0.58 * np.log10(D))
    p_a = 10.0 ** (-1.35 - 0.58 * np.log10(D))
    i_0 = 10.0 ** (-0.05 - 0.58 * np.log10(D))
    p_1 = 10.0 ** (-1.12 - 0.50 * np.log10(D))
    n = 27.0
    m = 1.0 + 54.0 * p_1 / (p_1 - p_a)
    if D < 1.0:
        C = 0.93
    else:
        C = 1.0
    x = sympy.Symbol("x")
    func = T - n / (
        m
        * (
            p_a * sympy.exp((i_0 - x * C) / beta_a)
            + (1.0 - p_a) * sympy.exp((i_0 - x * C) / beta_b)
        )
    )
    if tc <= 30:
        return sympy.nsolve(func, x, 150.0)
    elif tc <= 60:
        return sympy.nsolve(func, x, 50.0)
    elif tc <= 120:
        return sympy.nsolve(func, x, 30.0)
    else:
        return sympy.nsolve(func, x, 10.0)


# calculate the return year
# New 04/12/2018
def T_idf(intensity, duration):
    # input: intensity in mm/hr
    #        Duration in min
    # output:T in year
    i = float(intensity)
    D = float(duration) / 1440.0
    beta_a = 10.0 ** (-0.05 - 0.58 * np.log10(D))
    beta_b = 10.0 ** (-0.55 - 0.58 * np.log10(D))
    p_a = 10.0 ** (-1.35 - 0.58 * np.log10(D))
    i_0 = 10.0 ** (-0.05 - 0.58 * np.log10(D))
    p_1 = 10.0 ** (-1.12 - 0.5 * np.log10(D))
    n = 27.0
    m = 1.0 + 54.0 * p_1 / (p_1 - p_a)
    C = 0.93
    T = n / (
        m
        * (
            p_a * np.exp((i_0 - i * C) / beta_a)
            + (1 - p_a) * np.exp((i_0 - i * C) / beta_b)
        )
    )
    return T


# calculated T_idf using interpolation --> much faster
# NEW! 04/09/2019
# T_func_h = T_func()
# T_result = T_func_h.calc([[i, t]])
class T_func:
    def __init__(self):
        from scipy import interpolate

        i = np.arange(1, 200, 0.25)  # intenisty
        d = np.arange(5, 300, 5)  # duraiton
        ii, dd = np.meshgrid(i, d)
        tt = np.array(list(map(T_idf, ii.ravel(), dd.ravel()))).reshape(ii.shape)
        self.f = interpolate.NearestNDInterpolator(
            np.vstack([ii.ravel(), dd.ravel()]).T, tt.ravel()
        )

    def calc(self, data):
        # data should be np array of [i, d], or [[i,d], [i,d]]
        # for non zeros only
        if data.ndim == 1:
            data_T = np.nan
            try:
                data_T = self.f(data)
            except:
                pass
        else:
            data_T = (
                np.zeros(
                    data.shape[0],
                )
                * np.nan
            )
            for i, d in enumerate(data):
                try:
                    data_T[i] = self.f(d)
                except:
                    pass
        data_T[data_T < 0] = 0
        return data_T


# updated 27/11/2018: with sediment computation
# updated 13/08/2019: with irregular shape defination
def wetPipe(condshape, condwidth, condheight, sedimentdepth=None):
    shapes = ["RECT", "OVAL", "EGG", "ARCH", "CIRC", "OT", "ORECT"]
    if condshape.startswith("OT"):
        # OTy:x (error in infoworks) x = theta*y
        theta = float(int(condshape[4])) / float(int(condshape[2]))
        condshape = "OT"
    elif condshape.startswith("ORECT"):
        # ORECT:treat as OT
        theta = float(1)
        condshape = "OT"
    elif condshape.startswith("EGG"):
        # radian of theta , the small angle between small and big circle
        theta = np.arcsin((condwidth - condheight / 2.0) / (condheight / 2.0))
        condshape = "EGG"
    elif condshape in shapes:
        # conduit is in regular shape
        theta = 0
    else:
        # conduit is in irregular shape, simply as circ, using width
        theta = 0
        condshape = "CIRC"
    Area = {
        "RECT": lambda w, h, theta: w * h,
        "OVAL": lambda w, h, theta: np.pi * (w / 2.0) ** 2 + w * (h - w),
        "EGG": lambda w, h, theta: 1.0 / 2.0 * (2.0 * theta + np.pi) * (w / 2.0) ** 2
        + 1.0 / 2.0 * (-2.0 * theta + np.pi) * ((h - w) / 2.0) ** 2
        + h / 2.0 * (w - h / 2.0) / np.tan(theta),
        "ARCH": lambda w, h, theta: w * (h - w / 2.0)
        + 1.0 / 2.0 * np.pi * (w / 2.0) ** 2,
        "CIRC": lambda w, h, theta: np.pi * (w / 2.0) ** 2,
        "OT": lambda w, h, theta: (2.0 * w + 2.0 * theta * h) * h / 2,
    }
    Perimeter = {
        "RECT": lambda w, h, theta: 2.0 * w + 2.0 * h,
        "OVAL": lambda w, h, theta: 2.0 * np.pi * (w / 2.0) + 2.0 * (h - w),
        "EGG": lambda w, h, theta: (2.0 * theta + np.pi) * (w / 2.0)
        + (-2.0 * theta + np.pi) * ((h - w) / 2.0)
        + 2 * (w - h / 2.0) / np.tan(theta),
        "ARCH": lambda w, h, theta: w + 2.0 * (h - w / 2.0) + np.pi * (w / 2.0),
        "CIRC": lambda w, h, theta: 2 * np.pi * (w / 2.0),
        "OT": lambda w, h, theta: w + 2.0 * h * (1.0 + theta**2.0) ** (1.0 / 2.0),
    }
    minussediment_Area = {
        "RECT": lambda w, h, s, theta: w * (h - s),
        "OVAL": lambda w, h, s, theta: Area["OVAL"](w, h, theta)
        - arc_area(w / 2.0, arc_beta(w / 2.0, s))
        if s <= w / 2.0
        else (
            arc_area(w / 2.0, np.pi) + (h - w / 2.0 - s) * w
            if s <= (h - w / 2.0)
            else arc_area(w / 2.0, arc_beta(w / 2.0, h - s))
        ),
        "EGG": lambda w, h, s, theta: Area["EGG"](w, h, theta)
        - arc_area((h - w) / 2.0, arc_beta((h - w) / 2.0, s))
        if s <= arc_height((h - w) / 2.0, theta * 2.0)
        else (
            Area["EGG"](w, h, theta)
            - arc_area((h - w) / 2.0, 2 * theta)
            - egg_trap_area(w, h, s, theta)
            if s <= h - arc_height(w / 2.0, 2.0 * np.pi - theta * 2.0)
            else arc_area(w / 2.0, arc_beta(w / 2.0, h - s))
        ),
        "ARCH": lambda w, h, s, theta: arc_area(w / 2.0, np.pi) + (h - w / 2.0 - s) * w
        if s <= h - w / 2.0
        else arc_area(w / 2.0, arc_beta(w / 2.0, h - s)),
        "CIRC": lambda w, h, s, theta: arc_area(w / 2.0, 2.0 * np.pi)
        - arc_area(w / 2.0, arc_beta(w / 2.0, s)),
        "OT": lambda w, h, s, theta: (2.0 * w + 2.0 * theta * h) * h / 2
        - (2.0 * w + 2.0 * theta * s) * s / 2.0,
    }
    minussediment_Perimeter = {
        "RECT": lambda w, h, s, theta: 2.0 * w + 2.0 * (h - s),
        "OVAL": lambda w, h, s, theta: Perimeter["OVAL"](w, h, theta)
        - arc_arc(w / 2.0, arc_beta(w / 2.0, s))
        + arc_chord(w / 2.0, arc_beta(w / 2.0, s))
        if s <= w / 2.0
        else (
            arc_arc(w / 2.0, np.pi) + 2 * (h - w / 2.0 - s) + w
            if s <= (h - w / 2.0)
            else arc_arc(w / 2.0, arc_beta(w / 2.0, h - s))
            + arc_chord(w / 2.0, arc_beta(w / 2.0, h - s))
        ),
        "EGG": lambda w, h, s, theta: Perimeter["EGG"](w, h, theta)
        - arc_arc((h - w) / 2.0, arc_beta((h - w) / 2.0, s))
        + arc_chord((h - w) / 2.0, arc_beta((h - w) / 2.0, s))
        if s <= arc_height((h - w) / 2.0, theta * 2.0)
        else (
            arc_arc(w / 2.0, 2.0 * np.pi - 2.0 * theta)
            + egg_trap_peri(w, h, s, theta)
            - arc_chord(w / 2.0, 2.0 * np.pi - 2.0 * theta)
            if s <= arc_height(w / 2.0, 2.0 * np.pi - theta * 2.0)
            else arc_arc(w / 2.0, arc_beta(w / 2.0, h - s))
        ),
        "ARCH": lambda w, h, s, theta: arc_arc(w / 2.0, np.pi)
        + 2.0 * (h - w / 2.0 - s)
        + w
        if s <= h - w / 2.0
        else arc_arc(w / 2.0, arc_beta(w / 2.0, h - s))
        + arc_chord(w / 2.0, arc_beta(w / 2.0, h - s)),
        "CIRC": lambda w, h, s, theta: arc_arc(
            w / 2.0, 2.0 * np.pi - arc_beta(w / 2.0, s)
        )
        + arc_chord(w / 2.0, arc_beta(w / 2.0, s)),
        "OT": lambda w, h, s, theta: w
        + 2.0 * s * theta
        + 2.0 * (h - s) * (1.0 + theta**2.0) ** (1.0 / 2.0),
    }
    arc_beta = lambda r, d: 2 * np.arccos((r - d) / r)  # d is the sediment height
    arc_height = lambda r, beta: r * (1 - np.cos(beta / 2.0))
    arc_area = lambda r, beta: 1.0 / 2.0 * r**2 * (beta - np.sin(beta))
    arc_arc = lambda r, beta: beta * r
    arc_chord = lambda r, beta: 2 * r * np.sin(beta / 2)
    egg_trap_area = lambda w, h, s, theta: (
        arc_chord((h - w) / 2.0, 2.0 * theta)
        + egg_b(
            arc_chord((h - w) / 2.0, 2.0 * theta),
            (s - arc_height((h - w) / 2.0, theta * 2.0)),
            theta,
        )
    ) * (s - arc_height((h - w) / 2.0, theta * 2.0))
    egg_trap_peri = (
        lambda w, h, s, theta: (
            arc_chord((h - w) / 2.0, 2.0 * theta)
            + egg_b(
                arc_chord((h - w) / 2.0, 2.0 * theta),
                (s - arc_height((h - w) / 2.0, theta * 2.0)),
                theta,
            )
            + (s - arc_height((h - w) / 2.0, theta * 2.0)) / np.sin(theta)
        )
        * 2
    )
    egg_b = lambda a, ds, theta: (a * np.tan(theta) + ds) / np.tan(theta)
    if not sedimentdepth or sedimentdepth == 0:
        A, P = (
            Area[condshape](condwidth, condheight, theta),
            Perimeter[condshape](condwidth, condheight, theta),
        )
    else:
        A, P = (
            minussediment_Area[condshape](condwidth, condheight, sedimentdepth, theta),
            minussediment_Perimeter[condshape](
                condwidth, condheight, sedimentdepth, theta
            ),
        )
    return A, P


def V_CW(D, Sf, Ks):
    # D Hydraulic diameter calculated from 4A/P
    # Sf gradient [m/m]
    # Ks roughness coeffcient [mm]
    Sf_neg = False
    if Sf < 0:
        Sf = np.absolute(Sf)
        Sf_neg = True
    velocity = (
        -2.0
        * (2.0 * 9.807 * D * Sf) ** (1.0 / 2.0)
        * np.log10(
            Ks * (10.0 ** (-3.0)) / (3.7 * D)
            + (2.51 * 10.0 ** (-6.0)) / (D * (2.0 * 9.807 * D * Sf) ** (1.0 / 2.0))
        )
    )
    if Sf_neg:
        velocity = -1.0 * velocity
    return velocity


def Re_R(v, D):
    # D: hydraulic diameter calculated from 4A/P (wetPipe) (m)
    # v: veolocty (m/s)
    # viscosity: 20dC water: 10**(-6)
    Re = float(v) * float(D) / 10 ** (-6)
    return Re


def f_DW(Re, Ks, D):
    # D: hydraulic diameter calculated from 4A/P (wetPipe) (m)
    # Ks roughness coeffcient [mm]
    # Re: Reynolds number [-]
    D = float(D)
    Ks = float(Ks) * (10.0 ** (-3.0))
    Re = float(Re)
    f = 0.25 / (np.log10(Ks / (3.7 * D) + 5.74 / (Re) ** 0.9)) ** 2
    return f


def Hf_DW(f, v, L, D):
    # f: friction factor [m]
    # v: velocty [m/s]
    # L: Length of pipe [m]
    # D: hydaulic diamater 4A/P [m]
    # g: gravity 9.807 [m/s2]
    f = float(f)
    v = float(v)
    L = float(L)
    D = float(D)
    Hf = f * (L / D) * (v**2 / (2 * 9.807))
    return Hf


########################################################
# graph supplimentaroy tools
########################################################


# creat subgraph instead of providing subgraph view
def createSubgraph(G, largest_wcc):
    SG = G.__class__()
    SG.add_nodes_from((n, G.nodes[n]) for n in largest_wcc)
    try:
        SG.add_edges_from(
            (n, nbr, key, d)
            for n, nbrs in G.adj.items()
            if n in largest_wcc
            for nbr, keydict in nbrs.items()
            if nbr in largest_wcc
            for key, d in keydict.items()
        )
    except:
        SG.add_edges_from(
            (n, nbr, d)
            for n, nbrs in G.adj.items()
            if n in largest_wcc
            for nbr, d in nbrs.items()
            if nbr in largest_wcc
        )
    SG.graph.update(G.graph)
    return SG


########################################################
# sewer optimisation tools
########################################################


# correct loop system
def breakLoop(X):
    # X is a handel and X_c is the corrected graph
    X_c = X.copy()
    try:
        # find cycle
        loop = list(nx.find_cycle(X, orientation="original"))
        # find the loop and remove untile there is no loop anymore
        while loop:
            loop_nodes = list(set([j for i in map(list, loop) for j in i if j != 0]))
            print("\nDETECTED: loop of size " + str(len(loop)))
            # connection in and out of loop system
            in_nodes = [nn for nn in loop_nodes if len([i for i in X.in_edges(nn)]) > 1]
            out_nodes = [
                nn for nn in loop_nodes if len([i for i in X.out_edges(nn)]) > 1
            ]
            critical_nodes = in_nodes + out_nodes
            # check if self loop
            if len(loop_nodes) == 2:
                us_node = in_nodes[0]
                ds_node = out_nodes[0]
                X.remove_edge(ds_node, us_node)
                X_c.remove_edge(ds_node, us_node)
                print("SELF-LOOP: deletected negative flow path")
            else:
                # find the out nodes -> critical nodes that with a elevation break
                for o_n in out_nodes:
                    try:
                        l = {
                            c_n: nx.shortest_path_length(X, source=o_n, target=c_n)
                            for c_n in critical_nodes
                            if c_n != o_n and nx.has_path(X, source=o_n, target=c_n)
                        }
                        l = sorted(l.items(), key=itemgetter(1))
                        i_n = l[0][0]
                        # a list of nodes from o_n to i_n
                        p = nx.shortest_path(X, o_n, i_n)
                        # a list of elevation gain from o_n to i_n
                        pp = [
                            X.nodes[o_n]["chambfloor_m"] - X.nodes[nn]["chambfloor_m"]
                            for nn in p
                        ]
                        # pp = nx.dijkstra_path_length(sub_network, o_n, i_n, \
                        #                     weight = lambda u, v, e: e[0]['ds_invert_m'] - e[0]['us_invert_m'])
                        # the first index where a peak exisit between o_n -> i_n
                        point = [
                            i
                            for i in range(1, len(p))
                            if pp[i] < 0 and pp[i] - pp[i - 1] <= 0
                        ]
                        if len(out_nodes) == 1:
                            point = range(1, len(p))
                        try:
                            point = np.where(np.diff(point) != 1)[0][0]
                        except IndexError:
                            point = max(point)
                        # reverse pipe until point
                        for i in range(point):
                            us_node = p[i]
                            ds_node = p[i + 1]
                            attrs = X[us_node][ds_node]
                            # principle: do not revert hydraulic strucures
                            if attrs[0]["type"].lower() != "cond":
                                print(
                                    "\tSKIP: Break point is "
                                    + attrs[0]["type"].lower()
                                    + " for "
                                    + us_node
                                    + "-->"
                                    + ds_node
                                    + ", go to next out_node..."
                                )
                                break
                            # add new ds-> us, remove old us->ds
                            X.add_edges_from([(ds_node, us_node, attrs[0])])
                            X_c.add_edges_from([(ds_node, us_node, attrs[0])])
                            X.remove_edge(us_node, ds_node)
                            X_c.remove_edge(us_node, ds_node)
                            print(
                                "REVERSE: Break point found: "
                                + us_node
                                + "-->"
                                + ds_node
                                + ", go to next link..."
                            )
                    except IndexError:
                        print("\tSKIP: No break point found, go the next o_n...")
                    except ValueError:
                        print("\tSKIP: No path found anymore, go the next o_n...")
                    except nx.NetworkXNoPath:
                        print("\tSKIP: No path found anymore, go the next o_n...")
                    except:
                        raise
            # draw loop
            loop_network = X.subgraph(loop_nodes)
            pos = nx.get_node_attributes(loop_network, "geo")
            pos_xy = {n: pos[n][:2] for n in pos}
            pos_z = {n: n + ";\n " + str(pos[n][2]) for n in pos}
            plt.figure(figsize=(20, 10))
            nx.draw(loop_network, pos=pos_xy, labels=pos_z)
            labels = nx.get_edge_attributes(loop_network, "type")
            labels = {(e[0], e[1]): labels[e] for e in labels}
            nx.draw_networkx_edge_labels(loop_network, pos_xy, edge_labels=labels)
            # check if failed to correct
            if list(nx.find_cycle(X, orientation="original")) == loop:
                X.remove_edges_from(loop)
                print("DELETE: Loop cannot be corrected, deleting from network...")
                pass
            # detect next loop
            loop = list(nx.find_cycle(X, orientation="original"))
    except nx.exception.NetworkXNoCycle:
        print("\nSUCCESS: NO CYCLES ANYMORE\n")
    return X_c


# reverse outfall links
# Update 26/11/2018
# Update 12/02/2019 consideres crest level
# add consideration of outfall river levels (mean levels over a period of time - 6 hous)
# update 29/08/2019
# add water level to the link connecting to outfall (ds_invert_m)
# update 07/09/2019
# update: only reverse if the chambfloor is lower than water level
def reverseOutfall(X, levels=None):
    # if levels = None, then automatically correct all levels that are not strucure
    # if levels = dict(), then correct according to level gradient
    outfalls = [x for x in X.nodes if X.nodes[x]["type"].lower() in ["outfall"]]
    for o in outfalls:
        try:
            if (
                levels[o] and X.nodes[o]["chambfloor_m"] < levels[o]
            ):  # NEW! only reverse if the chambfloor is lower than water level
                X.nodes[o]["chambfloor_m"] = levels[o]
                print("ADD: Watrer level to outfall: " + o)
        except:
            pass
        if len(X.in_edges(o)) < 1 and len(X.out_edges(o)) > 0:
            for e in list(X.out_edges(o)):
                us_node, ds_node = e
                attrs = X[us_node][ds_node]
                if (
                    attrs[0]["type"] == "cond"
                    and X.nodes[us_node]["chambfloor_m"]
                    <= X.nodes[ds_node]["chambfloor_m"]
                ):
                    X.add_edges_from([(ds_node, us_node, attrs[0])])
                    X.remove_edge(us_node, ds_node)
                    print(
                        "REVERSE: "
                        + us_node
                        + "-->"
                        + ds_node
                        + " (gradient error), go to next link..."
                    )
        try:
            edges = list(X.in_edges(o, data=True))
            for edge in edges:
                if edge[2]["type"] == "weir":
                    X.nodes[o]["chambfloor_m"] = edge[2]["ds_invert_m"]
                    print("ADD: crest level added: " + o)
                if (
                    edge[2]["type"] == "cond"
                    and X.nodes[o]["chambfloor_m"] > edge[2]["ds_invert_m"]
                ):
                    X[edge[0]][edge[1]][0].update(
                        {"ds_invert_m": X.nodes[o]["chambfloor_m"]}
                    )
                    print("ADD: Higher level added to discharge pipe: " + o)
        except:
            pass
    return X


# split intervined tree branches !NOTE: using weight variable
def splitTree(X, weight_var):
    XX = X.reverse()
    outfalls = [x for x in X.nodes if X.nodes[x]["type"].lower() in ["outfall"]]
    X_new = nx.MultiDiGraph()
    for o1 in outfalls:
        sn1 = createSubgraph(X, nx.dfs_tree(XX, o1).nodes)
        # ap1 = list(nx.articulation_points(nx.Graph(sn1)))
        print(o1 + " inspecting:")
        for o2 in outfalls:
            if o1 == o2:
                continue
            sn2 = createSubgraph(X, nx.dfs_tree(XX, o2).nodes)
            # ap2 = list(nx.articulation_points(nx.Graph(sn2)))
            # check if duplicates
            shared = [n for n in sn1.nodes if n in sn2.nodes]
            if len(shared) == 0:
                continue
            else:
                print("\tVS " + o2 + ": Am I greedy? :S")
                # the most donwstream node
                s = [
                    n
                    for n in X.subgraph(shared)
                    if X.subgraph(shared).out_degree(n) == 0
                ][0]
                # path length
                l1 = [
                    nx.dijkstra_path_length(X, source=s, target=o1, weight=weight_var)
                    for s in shared
                ]
                l2 = [
                    nx.dijkstra_path_length(X, source=s, target=o2, weight=weight_var)
                    for s in shared
                ]
                # remove all upstream nodes (not just the shared ones) from the longer one
                if min(l1) > min(l2):
                    sn1.remove_nodes_from(shared)
                    print("\t\tNOT MINE: I am giving that away! :)  ")
                elif min(l2) > min(l1):
                    print("\t\tYES MINE: I am keeping them! :)  ")
                    continue
                else:
                    raise Exception("no shorter path is found!!")
        # then compose
        print(o1 + " finished! joing my new family: X_new\n")
        X_new = nx.compose_all([X_new, sn1])
    return X_new


# recover left over leaves !NOTE: may detroy the previous step splittree
# UPDATE 30/11/2018
# distinct between local depression and leaf stem
# correct for local depression (isolated component)
def recoverLeaf(X2, X1):
    # X2 is the new figure, X1 is the old figure
    # object is to recover X1 back to X2
    # Method: recover the stem back to main graph and correct the local depression to drain to stem
    def reverseEdgeDict(d):
        d_new = d.copy()
        d_new.update(
            {
                "gradient_mdm": (-d["gradient_mdm"] if d["gradient_mdm"] else 0),
                "capacity_m3ds": (-d["capacity_m3ds"] if d["capacity_m3ds"] else None),
                "us_invert_m": d["ds_invert_m"],
                "ds_invert_m": d["us_invert_m"],
            }
        )
        return d_new

    # gone nodes
    gone_nodes = [n for n in X1.nodes() if n not in X2.nodes()]
    gone_graph = createSubgraph(X1, gone_nodes)
    # draw gone nodes
    G = nx.DiGraph(gone_graph)
    pos = graphviz_layout(G, prog="dot", args="")
    plt.figure()
    nx.draw(G, pos, node_size=20, alpha=0.5, node_color="blue", with_labels=False)
    nx.draw(
        G.subgraph([n for n in G.nodes if len(G.out_edges(n)) == 0]),
        pos,
        node_size=20,
        alpha=0.5,
        node_color="red",
        with_labels=True,
    )
    X3 = nx.compose(X2, nx.MultiDiGraph(G))
    print("left over partitions:" + str(nx.number_connected_components(nx.Graph(G))))
    print("original components: " + str(nx.number_connected_components(nx.Graph(X2))))
    print("composed components: " + str(nx.number_connected_components(nx.Graph(X3))))
    # correct X3
    X = X1.copy()
    X_new = X3.copy()
    outfalls = [x for x in X.nodes if X.nodes[x]["type"].lower() in ["outfall"]]
    # leaves by leaves
    for c in nx.connected_components(nx.Graph(G)):
        SG = G.subgraph(c)
        # identify wrongly connected leaves and speperate into stem case and local depression cases
        ss = [
            n
            for n in SG.nodes
            if len(SG.out_edges(n)) == 0 and SG.degree(n) != X.degree(n)
        ]
        if len(ss) >= 1:
            stem = ss[0]
            depr = [
                n
                for n in SG.nodes
                if len(SG.out_edges(n)) == 0
                and SG.degree(n) == X.degree(n)
                and n != stem
            ]
        else:
            ss = [n for n in SG.nodes if SG.degree(n) != X.degree(n)]
            stem = ss[0]
            depr = [
                n
                for n in SG.nodes
                if len(SG.out_edges(n)) == 0
                and SG.degree(n) == X.degree(n)
                and n != stem
            ]
        # start correcting
        ns = stem
        print("Adding " + ns)
        if len(SG.out_edges(ns)) != len(X.out_edges(ns)):
            mismatch_es = [e for e in X.out_edges(ns) if e not in SG.out_edges(ns)]
            for me in mismatch_es:
                us, ds = me
                d = dict(X[us][ds])[0]
                if me not in X_new.edges():
                    print("\tout edge does not exist, adding...")
                    print("\t\tOut-edge added: " + us + " to" + ds)
                    X_new.add_edges_from([(us, ds, d)])
        elif len(SG.in_edges(ns)) != len(X.in_edges(ns)):
            mismatch_es = [e for e in X.in_edges(ns) if e not in SG.in_edges(ns)]
            for me in mismatch_es:
                us, ds = me
                d = dict(X[us][ds])[0]
                if me not in X_new.edges():
                    print("\tin edge does not exist, adding....")
                    print("\t\t Reversed in-edge added:" + ds + " to" + us)
                    d = reverseEdgeDict(d)
                    X_new.add_edges_from([(ds, us, d)])
                    # check if stem connext to outfall
                    if np.any([nx.has_path(X_new, ns, o) for o in outfalls]):
                        print("\t" + ns + " is sucessfully added to graph!")
                        # fix depressions
                        for nd in depr:
                            nds_path = nx.shortest_path(
                                nx.Graph(SG), source=nd, target=ns
                            )
                            for e in zip(nds_path[:-1], nds_path[1:]):
                                if (
                                    e not in X_new.edges()
                                    and (e[1], e[0]) in X_new.edges()
                                ):
                                    # the defination of us ds is the corrected edge direction, not the original
                                    us, ds = e
                                    d = dict(X_new[ds][us])[0]
                                    d = reverseEdgeDict(d)
                                    X_new.add_edges_from([(us, ds, d)])
                                    X_new.remove_edge(ds, us)
                                    print(
                                        "\t"
                                        + nd
                                        + "(depressions) is sucessfully corrected!"
                                    )
                    else:
                        raise (ValueError(ns + " can still not connect to path!"))
    # chekc if all nodes in gone nodes now connects to an outfall
    if np.all(
        [np.any([nx.has_path(X_new, ns, o) for o in outfalls]) for n in gone_nodes]
    ):
        print(
            "ALL NODES FOUND OUTFALL! After recovered components: "
            + str(nx.number_connected_components(nx.Graph(X_new)))
            + "\n"
        )
    return X_new


# add outfall information
def addOutfall(X):
    outfalls = [x for x in X.nodes if X.nodes[x]["type"].lower() in ["outfall"]]
    XX = X.reverse()
    for o in outfalls:
        for n in nx.dfs_tree(XX, o).nodes:
            X.nodes[n]["Outfall"] = o
    return X


### ---------------------------NEW!---------------------------------------####
# NOTE! new 20190826
def makeTREE(X):
    """remove dual pathways and local depressions"""
    # make new copy
    X_new = X.copy()
    outfalls = [x for x in X.nodes if X.nodes[x]["type"].lower() in ["outfall"]]
    ##########################################################
    #  procedure 1: break multiple pathways
    #               water mainly flows to the min cost path
    ###########################################################
    # locate the nodes that split driange
    split_nodes = [n for n in X.nodes if X.out_degree(n) > 1]
    # looking for edges that linked with the node to break the path way
    remove_edges = []
    for n in split_nodes:
        # find all pathways and compute the cost
        outs_dict = {oo: None for oo in outfalls if nx.has_path(X, n, oo)}
        if len(outs_dict) > 0:
            for oo in outs_dict.keys():
                l = nx.dijkstra_path_length(
                    X,
                    n,
                    oo,
                    weight=lambda u, v, e: e["Zloss1_m"]
                    + e["Zloss2_m"]
                    + e["Zloss3_m"],
                )
                outs_dict[oo] = l
            # find the pathway with min cost to remain
            o = min(outs_dict.items(), key=itemgetter(1))[0]
            opath = nx.dijkstra_path(
                X,
                n,
                o,
                weight=lambda u, v, e: np.abs(
                    X.nodes[v]["chambfloor_m"] - X.nodes[u]["chambfloor_m"]
                ),
            )
        else:
            # if does not drain to any outfall, no path remain
            opath = []
        # find other pathways to remove
        for out_e in X.out_edges(n):
            if out_e[1] not in opath:
                # register the edge that break pathway
                remove_edges.append(out_e)

    # remove edge from new copy
    X_new.remove_edges_from(remove_edges)
    print(f"{len(remove_edges)} edges removed!")
    ##########################################################
    #  procedure 2: remove isolated nodes and parts (after procedure 1)
    #               local depression that not connected to the main sewer
    ###########################################################
    # find each connected component
    isolated_component = 0
    for c in nx.connected_components(nx.Graph(X_new)):
        # if there is no outfall exist in the component --> local depression
        if np.sum([1 if o in c else 0 for o in outfalls]) == 0:
            # remove all nodes in the component
            X_new.remove_nodes_from(c)
            isolated_component += 1
    print(f"{isolated_component} isolated_component removed!")
    return X_new


# NOTE! new 20190826 (modified version fo recover leaf)
# added condition: if a was originally not connected, ignore
# added condition: if an edge contains no weight, add the orginal weight by reverse
# corrected to Digraph
def recoverLEAF(X2, X1):
    # X2 is the new figure, X1 is the old figure
    # object is to recover X1 back to X2
    # Method: recover the stem back to main graph and correct the local depression to drain to stem
    def reverseEdgeDict(d):
        d_new = d.copy()
        d_new.update(
            {
                "gradient_mdm": (-d["gradient_mdm"] if d["gradient_mdm"] else 0),
                "capacity_m3ds": (-d["capacity_m3ds"] if d["capacity_m3ds"] else None),
                "us_invert_m": d["ds_invert_m"],
                "ds_invert_m": d["us_invert_m"],
            }
        )
        return d_new

    # gone nodes
    gone_nodes = [n for n in X1.nodes() if n not in X2.nodes()]
    # draw gone nodes
    G = nx.DiGraph(X1.subgraph(gone_nodes))
    # pos = graphviz_layout(G, prog='dot', args='')
    # plt.figure()
    # nx.draw(G, pos, node_size=20, alpha=0.5, node_color="blue", with_labels=False)
    # nx.draw(G.subgraph([n for n in G.nodes if len(G.out_edges(n)) == 0 ]), pos,\
    #     node_size=20, alpha=0.5, node_color="black", with_labels=True)
    X3 = nx.compose(X2, G)
    print("left over partitions:" + str(nx.number_connected_components(nx.Graph(G))))
    print("original components: " + str(nx.number_connected_components(nx.Graph(X2))))
    print("composed components: " + str(nx.number_connected_components(nx.Graph(X3))))
    # correct X3
    X = X1.copy()
    X_new = X3.copy()
    outfalls = [x for x in X.nodes if X.nodes[x]["type"].lower() in ["outfall"]]
    # leaves by leaves
    for c in nx.connected_components(nx.Graph(G)):
        # if none of the component contains outfall
        if np.sum([1 for o in outfalls if o in c]) == 0:
            SG = G.subgraph(c)
            # identify wrongly connected leaves and speperate into stem case and local depression cases
            ss = [
                n
                for n in SG.nodes
                if len(SG.out_edges(n)) == 0 and SG.degree(n) != X.degree(n)
            ]
            if len(ss) >= 1:
                stem = ss[0]
                depr = [
                    n
                    for n in SG.nodes
                    if len(SG.out_edges(n)) == 0
                    and SG.degree(n) == X.degree(n)
                    and n != stem
                ]
            else:
                ss = [n for n in SG.nodes if SG.degree(n) != X.degree(n)]
                if len(ss) >= 1:
                    stem = ss[0]
                    depr = [
                        n
                        for n in SG.nodes
                        if len(SG.out_edges(n)) == 0
                        and SG.degree(n) == X.degree(n)
                        and n != stem
                    ]
                else:
                    continue
        # start correcting
        ns = stem
        print("Adding " + ns)
        if len(SG.out_edges(ns)) != len(X.out_edges(ns)):
            mismatch_es = [e for e in X.out_edges(ns) if e not in SG.out_edges(ns)]
            for me in mismatch_es:
                us, ds = me
                d = dict(X[us][ds])
                if me not in X_new.edges():
                    print("\tout edge does not exist, adding...")
                    print("\t\tOut-edge added: " + us + " to" + ds)
                    X_new.add_edges_from([(us, ds, d)])
        elif len(SG.in_edges(ns)) != len(X.in_edges(ns)):
            mismatch_es = [e for e in X.in_edges(ns) if e not in SG.in_edges(ns)]
            for me in mismatch_es:
                us, ds = me
                d = dict(X[us][ds])
                if me not in X_new.edges():
                    print("\tin edge does not exist, adding....")
                    print("\t\t Reversed in-edge added:" + ds + " to" + us)
                    d = reverseEdgeDict(d)
                    X_new.add_edges_from([(ds, us, d)])
                    # check if stem connext to outfall
                    if np.any([nx.has_path(X_new, ns, o) for o in outfalls]):
                        print("\t" + ns + " is sucessfully added to graph!")
                        # fix depressions
                        for nd in depr:
                            nds_path = nx.shortest_path(
                                nx.Graph(SG), source=nd, target=ns
                            )
                            for e in zip(nds_path[:-1], nds_path[1:]):
                                if (
                                    e not in X_new.edges()
                                    and (e[1], e[0]) in X_new.edges()
                                ):
                                    # the defination of us ds is the corrected edge direction, not the original
                                    us, ds = e
                                    d = dict(X_new[ds][us])
                                    d = reverseEdgeDict(d)
                                    X_new.add_edges_from([(us, ds, d)])
                                    X_new.remove_edge(ds, us)
                                    print(
                                        "\t"
                                        + nd
                                        + "(depressions) is sucessfully corrected!"
                                    )
                    else:
                        raise (ValueError(ns + " can still not connect to path!"))
    # chekc if all nodes in gone nodes now connects to an outfall
    if np.all(
        [np.any([nx.has_path(X_new, n, o) for o in outfalls]) for n in gone_nodes]
    ):
        print(
            "ALL NODES FOUND OUTFALL! After recovered components: "
            + str(nx.number_connected_components(nx.Graph(X_new)))
            + "\n"
        )
    # check if all edge contain weigths
    old_edges = X.edges()
    for e in X_new.edges():
        us, ds = e
        if (us, ds) not in old_edges and (ds, us) in old_edges:
            d = reverseEdgeDict(dict(X[ds][us]))
            X_new[us][ds].update(d)
    print("ALL EDGES FOUND WEIGHTS\n")
    return X_new


#### ---------------------------DAG related------------------------------------#######


# add cost computation to edge
# NEW 06/12/2018
def calcCost(X):
    # compute all cost variables and assign to edge
    # no data as 0.
    for e in X.edges(data=True):
        us, ds = e[:2]
        # calculate time of travel is the pipe has velocity and gradient is not 0

        if e[2]["shape"] != 0 and e[2]["gradient_mdm"] != 0 and e[2]["shape"] != None:
            condshape = e[2]["shape"].upper()

            w = e[2]["width_m"]
            h = e[2]["height_m"]
            s = e[2]["sediment_m"]
            A, P = wetPipe(condshape, w, h, s)
            # capacity condition
            Dh = 4.0 * A / P
            Sf = e[2]["gradient_mdm"]
            Ks = e[2]["rcw_mm"]
            L = e[2]["length_m"]
            v = V_CW(Dh, Sf, Ks)
            Qcap = v * A
            if v != 0:
                Re = Re_R(np.abs(v), Dh)
                f = f_DW(Re, Ks, Dh)
                Hf = Hf_DW(f, np.abs(v), L, Dh)
            else:
                Hf = 0.0
            # pressure conditon
            D = X.nodes[us]["chambroof_m"] - e[2]["ds_invert_m"] + e[2]["height_m"]
            Z = X.nodes[us]["chambroof_m"] - X.nodes[ds]["chambroof_m"]
            if Dh != 0 and D != 0:
                vmax = V_CW(Dh, D / L, Ks)
                Qmax = A * vmax
                if Qmax != 0:
                    Hfmax = Hf_DW(
                        f_DW(Re_R(np.abs(vmax), Dh), Ks, Dh), np.abs(vmax), L, Dh
                    )
                else:
                    Hfmax = 0.0
            else:
                Qmax = Qcap
                vmax = v
                if Qmax != 0:
                    Hfmax = Hf_DW(
                        f_DW(Re_R(np.abs(vmax), Dh), Ks, Dh), np.abs(vmax), L, Dh
                    )
                else:
                    Hfmax = 0.0
            # minumum conditon
            if Dh != 0 and Z != 0:
                Qmin = A * V_CW(Dh, Z / L, Ks)
            else:
                Qmin = 0.0
        else:
            A = 0.0
            P = 0.0
            D = 0.0
            v = 0.0
            Qcap = 0
            Hf = 0
            vmax = 0
            Qmax = 0
            Hfmax = 0
            Qmin = 0
        e[2]["A_m2"] = A
        e[2]["P_m"] = P
        e[2]["Dh_m"] = D
        e[2]["vcap_mds"] = v
        e[2]["Qcap_m3ds"] = Qcap
        e[2]["Hf_m"] = Hf
        e[2]["vmax_mds"] = vmax
        e[2]["Qmax_m3ds"] = Qmax
        e[2]["Hfmax_m"] = Hfmax
        e[2]["Qmin_m3ds"] = Qmin
        Zloss1 = (
            e[2]["us_invert_m"] - X.nodes[us]["chambfloor_m"]
            if e[2]["us_invert_m"] - X.nodes[us]["chambfloor_m"] > 0
            else 0.0
        )
        Zloss2 = (
            -(e[2]["us_invert_m"] - e[2]["ds_invert_m"])
            if (e[2]["us_invert_m"] - e[2]["ds_invert_m"]) < 0
            else 0.0
        )
        Zloss3 = (
            (e[2]["us_invert_m"] - e[2]["ds_invert_m"])
            if (e[2]["us_invert_m"] - e[2]["ds_invert_m"]) > 0
            else 0.0
        )
        e[2]["Zloss1_m"] = Zloss1
        e[2]["Zloss2_m"] = Zloss2
        e[2]["Zloss3_m"] = Zloss3
    print("cost related parameters calculated.")
    return X


# ultidigraph to digraph
# NEW 06/12/2018
# aggregate if multiple edges exist between us and ds node
def multi2uniDiGraph(X):
    # sum if multi edge exist
    # no data as no.nan instead of None
    X_new = nx.DiGraph(X)
    for us, ds in X.edges():
        Xdata = X[us].get(ds, {})
        # multi edges detection
        if len(Xdata) > 1:
            new_weight = {
                k: np.sum([np.array(Xdata[e][k], dtype=np.float) for e in Xdata])
                if type(Xdata[0][k]) != type("str")
                else Xdata[0][k]
                for k in Xdata[0].keys()
            }
            X_new[us][ds].update(new_weight)
    return X_new


# make DAG graph
# NEW 06/12/2018
# Update 26/02/2019 the cost is not calculated as the sum of Zlosses
# sort sewer to follow min cost flow direction (DAG)
def makeDAG(X):
    # create a DAG allowing flow along min cost
    # create a 'clean' DAG
    X_new = nx.DiGraph(X)
    X_new.remove_edges_from(X.edges())
    XX = X.reverse()
    outfalls = [x for x in X.nodes if X.nodes[x]["type"].lower() in ["outfall"]]
    for o in outfalls:
        sub_network = X.subgraph(nx.dfs_tree(XX, o).nodes)
        for n in sub_network.nodes:
            path = nx.dijkstra_path(
                sub_network,
                n,
                o,
                weight=lambda u, v, e: e["Zloss1_m"] + e["Zloss2_m"] + e["Zloss3_m"],
            )  ##########
            if len(path) > 1:
                X_new.add_edge(
                    path[0], path[1]
                )  # It used to add the entire path. Now I add edges manually one-by-one
    for u, v, weight in X_new.edges(data=True):
        # get the old data from X
        xdata = X[u].get(v)
        non_shared = set(xdata) - set(weight)
        if non_shared:
            # add old weights to new weights if not in new data
            weight.update(dict((key, xdata[key]) for key in non_shared))
    return X_new


# calculate tc
# NEW 06/12/2018
# Update 26/02/2019 add the method by Garcia
# calculate concentration time for DAG graph
# Update 27/08/2019 modeify to adapt for python3
# UPDATE 29/08/2019
# added meanv: mean velocity flow on maximum flow path
def calcDAGTc(X, method="CW_cap"):
    # for use if DAG and the edge cost has been calculated,
    XX = X.reverse()
    for n in X.nodes:
        # sub-network of node n
        ns = list(nx.dfs_postorder_nodes(XX, n))
        sub_network = XX.subgraph(ns)
        X.nodes[n]["us_area_m2"] = sum(
            sub_network.nodes[k]["cont_area_m2"] for k in ns
        )  # ( for exclude n ass: if k != n)
        X.nodes[n]["us_count"] = len(
            sub_network.nodes
        )  # ( for exclude n ass: if k != n)
        X.nodes[n]["gross_us_area_m2"] = sum(
            sub_network.nodes[k]["gross_area_m2"] for k in ns
        )  # ( for exclude n ass: if k != n)
        if method == "CW_cap":
            # calculate trvel time using colebrook-white under capacity condition
            t_path = nx.single_source_dijkstra_path_length(
                sub_network,
                n,
                weight=lambda u, v, e: e["length_m"] / (e["vcap_mds"])
                if e["vcap_mds"] > 0
                else 0,
            )
            tc = max(
                [
                    t_path[k] if not np.isnan(t_path[k]) or t_path[k] == 0 else np.nan
                    for k in t_path
                ]
            )
            X.nodes[n]["tc_min"] = (
                (int(tc / 60.0 / 5.0) + 1) * 5 if not np.isnan(tc) else np.nan
            )  # longest time of travel path in minutes (rounded to 5min)
        elif method == "CW_max":
            # calculate trvel time using colebrook-white under capacity condition
            t_path = nx.single_source_dijkstra_path_length(
                sub_network,
                n,
                weight=lambda u, v, e: e["length_m"] / (e["vmax_mds"])
                if e["vmax_mds"] > 0
                else 0,
            )
            tc = max([t_path[k] if not np.isnan(t_path[k]) else 0 for k in t_path])
            X.nodes[n]["tc_min"] = (
                int(tc / 60.0 / 5.0) + 1
            ) * 5  # longest time of travel path in minutes (rounded to 5min)
        elif method == "C":
            # calculate longest flow path length and slope for cater lag equation
            l_path = nx.single_source_dijkstra_path_length(
                sub_network, n, weight="length_m"
            )
            (ns, l) = max(l_path.items(), key=itemgetter(1))
            l_m = l  # maximum flow path length meters
            s_m = (
                X.nodes[ns]["chambroof_m"] - X.nodes[n]["chambfloor_m"]
            )  # maximum elevation differences meters
            tc_min = (
                100
                * (l_m / 1609.344) ** 0.6
                * ((s_m * 3.281) / (l_m / 1609.344)) ** (-0.3)
                if l_m != 0 and s_m > 0
                else 0
            )
            X.nodes[n]["tc_min"] = (int(tc_min / 5.0) + 1) * 5  # (rounded to 5min)
        elif method == "Garcia":
            # calculate longest flow path length and slope for Garcia's (https://www.mdpi.com/2073-4441/9/5/303/htm)
            l_path = nx.single_source_dijkstra_path_length(
                sub_network, n, weight="length_m"
            )
            (ns, l) = max(l_path.items(), key=itemgetter(1))
            try:
                l_m = l  # maximum flow path length meters
                s_m = (
                    X.nodes[ns]["chambroof_m"] - X.nodes[n]["chambfloor_m"]
                )  # maximum elevation differences meters
                miu = X.nodes[n]["us_area_m2"] / X.nodes[n]["gross_us_area_m2"]
                tc_min = (0.3 * (l_m**0.76 / (s_m / l_m) ** 0.19)) / (
                    1 + 3 * np.sqrt(miu * (2 - miu))
                )
                X.nodes[n]["tc_min"] = (int(tc_min / 5.0) + 1) * 5  # (rounded to 5min)
            except:
                X.nodes[n]["tc_min"] = np.nan
                print(n + ": no tc_min calculated")
        elif method == "meanv":
            mean_v = np.nanmean(
                [e[2] for e in sub_network.edges(data="vcap_mds") if e[2] > 0]
            )
            if mean_v != 0 and ~np.isnan(mean_v):
                # calculate trvel time using colebrook-white under capacity condition
                l_path = nx.single_source_dijkstra_path_length(
                    sub_network, n, weight=lambda u, v, e: e["length_m"]
                )
                l = np.nanmean(np.array([l_path[k] for k in l_path]))
                tc = l / mean_v
            else:
                tc = np.nan
            X.nodes[n]["tc_min"] = (
                (int(tc / 60.0 / 5.0) + 1) * 5 if not np.isnan(tc) else np.nan
            )
        else:
            print(
                'method unknown, please choose between: "CW_cap": Colebrook-while under capacity condition,',
                '"CW_cap": Colebrook-while under capacity condition',
                'or "C": carters lag',
            )
            raise (TypeError)
    print("concentration time calculated. method: " + method + "\n")
    return X


# calculate static FI
# NEW 06/12/2018
# calculate static FI for DAG graph to ndoes
def calcDAGStaticFI_reduced(X):
    #
    for e in X.edges(data=True):
        us, ds = e[:2]
        X.nodes[us]["Qmax_m3ds"] = e[2]["Qmax_m3ds"]
        X.nodes[us]["Qmin_m3ds"] = e[2]["Qmin_m3ds"]
        X.nodes[us]["Qcap_m3ds"] = e[2]["Qcap_m3ds"]
    print("DS Edge related flooding indices calculated.")
    return X


# calculate static FI
# NEW 12/02/2019
# calculate static FI for DAG graph to ndoes
# calculate Qo, Qo0
def calcDAGStaticFI(X):
    # calculate static FI with Qo_m3ds
    outfalls = [x for x in X.nodes if X.nodes[x]["type"].lower() in ["outfall"]]
    XX = X.reverse()
    for o in outfalls:
        sub_network = X.subgraph(nx.dfs_tree(XX, o).nodes)
        for n in sub_network.nodes:
            dist = np.sqrt(
                (X.nodes[n]["geo"][0] - X.nodes[o]["geo"][0]) ** 2
                + (X.nodes[n]["geo"][1] - X.nodes[o]["geo"][1]) ** 2
            )
            delta_h = X.nodes[n]["chambroof_m"] - X.nodes[o]["chambfloor_m"]
            delta_h_0 = X.nodes[n]["chambfloor_m"] - X.nodes[o]["chambfloor_m"]
            if delta_h != 0:
                X.nodes[n]["hydraulicgradient_o"] = delta_h / dist
                X.nodes[n]["hydraulicgradient_o_0"] = delta_h_0 / dist
            else:
                X.nodes[n]["hydraulicgradient_o"] = 0.001
                X.nodes[n]["hydraulicgradient_o_0"] = 0.001
            X.nodes[n]["pathlength_o"] = nx.dijkstra_path_length(
                sub_network, n, o, weight=lambda u, v, e: e["length_m"]
            )
            X.nodes[n]["gross_us_area_m2"] = sum(
                X.nodes[k]["gross_area_m2"] for k in list(nx.dfs_postorder_nodes(XX, n))
            )
    for e in X.edges(data=True):
        us, ds = e[:2]
        if e[2]["Dh_m"]:
            vmax = V_CW(
                e[2]["Dh_m"], X.nodes[us]["hydraulicgradient_o"], e[2]["rcw_mm"]
            )
            Qmax = e[2]["A_m2"] * vmax
            e[2]["Qo_m3ds"] = Qmax
            X.nodes[us]["Qo_m3ds"] = e[2]["Qo_m3ds"]
            vmin = V_CW(
                e[2]["Dh_m"], X.nodes[us]["hydraulicgradient_o_0"], e[2]["rcw_mm"]
            )
            Qmin = e[2]["A_m2"] * vmin
            e[2]["Qo_0_m3ds"] = Qmin
            X.nodes[us]["Qo_0_m3ds"] = e[2]["Qo_0_m3ds"]
        X.nodes[us]["A_m2"] = e[2]["A_m2"]
        X.nodes[us]["Dh_m"] = e[2]["Dh_m"]
        X.nodes[us]["rcw_mm"] = e[2]["rcw_mm"]
        X.nodes[us]["Qmax_m3ds"] = e[2]["Qmax_m3ds"]
        X.nodes[us]["Qmin_m3ds"] = e[2]["Qmin_m3ds"]
        X.nodes[us]["Qcap_m3ds"] = e[2]["Qcap_m3ds"]
    print("DS Edge related flooding indices calculated.")
    return X


# calculate new capacity fill in hydraulic structures and negative NODES
# NEW 10/02/2019
def fillNegCapacity_nodes(X, param="Qcap_m3ds"):
    #  find Qcap_m3ds <= 0 or None
    for n in X.nodes:
        if param not in X.nodes[n].keys() or X.nodes[n][param] <= 0:
            X.nodes[n][param] = np.nan
    nan_nodes = len([i[1] for i in list(X.nodes(data=param)) if np.isnan(i[1])])
    # interpolate along each path
    XX = X.reverse()
    outfalls = [x for x in X.nodes if X.nodes[x]["type"].lower() in ["outfall"]]
    for o in outfalls:
        sub_network = X.subgraph(nx.dfs_tree(XX, o).nodes)
        end_nodes = [n for n in sub_network.nodes if len(sub_network.in_edges(n)) == 0]
        for n in end_nodes:
            path = nx.shortest_path(XX, o, n)
            y = np.array([X.nodes[n][param] for n in path])
            if np.any(np.isnan(y)) and not np.all(np.isnan(y)):
                try:
                    mask = np.isnan(y)
                    y[mask] = np.interp(
                        np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask]
                    )
                    for nn, ny in zip(path, y):
                        X.nodes[nn][param] = ny
                except:
                    pass
    nan_nodes_new = len([i[1] for i in list(X.nodes(data=param)) if np.isnan(i[1])])
    print(param + ": Reduced nan from %d to %d" % (nan_nodes, nan_nodes_new))
    return X


# recalculate new tc using interpolated Qo_m3ds at node attributes
# NEW 13/02/2019
def reCalcDAGTc(X, v_var="vcap_mds", q_var="Qcap_m3ds"):
    # recalculate tc under Qo condition
    for e in X.edges(data=True):
        us, ds = e[:2]
        e[2][v_var] = X.nodes[us][q_var] / e[2]["A_m2"] if e[2]["A_m2"] > 0 else 0
    XX = X.reverse()
    for n in X.nodes:
        # sub-network of node n
        ns = list(nx.dfs_postorder_nodes(XX, n))
        sub_network = XX.subgraph(ns)
        # calculate trvel time using colebrook-white under o condition
        t_path = nx.single_source_dijkstra_path_length(
            sub_network,
            n,
            weight=lambda u, v, e: e["length_m"] / (e[v_var]) if e[v_var] > 0 else 0,
        )
        tc = max([t_path[k] if not np.isnan(t_path[k]) else 0 for k in t_path])
        X.nodes[n]["tc_min"] = (
            int(tc / 60.0 / 5.0) + 1
        ) * 5  # longest time of travel path in minutes (rounded to 5min)
    print("Tc is calculated.")
    return X


#### ---------------------------NON-DAG related------------------------------------#######


# calculate concentration time
# Update 30/11/2018
# change the definaiton of us_area, do include the current node
# Update 27/11/2018
# change the definaiton of us_area, to seperate Qin1 and Qin2
# add sediment in calculation of pipe velocity
# add the option to choose between method: 'CW_cap', 'CW_max', 'C' (for charters)
# modify edges in a different way
def calcConcentrationTime(X, method="CW_cap"):
    for e in X.edges(data=True):
        us, ds = e[:2]
        # calculate time of travel is the pipe has velocity and gradient is not 0
        if e[2]["shape"] != 0 and e[2]["gradient_mdm"] != 0:
            condshape = e[2]["shape"].upper()
            w = e[2]["width_m"]
            h = e[2]["height_m"]
            s = e[2]["sediment_m"]
            A, P = wetPipe(condshape, w, h, s)
            # capacity condition
            Dh = 4.0 * A / P
            Sf = e[2]["gradient_mdm"]
            Ks = e[2]["rcw_mm"]
            L = e[2]["length_m"]
            v = V_CW(Dh, Sf, Ks)
            Qcap = v * A
            if v != 0:
                Re = Re_R(np.abs(v), Dh)
                f = f_DW(Re, Ks, Dh)
                Hf = Hf_DW(f, np.abs(v), L, Dh)
            else:
                Hf = 0.0
            # pressure conditon
            D = X.nodes[us]["chambroof_m"] - e[2]["ds_invert_m"] + e[2]["height_m"]
            Z = X.nodes[us]["chambroof_m"] - X.nodes[ds]["chambroof_m"]
            if Dh != 0 and D != 0:
                vmax = V_CW(Dh, D / L, Ks)
                Qmax = A * vmax
                if Qmax != 0:
                    Hfmax = Hf_DW(
                        f_DW(Re_R(np.abs(vmax), Dh), Ks, Dh), np.abs(vmax), L, Dh
                    )
                else:
                    Hfmax = 0.0
            else:
                Qmax = Qcap
                vmax = v
                if Qmax != 0:
                    Hfmax = Hf_DW(
                        f_DW(Re_R(np.abs(vmax), Dh), Ks, Dh), np.abs(vmax), L, Dh
                    )
                else:
                    Hfmax = 0.0
            # minumum conditon
            if Dh != 0 and Z != 0:
                Qmin = A * V_CW(Dh, Z / L, Ks)
            else:
                Qmin = 0.0
        else:
            A = 0.0
            P = 0.0
            D = 0.0
            v = 0.0
            Qcap = 0
            Hf = 0
            vmax = 0
            Qmax = 0
            Hfmax = 0
            Qmin = 0
        e[2]["A_m2"] = A
        e[2]["P_m"] = P
        e[2]["Dh_m"] = D
        e[2]["vcap_mds"] = v
        e[2]["Qcap_m3ds"] = Qcap
        e[2]["Hf_m"] = Hf
        e[2]["vmax_mds"] = vmax
        e[2]["Qmax_m3ds"] = Qmax
        e[2]["Hfmax_m"] = Hfmax
        e[2]["Qmin_m3ds"] = Qmin
    print("velocity related parameters calculated.")
    XX = X.reverse()
    for n in X.nodes:
        # sub-network of node n
        ns = list(nx.dfs_postorder_nodes(XX, n))
        sub_network = XX.subgraph(ns)
        X.nodes[n]["us_area_m2"] = sum(
            sub_network.nodes[k]["cont_area_m2"] for k in ns
        )  # ( for exclude n ass: if k != n)
        if method == "CW_cap":
            # calculate trvel time using colebrook-white under capacity condition
            t_path = nx.single_source_dijkstra_path_length(
                sub_network,
                n,
                weight=lambda u, v, e: e[0]["length_m"] / (e[0]["vcap_mds"])
                if e[0]["vcap_mds"] > 0
                else 0,
            )
            tc = max([t_path[k] if not np.isnan(t_path[k]) else 0 for k in t_path])
            X.nodes[n]["tc_min"] = (
                int(tc / 60.0 / 5.0) + 1
            ) * 5  # longest time of travel path in minutes (rounded to 5min)
        elif method == "CW_max":
            # calculate trvel time using colebrook-white under capacity condition
            t_path = nx.single_source_dijkstra_path_length(
                sub_network,
                n,
                weight=lambda u, v, e: e[0]["length_m"] / (e[0]["vmax_mds"])
                if e[0]["vmax_mds"] > 0
                else 0,
            )
            tc = max([t_path[k] if not np.isnan(t_path[k]) else 0 for k in t_path])
            X.nodes[n]["tc_min"] = (
                int(tc / 60.0 / 5.0) + 1
            ) * 5  # longest time of travel path in minutes (rounded to 5min)
        elif method == "C":
            # calculate longest flow path length and slope for cater lag equation
            l_path = nx.single_source_dijkstra_path_length(
                sub_network, n, weight="length_m"
            )
            (ns, l) = max(l_path.iteritems(), key=itemgetter(1))
            l_m = l  # maximum flow path length meters
            s_m = (
                X.nodes[ns]["chambroof_m"] - X.nodes[n]["chambfloor_m"]
            )  # maximum elevation differences meters
            tc_min = (
                100
                * (l_m / 1609.344) ** 0.6
                * ((s_m * 3.281) / (l_m / 1609.344)) ** (-0.3)
                if l_m != 0 and s_m > 0
                else 0
            )
            X.nodes[n]["tc_min"] = (int(tc_min / 5.0) + 1) * 5  # (rounded to 5min)
        else:
            print(
                'method unknown, please choose between: "CW_cap": Colebrook-while under capacity condition,',
                '"CW_cap": Colebrook-while under capacity condition',
                'or "C": carters lag',
            )
            raise (TypeError)
    print("upstream area (include current node) calculated.")
    print("concentration time calculated. method: " + method + "\n")
    return X


# calculate reduced flooding indices
# New 27/11/2018
def calcStaticFI_reduced(X):
    ################ computing flood indices (reduced version, less is more)
    ## Ds Links related
    # corrected D_m to count until pipe upper edge
    # added Hfmax_m to count for maximum friction loss
    for e in X.edges(data=True):
        us, ds = e[:2]
        X.nodes[us]["Qmax_m3ds"] = e[2]["Qmax_m3ds"]
        X.nodes[us]["Qmin_m3ds"] = e[2]["Qmin_m3ds"]
        X.nodes[us]["Qcap_m3ds"] = e[2]["Qcap_m3ds"]
        X.nodes[us]["Hf_m"] = e[2]["Hf_m"]
        X.nodes[us]["Hfmax_m"] = e[2]["Hfmax_m"]
    print("DS Edge related flooding indices calculated.")
    return X


# calculate path related static FIs
# New 01/12/2018
def calcStaticFI_path(X):
    ## Path cost
    for e in X.edges(data=True):
        us, ds = e[:2]
        edge_data = X.edges[us, ds, 0]
        Zloss1 = (
            edge_data["us_invert_m"] - X.nodes[us]["chambfloor_m"]
            if edge_data["us_invert_m"] - X.nodes[us]["chambfloor_m"] > 0
            else 0.0
        )
        Zloss2 = (
            -(edge_data["us_invert_m"] - edge_data["ds_invert_m"])
            if (edge_data["us_invert_m"] - edge_data["ds_invert_m"]) < 0
            else 0.0
        )
        e[2]["Zloss1_m"] = Zloss1
        e[2]["Zloss2_m"] = Zloss2
    print("Edge cost information updated.")
    ## Path to outfall related
    Path_L_m = {k: None for k in X.nodes}
    Path_rl_m = {k: None for k in X.nodes}
    Path_dZ_m = {k: None for k in X.nodes}
    Path_Slope_mdm = {k: None for k in X.nodes}
    Path_Zloss1_m = {k: None for k in X.nodes}
    Path_Zloss2_m = {k: None for k in X.nodes}
    Path_Zloss_m = {k: None for k in X.nodes}
    Path_Hf_m = {k: None for k in X.nodes}
    XX = X.reverse()
    outfalls = [x for x in X.nodes if X.nodes[x]["type"].lower() in ["outfall"]]
    for o in outfalls:
        sub_network = createSubgraph(X, nx.dfs_tree(XX, o).nodes)
        ons_l = nx.single_source_dijkstra_path_length(
            XX.subgraph(nx.dfs_tree(XX, o).nodes), o, weight="length_m"
        )
        ons_zloss1 = nx.single_source_dijkstra_path_length(
            XX.subgraph(nx.dfs_tree(XX, o).nodes), o, weight="Zloss1_m"
        )
        ons_zloss2 = nx.single_source_dijkstra_path_length(
            XX.subgraph(nx.dfs_tree(XX, o).nodes), o, weight="Zloss2_m"
        )
        ons_zloss = nx.single_source_dijkstra_path_length(
            XX.subgraph(nx.dfs_tree(XX, o).nodes),
            o,
            weight=lambda u, v, e: e[0]["Zloss1_m"] + e[0]["Zloss2_m"],
        )
        ons_Hf_m = nx.single_source_dijkstra_path_length(
            XX.subgraph(nx.dfs_tree(XX, o).nodes), o, weight="Hf_m"
        )
        for n in sub_network.nodes:
            Path_dZ_m[n] = (
                sub_network.nodes[n]["chambfloor_m"]
                - sub_network.nodes[o]["chambfloor_m"]
            )
        Path_L_m.update(ons_l)
        ons_l_mean = float(sum(ons_l.values())) / len(ons_l)
        Path_rl_m.update(
            {k: ons_l[k] / ons_l_mean if ons_l_mean != 0 else 0 for k in ons_l}
        )
        Path_Slope_mdm.update(
            {k: Path_dZ_m[k] / ons_l[k] if ons_l[k] != 0 else 0 for k in ons_l}
        )
        Path_Zloss1_m.update(ons_zloss1)
        Path_Zloss2_m.update(ons_zloss2)
        Path_Zloss_m.update(ons_zloss)
        Path_Hf_m.update(ons_Hf_m)
    print("Path related flooding indices calculated.")
    #############  update information
    for n in X.nodes:
        X.nodes[n].update(
            {
                "Path_L_m": Path_L_m[n],
                "Path_rl_m": Path_rl_m[n],
                "Path_dZ_m": Path_dZ_m[n],
                "Path_Slope_mdm": Path_Slope_mdm[n],
                "Path_Zloss1_m": Path_Zloss1_m[n],
                "Path_Zloss2_m": Path_Zloss2_m[n],
                "Path_Zloss_m": Path_Zloss_m[n],
                "Path_Hf_m": Path_Hf_m[n],
            }
        )
    print("Flooding indices updated to node.\n")
    return X


########################################################
# calibraiton and validation tools
########################################################


# linear reservior model for computing subcatchment Qin
# NEW 29/11/2018
def linearReservior(R_df, A):
    # R is effective rainfall, m^3/hr - DataFrame
    # A is reservior constant - 15hr
    # Q is m3/s
    A = 1.0 / A
    R = R_df.values
    Q = np.zeros(len(R))
    for i in range(1, len(R)):
        Q[i] = Q[i - 1] * np.exp(-A * 5.0) + R[i - 1] * (1 - np.exp(-A * 5.0))
    return pd.Series(Q / 3600.0, index=R_df.index)


# calculate Q
# Updated 1/12/2018
# rainfall runoff linear model included
# sim_n_RO = I(R,t)cont_area
# sim_nc_RO = I(R,t)us_area
# sim_nc_Q = I(R,tc)us_area
def calcQ(X, sim_n_RR):
    ######### node n
    # Qnode = I(R,t)cont_area
    sim_n_RO = pd.DataFrame(
        {
            n: linearReservior(
                sim_n_RR[n] / (10**3) * X.nodes[n]["cont_area_m2"], 15.0
            )
            if n in sim_n_RR.columns and X.nodes[n]["type"].lower() not in ["outfall"]
            else sim_n_RR.iloc[:, 0] * 0.0
            for n in X.nodes
        }
    )

    ########## subcatchment contributing to node n
    sim_nc_RO = sim_n_RO * 0.0
    sim_nc_Q = sim_n_RO * 0.0
    XX = X.reverse()
    for n in XX.nodes:
        ns = list(nx.dfs_postorder_nodes(XX, n))
        # subcatchment runoff - accumulation in space (sum)
        RO_t = pd.concat(
            [sim_n_RO[[nsi for nsi in ns if nsi in sim_n_RO.columns]]], axis=1
        ).sum(axis=1)
        # subcatchment discharge - accumulation in time (convolve)
        if X.nodes[n]["tc_min"]:
            window = int(X.nodes[n]["tc_min"] / 5)
            if window > 1:
                Q_tc = np.convolve(
                    RO_t,
                    np.ones(
                        window,
                    )
                    / window,
                    mode="full",
                )[: RO_t.size]
            else:
                Q_tc = RO_t
            sim_nc_RO[n] = RO_t
            sim_nc_Q[n] = Q_tc
        else:
            sim_nc_RO = sim_nc_RO.drop([n], axis=1)
            sim_nc_Q = sim_nc_Q.drop([n], axis=1)
    return sim_n_RO, sim_nc_RO, sim_nc_Q


# calculate return period
# NEW 4/12/2018
# Update 5/12/2018 only calculate for outfall, and only calculate if > T=0.1 else 0
# Updated 06/02/2019 should not include nodes who has no runoff area, because rainfall is homogeneous (more or less)
def calcOT(X, sim_n_RR):
    outfalls = [x for x in X.nodes if X.nodes[x]["type"].lower() in ["outfall"]]
    sim_o_T = {o: np.nan * sim_n_RR.iloc[:, 1].values for o in outfalls}
    XX = X.reverse()
    for o in outfalls:
        ns = list(nx.dfs_postorder_nodes(XX, o))
        # subcatchment rainfall rate - average in space (sum)
        RR_t = np.mean(
            sim_n_RR[[nsi for nsi in ns if nsi in sim_n_RR.columns]].values, axis=1
        )
        if not np.any(np.isnan(RR_t)):
            # subcatchment rainfall rate - average in time (convolve over concentrationt time)
            if X.nodes[o]["tc_min"]:
                window = int(X.nodes[o]["tc_min"] / 5)
                if window > 1:
                    RR_tc = np.convolve(
                        RR_t,
                        np.ones(
                            window,
                        )
                        / window,
                        mode="full",
                    )[: RR_t.size]
                else:
                    RR_tc = RR_t
                I_I1month = I_idf(1.0 / 12.0, X.nodes[o]["tc_min"])
                sim_o_T[o] = np.array(
                    [
                        T_idf(intensity, X.nodes[o]["tc_min"])
                        if intensity > I_I1month
                        else 0.0
                        for intensity in RR_tc
                    ]
                )
    return pd.DataFrame.from_dict(sim_o_T, orient="index").T.set_index(sim_n_RR.index)


# calculate F and FV at node locations
def calcF(X, sim_n_FV):
    sim_n_FV = sim_n_FV.where(sim_n_FV > 0, 0)
    sim_n_F = sim_n_FV.mask(sim_n_FV > 0, 1)
    return sim_n_F, sim_n_FV


# obtain the summary table for an event
# Updated 1/12/2018
# removed thew calculation of weighted path length
# added Qoutfall
# Updated 3/12/2018
# added max node runoff
# Update 4/12/2018
# add consideration of Return period T of outfall
def summarizeEvent(
    X, sim_nc_RO, sim_nc_Q, sim_nc_IT, sim_nc_ITC, sim_o_T, sim_n_F, sim_n_FV
):
    nodes = dict(X.nodes(data=True))
    outfalls = [x for x in X.nodes if X.nodes[x]["type"].lower() in ["outfall"]]
    T_func_h = T_func()
    # static FIS (characteristics)
    table_Cs = pd.DataFrame.from_dict(nodes, orient="index")
    table_Cs = table_Cs.drop(outfalls)
    table_Cs.loc[:, "chambdepth_m"] = (
        table_Cs.loc[:, "chambvolume_m3"].values
        / table_Cs.loc[:, "chambarea_m2"].values
    )
    # dynamic FIs - option 1 - max at q
    table_Qs = sim_nc_Q.max().to_frame(name="nc_Qmq_m3ds")
    table_Qs = table_Qs.drop([o for o in outfalls if o in table_Qs.index])
    for n in table_Qs.index:
        mq_idx = sim_nc_Q[n].idxmax()
        table_Qs.loc[n, "nc_ROmq_m3ds"] = sim_nc_RO.loc[mq_idx, n]
        table_Qs.loc[n, "o_Tmq_yr"] = sim_o_T.loc[mq_idx, nodes[n]["Outfall"]]

    # dynamic FIs - option 2 - max at itc
    table_Ts = sim_nc_ITC.max().to_frame(name="nc_ITCmitc_yr")
    table_Ts = table_Ts.drop([o for o in outfalls if o in table_Ts.index])
    for n in table_Ts.index:
        mitc_idx = sim_nc_ITC[n].idxmax()
        table_Ts.loc[n, "nc_ITTmitc_yr"] = T_func_h.calc(
            np.array([sim_nc_IT.loc[mitc_idx, n], 5])
        )
        table_Ts.loc[n, "nc_ITCTmitc_yr"] = (
            T_func_h.calc(np.array([sim_nc_ITC.loc[mitc_idx, n], X.nodes[n]["tc_min"]]))
            if X.nodes[n]["tc_min"]
            else np.nan
        )
        table_Ts.loc[n, "o_Tmitc_yr"] = sim_o_T.loc[mitc_idx, nodes[n]["Outfall"]]
    # dynamic FIs - opention 1 2 - max at outfall
    table_Os = pd.DataFrame(index=table_Ts.index)
    for n in table_Os.index:
        if "Outfall" in nodes[n].keys() and n not in outfalls:
            mo_idx = sim_o_T[nodes[n]["Outfall"]].idxmax()
            table_Os.loc[n, "o_Tmo_yr"] = sim_o_T.loc[mo_idx, nodes[n]["Outfall"]]
    # flooding
    table_Fs = pd.concat([sim_n_F.max(), sim_n_FV.max()], axis=1, sort=False)
    table_Fs.columns = ["n_F", "n_FVm_m3"]
    # together
    table = pd.concat(
        [table_Cs, table_Qs, table_Ts, table_Os, table_Fs], axis=1, sort=False
    )
    table_columns = [
        "chambdepth_m",
        "us_area_m2",
        "Qo_m3ds",
        "nc_Qmq_m3ds",
        "nc_ROmq_m3ds",
        "o_Tmq_yr",
        "nc_ITTmitc_yr",
        "nc_ITCTmitc_yr",
        "o_Tmitc_yr",
        "o_Tmo_yr",
        "n_F",
        "n_FVm_m3",
    ]
    return table[table_columns]


def summarizeEventOld(
    X, sim_n_RO, sim_nc_RO, sim_nc_Q, sim_n_F, sim_n_FV, sim_o_T=None
):
    nodes = dict(X.nodes(data=True))
    outfalls = [x for x in X.nodes if X.nodes[x]["type"].lower() in ["outfall"]]
    # static FIS (characetersitcs)
    table_Cs = pd.DataFrame.from_dict(nodes, orient="index")
    table_Cs = table_Cs.drop(outfalls)  # Read node att
    # dynamic FIs
    table_Qs = sim_nc_Q.max().to_frame(name="nc_Qmq_m3ds")
    table_Qs = table_Qs.drop([o for o in outfalls if o in table_Qs.index])
    for n in table_Qs.index:
        mq_idx = sim_nc_Q[n].idxmax()
        # intensity = table_Qs.loc[n, 'nc_Qmq_m3ds']/X.nodes[n]['us_area_m2'] * 3600. * 1000.
        # if not np.isnan(intensity) and  intensity > I_idf(1./12., X.nodes[n]['tc_min']):
        #     table_Qs.loc[n, 'nc_Tmq_yr'] =  T_idf(intensity, X.nodes[n]['tc_min'])
        # elif np.isnan(intensity):
        #     table_Qs.loc[n, 'nc_Tmq_yr'] =  0.
        # else:
        #     table_Qs.loc[n, 'nc_Tmq_yr'] =  np.nan
        table_Qs.loc[n, "nc_ROmq_m3ds"] = sim_nc_RO.loc[mq_idx, n]
        table_Qs.loc[n, "n_ROmq_m3ds"] = sim_n_RO.loc[mq_idx, n]
        table_Qs.loc[n, "o_Tmq_yr"] = sim_o_T.loc[mq_idx, nodes[n]["Outfall"]]
        if "Outfall" in nodes[n].keys() and n not in outfalls:
            mo_idx = sim_o_T[nodes[n]["Outfall"]].idxmax()
            if isinstance(mo_idx, float) and np.isnan(mo_idx):
                table_Qs.loc[n, "nc_Qmo_m3ds"] = None
                table_Qs.loc[n, "nc_ROmo_m3ds"] = None
                table_Qs.loc[n, "n_ROmo_m3ds"] = None
                table_Qs.loc[n, "o_Tmo_yr"] = None
            else:
                table_Qs.loc[n, "nc_Qmo_m3ds"] = sim_nc_Q.loc[mo_idx, n]
                table_Qs.loc[n, "nc_ROmo_m3ds"] = sim_nc_RO.loc[mo_idx, n]
                table_Qs.loc[n, "n_ROmo_m3ds"] = sim_n_RO.loc[mo_idx, n]
                table_Qs.loc[n, "o_Tmo_yr"] = sim_o_T.loc[mo_idx, nodes[n]["Outfall"]]
            # table_Qs.loc[n, 'o_Qmq_m3ds'] = sim_nc_Q.loc[mq_idx, nodes[n]['Outfall']]
            # table_Qs.loc[n, 'o_Qcap_m3ds'] = np.sum([e[2] for e in X.in_edges(nodes[n]['Outfall'], data = 'Qcap_m3ds')])
            # table_Qs.loc[n, 'o_Qmax_m3ds'] = np.sum([e[2] for e in X.in_edges(nodes[n]['Outfall'], data = 'Qmax_m3ds')])
            # if sim_o_T is not None:
            #     table_Qs.loc[n, 'o_Tmq_yr'] = sim_o_T.loc[mq_idx, nodes[n]['Outfall']]
            #     table_Qs.loc[n, 'o_Tmo_yr'] = sim_o_T.loc[mo_idx, nodes[n]['Outfall']]
            # else:
            #     table_Qs.loc[n, 'o_Tmq_yr'] = None
            #     table_Qs.loc[n, 'o_Tm_yr'] = None
        mnro_idx = sim_nc_RO[n].idxmax()
        table_Qs.loc[n, "n_ROmnro_m3ds"] = sim_n_RO.loc[mnro_idx, n]
        table_Qs.loc[n, "nc_Qmnro_m3ds"] = sim_nc_Q.loc[mnro_idx, n]
        table_Qs.loc[n, "nc_ROmnro_m3ds"] = sim_nc_RO.loc[mnro_idx, n]
        if "Outfall" in nodes[n].keys() and n not in outfalls:
            #     table_Qs.loc[n, 'o_Qmnro_m3ds'] = sim_nc_Q.loc[mnro_idx, nodes[n]['Outfall']]
            table_Qs.loc[n, "o_Tmnro_yr"] = sim_o_T.loc[mnro_idx, nodes[n]["Outfall"]]
    # flooding
    table_Fs = pd.concat([sim_n_F.max(), sim_n_FV.max()], axis=1, sort=False)
    table_Fs.columns = ["n_F", "n_FVm_m3"]
    # together
    table = pd.concat(
        [table_Cs, table_Qs, table_Fs], axis=1, sort=False
    )  # Constants, discharges, floods

    table_columns = [
        "nc_Qmq_m3ds",
        "nc_ROmq_m3ds",
        "n_ROmq_m3ds",
        "o_Tmq_yr",  # , u'o_Qmq_m3ds','nc_Tmq_yr',
        "nc_Qmnro_m3ds",
        "nc_ROmnro_m3ds",
        "n_ROmnro_m3ds",
        "o_Tmnro_yr",  # u'o_Qmnro_m3ds',
        "Qcap_m3ds",
        "Qmax_m3ds",
        "Qmin_m3ds",
        "Outfall",
        "nc_Qmo_m3ds",
        "nc_ROmo_m3ds",
        "n_ROmo_m3ds",
        "o_Tmo_yr",  # u'o_Qcap_m3ds', u'o_Qmax_m3ds','o_Tmq_yr','o_Tm_yr',
        "n_F",
        "n_FVm_m3",
        "chambvolume_m3",
        "chambarea_m2",
        "cont_area_m2",
        "us_area_m2",
        "chambdepth_m",
        # 'Path_L_m', 'Path_rl_m', 'Path_dZ_m', 'Path_Slope_mdm',
        # 'Path_Zloss1_m', 'Path_Zloss2_m', 'Path_Zloss_m', 'Path_Hf_m',
    ]
    return table[table_columns]


########################################################
# multivariate statistics tools
########################################################


# calculate seperator
# = withingroup variance/between group variance
def calcWithinGroupsVariance(variable, groupvariable):
    # find out how many values the group variable can take
    levels = sorted(set(groupvariable))
    numlevels = len(levels)
    # get the mean and standard deviation for each group:
    numtotal = 0
    denomtotal = 0
    for leveli in levels:
        levelidata = variable[groupvariable == leveli]
        levelilength = len(levelidata)
        # get the standard deviation for group i:
        sdi = np.std(levelidata)
        numi = (levelilength) * sdi**2
        denomi = levelilength
        numtotal = numtotal + numi
        denomtotal = denomtotal + denomi
    # calculate the within-groups variance
    Vw = numtotal / (denomtotal - numlevels)
    return Vw


def calcBetweenGroupsVariance(variable, groupvariable):
    # find out how many values the group variable can take
    levels = sorted(set((groupvariable)))
    numlevels = len(levels)
    # calculate the overall grand mean:
    grandmean = np.mean(variable)
    # get the mean and standard deviation for each group:
    numtotal = 0
    denomtotal = 0
    for leveli in levels:
        levelidata = variable[groupvariable == leveli]
        levelilength = len(levelidata)
        # get the mean and standard deviation for group i:
        meani = np.mean(levelidata)
        sdi = np.std(levelidata)
        numi = levelilength * ((meani - grandmean) ** 2)
        denomi = levelilength
        numtotal = numtotal + numi
        denomtotal = denomtotal + denomi
    # calculate the between-groups variance
    Vb = numtotal / (numlevels - 1)
    return Vb


def calcSeparations(variables, groupvariable):
    # calculate the separation for each variable
    for variablename in variables:
        variablei = variables[variablename]
        Vw = calcWithinGroupsVariance(variablei, groupvariable)
        Vb = calcBetweenGroupsVariance(variablei, groupvariable)
        sep = Vb / Vw
        if sep > 1:
            print("variable", variablename, "Vw=", Vw, "Vb=", Vb, "separation=", sep)


# calculate bivariate seperator
def calcWithinGroupsCovariance(variable1, variable2, groupvariable):
    levels = sorted(set(groupvariable))
    numlevels = len(levels)
    Covw = 0.0
    # get the covariance of variable 1 and variable 2 for each group:
    for leveli in levels:
        levelidata1 = variable1[groupvariable == leveli]
        levelidata2 = variable2[groupvariable == leveli]
        mean1 = np.mean(levelidata1)
        mean2 = np.mean(levelidata2)
        levelilength = len(levelidata1)
        # get the covariance for this group:
        term1 = 0.0
        for levelidata1j, levelidata2j in zip(levelidata1, levelidata2):
            term1 += (levelidata1j - mean1) * (levelidata2j - mean2)
        Cov_groupi = term1  # covariance for this group
        Covw += Cov_groupi
    totallength = len(variable1)
    Covw /= totallength - numlevels
    return Covw


def calcBetweenGroupsCovariance(variable1, variable2, groupvariable):
    # find out how many values the group variable can take
    levels = sorted(set(groupvariable))
    numlevels = len(levels)
    # calculate the grand means
    variable1mean = np.mean(variable1)
    variable2mean = np.mean(variable2)
    # calculate the between-groups covariance
    Covb = 0.0
    for leveli in levels:
        levelidata1 = variable1[groupvariable == leveli]
        levelidata2 = variable2[groupvariable == leveli]
        mean1 = np.mean(levelidata1)
        mean2 = np.mean(levelidata2)
        levelilength = len(levelidata1)
        term1 = (mean1 - variable1mean) * (mean2 - variable2mean) * levelilength
        Covb += term1
    Covb /= numlevels - 1
    return Covb


########################################################
# figures
########################################################


# draw loop and label with nodes name, node chambfloor and edge capacity
def drawLoop(cycle, graph):
    loop_nodes = list(set([j for i in map(list, cycle) for j in i if j != 0]))
    loop_network = graph.subgraph(loop_nodes)
    pos = nx.get_node_attributes(loop_network, "geo")
    pos_xy = {n: pos[n][:2] for n in pos}
    pos_z = {n: n + ";\n " + str(pos[n][2]) for n in pos}
    f = plt.figure(figsize=(20, 10))
    nx.draw(loop_network, pos=pos_xy, labels=pos_z)
    labels = nx.get_edge_attributes(loop_network, "gradient_mdm")
    labels = {(e[0], e[1]): labels[e] for e in labels}
    nx.draw_networkx_edge_labels(loop_network, pos_xy, edge_labels=labels)
    return f


# draw graph in a tree shape
def drawGraphTree(X):
    # draw new graph
    outfalls = [x for x in X.nodes if X.nodes[x]["type"].lower() in ["outfall"]]
    G = nx.DiGraph()
    G.add_nodes_from(X.nodes())
    G.add_edges_from(X.edges())
    # pos = graphviz_layout(G, prog='dot', args='')
    # plt.figure(figsize=(20,10))
    # nx.draw(G, pos, node_size=20, alpha=0.5, node_color="blue", with_labels=False)
    # nx.draw(G.subgraph([n for n in G.nodes if n in outfalls]), pos,\
    #     node_size=20, alpha=0.5, node_color="red", with_labels=True)
    # plt.title('optimised sewer network: no of components ' + str(nx.number_connected_components(nx.Graph(G))))
    return G


########################################################
# Others
########################################################


def ensure_dir(directory):
    # ensure input path is a directory and if not, create a new one
    if not os.path.exists(directory):
        os.makedirs(directory)
    return None


# read outfall levels given csv
# updated 13/08/2019
# included the temporal resolution, and identify the number of outfalls auto
def readOutfallLevels(outfall_level_filename, hours):
    length = hours * 60  # min
    fl = pd.read_csv(outfall_level_filename, skiprows=5, nrows=1)
    n = fl["G_NPROFILES"].values[0]
    fl = pd.read_csv(outfall_level_filename, skiprows=7, nrows=n)
    ids2outfalls = {str(i + 1): iid for i, iid in enumerate(fl["L_NODEID"])}
    fl = pd.read_csv(outfall_level_filename, skiprows=7 + n + 1, index_col=[0])
    fl = fl.rename(ids2outfalls, axis="columns")
    t_res = int(fl.index[1][-8:-6]) * 60 + int(
        fl.index[1][-5:-3]
    )  # data resolution temporal
    length = int(length / t_res)
    levels = fl.iloc[: length + 1, :].mean()
    return levels.to_dict()


import collections
import copy
import datetime

# tools
import os
import pickle
import sys
import time
from collections import defaultdict
from inspect import signature
from operator import itemgetter
from random import randint

import matplotlib._pylab_helpers

# ploting
import matplotlib.pyplot as plt

########################################################
# Packages
########################################################
# cores
import networkx as nx
import numpy as np
import pandas as pd
import patsy
import seaborn as sns
import statsmodels.api as sm

# computations
import sympy
from matplotlib.cm import get_cmap
from scipy import stats
from scipy.signal import convolve
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

# min frequent subgraph
# from parsemis_wrapper import ParsemisMiner
# logistic regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

########################################################
# functions
########################################################


def readTablemq(X, filename, version, tablepath):
    # read calibration table
    # !NOTE: the calculaiton method might be wrong, need more physical meaning
    # output:
    #           d: all nodes
    #           f_d: only flooding nodes
    np.seterr(divide="ignore", invalid="ignore")
    table = pd.read_csv(tablepath + filename, index_col=[0])
    d = table.copy()
    d.loc[:, "d"] = d.loc[:, "chambvolume_m3"].values / d.loc[:, "chambarea_m2"].values
    d.loc[:, "NS"] = (
        d.loc[:, "n_ROmq_m3ds"].values * 300 / d.loc[:, "chambvolume_m3"].values
    )
    d.loc[:, "PS"] = d.loc[:, "nc_Qmq_m3ds"].values / d.loc[:, "Qmax_m3ds"].values
    d.loc[:, "OT"] = d.loc[:, "o_Tmq_yr"]
    d2 = d[["NS", "PS", "d", "OT", "n_F", "n_FVm_m3"]]
    d2 = d2.replace([np.inf, -np.inf], np.nan)
    d2 = d2.dropna(how="any", axis=0)
    d2["n_F"] = d2.loc[:, "n_F"].astype(int)
    d2["Event"] = filename
    d2 = d.dropna(how="any")
    # flooding data
    f_nodes = list(d.loc[d2["n_F"] > 0, "n_F"].index)
    f_d = d2.loc[f_nodes, :]
    print("there are " + str(len(f_nodes)) + " flooding nodes!")
    return d2  # , f_d


# NEW! 21/01/2019
# consider independent time series (with regard to maximum timing)
def readTable(X, keyword, version):
    # read calibration table
    # !NOTE: the calculaiton method might be wrong, need more physical meaning
    # output:
    #           d: all nodes
    #           f_d: only flooding nodes
    np.seterr(divide="ignore", invalid="ignore")
    table = pd.read_csv(
        r"C:\Users\u0107727\Desktop\FloodMAPPING\gent\Calibration events\\"
        + version
        + "\Calibration_"
        + keyword
        + ".csv",
        index_col=[0],
    )
    d = table.copy()
    d.loc[:, "d"] = d.loc[:, "chambvolume_m3"].values / d.loc[:, "chambarea_m2"].values
    d.loc[:, "NS"] = (
        d.loc[:, ["n_ROmq_m3ds", "n_ROmro_m3ds"]].max(axis=1).values
        * 300
        / d.loc[:, "chambvolume_m3"].values
    )
    d.loc[:, "PS"] = (
        d.loc[:, ["nc_Qmq_m3ds", "nc_Qmo_m3ds"]].max(axis=1).values
        / d.loc[:, "Qmax_m3ds"].values
    )
    d.loc[:, "OT"] = d.loc[:, ["o_Tmq_yr", "o_Tmo_yr"]].max(axis=1)
    d = d[["NS", "PS", "d", "OT", "n_F", "n_FVm_m3"]]
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(how="any", axis=0)
    d["n_F"] = d.loc[:, "n_F"].astype(int)
    d["Event"] = keyword
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(how="any", axis=0)
    # flooding data
    f_nodes = list(d.loc[d["n_F"] > 0, "n_F"].index)
    f_d = d.loc[f_nodes, :]
    print("there are " + str(len(f_nodes)) + " flooding nodes!")
    return d, f_d


########################################################
# frequent graph mining
########################################################


def mineFrequentSubgraph(graphs, minfreq, undirectional):
    # mine frequent graphs using parsemis wrerapper
    if undirectional:
        frequent_graphs = ParsemisMiner(
            "FrequentSubgraph", debug=True, mine_undirected=undirectional
        ).mine_graphs(
            graphs,
            minimum_frequency=minfreq,
            find_paths_only=False,
            single_rooter=False,
            close_graph=True,
            store_embeddings=False,
        )
    else:
        frequent_graphs = ParsemisMiner(
            "FrequentSubgraph", debug=True, mine_undirected=undirectional
        ).mine_graphs(
            graphs,
            minimum_frequency=minfreq,
            find_paths_only=False,
            single_rooter=False,
            close_graph=True,
            store_embeddings=True,
        )
    # get nodes
    frequent_nodes = [g.graph.nodes for g in frequent_graphs]
    frequent_nodes = set([j for i in frequent_nodes for j in i])
    return frequent_graphs, frequent_nodes


def mineFrequentNode(X, f_datas, T, event_thres, minfreq):
    # mine frequent node using pure stastitics (appearing frequency -  no spatial correlation is considered)
    # input:
    #       X: networkx graph
    #       f_data = []: list of all historical flooding recod
    #       T = [-1, 2] : the return period range for mining frequent nodes
    #       event_thres = 3: the minimum historical flood occured in the outfall subcatchment
    #       minfreq = '50%': the min frequency for a node to be identified as frequent
    outfalls = [x for x in X.nodes if X.nodes[x]["type"].lower() in ["outfall"]]
    frequent_nodes = []
    f_data = pd.concat([data for data in f_datas])
    f_data = f_data.loc[(f_data.OT > T[0]) & (f_data.OT <= T[1]), :]
    count_nodes = f_data.groupby(f_data.index).count()["Event"]
    for o in outfalls:
        o_nodes = nx.ancestors(X, o)
        f_nodes = set(count_nodes.index) & o_nodes
        f_count_nodes = count_nodes.loc(axis=0)[list(f_nodes)]
        event = f_count_nodes.max()
        if event > event_thres:
            min_event = int(np.ceil(int(minfreq[:2]) / 100 * event))
            o_frequent_nodes = list(f_count_nodes[f_count_nodes >= min_event].index)
            frequent_nodes.append(o_frequent_nodes)
    return list([j for i in frequent_nodes for j in i])


def mineFrequentForest(X, f_datas, T, event_thres, minfreq):
    # mine frequent forest using parsemis_wrapper (spatial propogation pattern that appears frequrntly)
    # input:
    #       X: networkx graph
    #       f_data = []: list of all historical flooding recod
    #       T = [-1, 2] : the return period range for mining frequent nodes
    #       event_thres = 3: the minimum historical flood occured in the outfall subcatchment
    #       minfreq = '50%': the min frequency for a node to be identified as frequent
    # !NOTE: the secon output, frequent node is not used
    outfalls = [x for x in X.nodes if X.nodes[x]["type"].lower() in ["outfall"]]
    forest = []
    event = 0
    for data in f_datas:
        e_nodes = list(data.loc[(data.OT > T[0]) & (data.OT <= T[1]), :].index)
        G = nx.Graph(id=event)
        G.add_nodes_from(X.nodes())
        G.add_edges_from(X.edges(), label="appears_in")
        e_graph = G.subgraph(e_nodes)
        graphs = (e_graph.subgraph(n) for n in nx.connected_components(e_graph))
        forest.append([tree for tree in graphs])
        event += 1
    forest = list([j for i in forest for j in i])
    # compose forest
    forest_composed = nx.compose_all(forest)
    # deocompose forest
    forest_cluster_nodes = [n for n in nx.connected_components(forest_composed)]
    # frequent forest
    frequent_forest = []
    frequent_nodes = set()
    for c_nodes in forest_cluster_nodes:
        c_trees = [tree for tree in forest if len(set(tree.nodes) & c_nodes) > 0]
        event = len(c_trees)
        if event > event_thres:  # only mine when at least 3 times flooded
            c_frequent_graphs, c_frequent_nodes = mineFrequentSubgraph(
                c_trees, minfreq, undirectional="True"
            )
            print(
                "\t",
                int(np.ceil(int(minfreq[:2]) / 100 * event)),
                "/",
                event,
                c_frequent_nodes,
            )
            frequent_forest.append(c_frequent_graphs)
            frequent_nodes.update(c_frequent_nodes)
    frequent_forest = [ff.graph for f in frequent_forest for ff in f]
    return frequent_forest, frequent_nodes


########################################################
# Others - compatibility issue
########################################################


# read_gpickle when encountering compatibility issues
def read_gpickle(filename):
    import pickle

    with open(filename, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        p = u.load()
    return p


# write_gpickle when encountering compatibility issues in python3
def write_gpickle(G, filename):
    import pickle

    with open(filename, "wb") as f:
        p = pickle.dump(G, f, protocol=2)
    return p


# read_pickle when encountering compatibility issues
def read_pickle(filename):
    import pickle

    with open(filename, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        p = u.load()
    return p


###############################################
# run model - 2019-08-10
################################################
# 1 run graph model
def runGraphModel(X, rgRR, rg_id):
    def linearReservior(R, A):
        # R is effective rainfall, m^3/hr - numpy array
        # A is reservior constant - 15hr
        # Q is m3/s
        A = 1.0 / A
        Q = np.zeros(len(R))
        for i in range(1, len(R)):
            Q[i] = Q[i - 1] * np.exp(-A * 5.0) + R[i - 1] * (1 - np.exp(-A * 5.0))
        return Q / 3600.0

    # sim_n_RR
    sim_n_RR = {n: rgRR.loc[:, str(i.rg_id)].values for n, i in rg_id.iterrows()}
    sim_n_RR_dur = len(rgRR)
    # sim_n_RO = I(R,t)cont_area
    sim_n_RO = {
        n: linearReservior(sim_n_RR[n] / (10**3) * X.nodes[n]["cont_area_m2"], 15.0)
        if n in sim_n_RR.keys() and X.nodes[n]["type"].lower() not in ["outfall"]
        else np.zeros(shape=(sim_n_RR_dur,))
        for n in X.nodes
    }
    # sim_nc_RO = I(R,t)us_area
    # sim_nc_Q = I(R,tc)us_area
    sim_nc_RO = {}
    sim_nc_Q = {}
    XX = X.reverse()
    for n in XX.nodes:
        ns = list(nx.dfs_postorder_nodes(XX, n))
        # subcatchment runoff - accumulation in space (sum)
        ns_ = set(ns) - (
            set(ns) - set(sim_n_RO.keys())
        )  # filter out nodes that have no inflow
        RO_t = np.array([sim_n_RO[nsi] for nsi in ns_]).sum(axis=0)
        # subcatchment discharge - accumulation in time (convolve)
        if X.nodes[n]["tc_min"]:
            window = int(X.nodes[n]["tc_min"] / 5)
            if window > 1:
                Q_tc = np.convolve(
                    RO_t,
                    np.ones(
                        window,
                    )
                    / window,
                    mode="full",
                )[: RO_t.size]
            else:
                Q_tc = RO_t
            sim_nc_RO[n] = RO_t
            sim_nc_Q[n] = Q_tc
    # sim_o_T =
    outfalls = [x for x in X.nodes if X.nodes[x]["type"].lower() in ["outfall"]]
    sim_o_T = {o: np.nan * np.zeros(shape=(sim_n_RR_dur,)) for o in outfalls}
    for o in outfalls:
        ns = list(nx.dfs_postorder_nodes(XX, o))
        # subcatchment rainfall rate - average in space (sum)
        ns_ = set(ns) - (
            set(ns) - set(sim_n_RR.keys())
        )  # filter out nodes that have no inflow
        RR_t = np.array([sim_n_RR[nsi] for nsi in ns_]).mean(axis=0)
        if not np.any(np.isnan(RR_t)):
            # subcatchment rainfall rate - average in time (convolve over concentrationt time)
            if X.nodes[o]["tc_min"]:
                window = int(X.nodes[o]["tc_min"] / 5)
                if window > 1:
                    RR_tc = np.convolve(
                        RR_t,
                        np.ones(
                            window,
                        )
                        / window,
                        mode="full",
                    )[: RR_t.size]
                else:
                    RR_tc = RR_t
                I_I1month = I_idf(1.0 / 12.0, X.nodes[o]["tc_min"])
                sim_o_T[o] = np.array(
                    [
                        T_idf(intensity, X.nodes[o]["tc_min"])
                        if intensity > I_I1month
                        else 0.0
                        for intensity in RR_tc
                    ]
                )
    sim_n_RO = pd.DataFrame.from_dict(sim_n_RO)
    sim_n_RO.index = rgRR.index
    sim_nc_RO = pd.DataFrame.from_dict(sim_nc_RO)
    sim_nc_RO.index = rgRR.index
    sim_nc_Q = pd.DataFrame.from_dict(sim_nc_Q)
    sim_nc_Q.index = rgRR.index
    sim_o_T = pd.DataFrame.from_dict(sim_o_T)
    sim_o_T.index = rgRR.index
    return sim_n_RO, sim_nc_RO, sim_nc_Q, sim_o_T


# 2 get predictors for logsitc regression
def getPredictors(X, sim_n_RO, sim_nc_RO, sim_nc_Q, sim_o_T):
    # static FIss
    table_Cs = pd.DataFrame.from_dict(dict(X.nodes(data=True)), orient="index")
    table_Cs.loc[:, "d"] = (
        table_Cs.loc[:, "chambvolume_m3"] / table_Cs.loc[:, "chambarea_m2"]
    ).rename("d")
    table_Cs = table_Cs[["Outfall", "d", "Cat", "Qo_m3ds", "us_area_m2"]]
    # dynamic FIs
    outfalls = [x for x in X.nodes if X.nodes[x]["type"].lower() in ["outfall"]]
    table_Qs = sim_nc_Q.max().to_frame(name="nc_Qmq_m3ds")
    table_Qs = table_Qs.drop([o for o in outfalls if o in table_Qs.index])
    for n in table_Qs.index:
        mq_idx = sim_nc_Q[n].idxmax()
        table_Qs.loc[n, "flooding_time"] = mq_idx
        table_Qs.loc[n, "nc_ROmq_m3ds"] = sim_nc_RO.loc[mq_idx, n]
        table_Qs.loc[n, "n_ROmq_m3ds"] = sim_n_RO.loc[mq_idx, n]
        table_Qs.loc[n, "o_Tmq_yr"] = sim_o_T.loc[mq_idx, X.nodes[n]["Outfall"]]
    # all
    d = pd.concat([table_Cs, table_Qs], axis=1)
    d.loc[:, "NS"] = (
        d.loc[:, "nc_ROmq_m3ds"].values * 300 / d.loc[:, "us_area_m2"].values * 1000.0
    )
    d.loc[:, "PS"] = d.loc[:, "nc_Qmq_m3ds"].values / d.loc[:, "Qo_m3ds"].values
    d.loc[:, "OT"] = d.loc[:, "o_Tmq_yr"]
    d.loc[:, "OT"] = d.loc[:, "OT"].round(1)
    d.loc[d.OT < 0.1, "OT"] = 0.01
    d = d[["NS", "PS", "d", "OT", "Cat", "flooding_time"]]
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(how="any", axis=0)
    return d


def runGraphModel_2options(X, rgRR, rg_id):
    def WallingfordMethod(A, S, i10):
        # S - slope (m/m)
        # A - the area (m2) of the surface, for example the area of the paved surface
        # i10 - running ten minute average of rainfall intensity
        # Now using T2 rainfall 73mm/hr
        if S < 0.002:
            S = 0.002
        if A < 1000:
            A = 1000
        if A > 10000:
            A = 10000
        C = 0.117 * S ** (-0.13) * A ** (0.24)
        i_ = 0.5 * (1 + i10)
        k = C * i_ ** (-0.39)
        return k

    def linearReservior(R, A):
        # R is effective rainfall, m^3/hr - numpy array
        # A is reservior constant - hr
        # Q is m3/s
        A = 1.0 / A
        Q = np.zeros(len(R))
        for i in range(1, len(R)):
            Q[i] = Q[i - 1] * np.exp(-A * 5.0 / 60) + R[i - 1] * (
                1 - np.exp(-A * 5.0 / 60)
            )
        return Q / 3600.0

    ###################################
    # Assign rainfall
    ###################################
    # sim_n_RR: rainfall profile at node
    sim_n_RR_dur = len(rgRR)
    sim_n_RR = {i[0]: rgRR.loc[:, str(i[0])].values for n, i in rg_id.iterrows()}
    #
    ###################################
    # Bottom up method
    ###################################
    # sim_n_RO: Runoff generation and routing at node
    #   1. fixed runoff method (sim_n_RO = I(R,t)*cont_area * 0.8), not output, integrated in the enxt step
    #   2. single linear reservior following wallingford method, rainfall = T2
    sim_n_RO = {
        n: linearReservior(
            sim_n_RR[n] / (10**3) * X.nodes[n]["cont_area_m2"] * 0.8,
            WallingfordMethod(
                X.nodes[n]["cont_area_m2"], X.nodes[n]["subcatslope"], 73.0
            ),
        )
        for n in sim_n_RR.keys()
    }  # m3/s
    # sim_nc_RO: Runoff accumulation in lumped sewer subnetwork
    #   simple sum
    # sim_nc_Q: Flow routing in lumped sewer subnetwork
    #   single linear reservior with tc as time constant
    sim_nc_RO = {}
    sim_nc_Q = {}
    XX = X.reverse()
    for n in XX.nodes:
        if X.nodes[n]["tc_min"] > 0:
            # upstream catchment
            ns = list(nx.dfs_postorder_nodes(XX, n))
            # include only nodes with contributing subcatchment
            ns_ = set(ns) - (set(ns) - set(sim_n_RO.keys()))
            # runoff - accumulation in lumped sewer subnetwork manholes
            if len(ns_) > 0:
                RO_t = np.array([sim_n_RO[nsi] for nsi in ns_]).sum(axis=0)  # m3/s
                # Flow routing in lumped sewer subnetwork
                Q_t = linearReservior(RO_t * 3600, X.nodes[n]["tc_min"] / 60)  # m3/s
            else:
                RO_t = np.zeros(shape=(sim_n_RR_dur,))
                Q_t = np.zeros(shape=(sim_n_RR_dur,))
            sim_nc_RO[n] = RO_t[:]
            sim_nc_Q[n] = Q_t[:]
    ###################################
    # Top down method
    ###################################
    # sim_nc_II: mean  instatenous rainfallprofile at lumped sewer subnetwork
    #   simple average
    sim_nc_IT = {}
    # sim_nc_ITC: rainfall over concentration time at lumppedsewer subnetwork
    #   simple convolve
    sim_nc_ITC = {}
    for n in XX.nodes:
        if X.nodes[n]["tc_min"]:
            # upstream catchment
            ns = list(nx.dfs_postorder_nodes(XX, n))
            # include only nodes with contributing rg
            ns_ = set(ns) - (set(ns) - set(sim_n_RR.keys()))
            if len(ns_) > 0:
                # rainfall averaging
                I_t = np.array([sim_n_RR[nsi] for nsi in ns_]).mean(
                    axis=0
                )  # should not include 0
                # rainfall rounting
                window = int(X.nodes[n]["tc_min"] / 5)
                I_tc = np.convolve(
                    I_t,
                    np.ones(
                        window,
                    )
                    / window,
                    mode="full",
                )[: I_t.size]
            else:
                I_t = np.zeros(shape=(sim_n_RR_dur,))
                I_tc = np.zeros(shape=(sim_n_RR_dur,))
            sim_nc_IT[n] = I_t[:]
            sim_nc_ITC[n] = I_tc[:]
    ###################################
    # shared method
    ###################################
    # sim_o_T
    outfalls = [x for x in X.nodes if X.nodes[x]["type"].lower() in ["outfall"]]
    sim_o_T = {o: np.nan * np.zeros(shape=(sim_n_RR_dur,)) for o in outfalls}
    for o in outfalls:
        I_tc = sim_nc_ITC[o]
        I_I1month = I_idf(1.0 / 12.0, X.nodes[o]["tc_min"])
        sim_o_T[o] = np.array(
            [
                T_idf(intensity, X.nodes[o]["tc_min"]) if intensity > I_I1month else 0.0
                for intensity in I_tc
            ]
        )
    # result
    sim_temp = {o: np.nan * np.zeros(shape=(sim_n_RR_dur,)) for o in outfalls}
    for o in outfalls:
        I_tc = sim_nc_ITC[o]
        tc = X.nodes[o]["tc_min"]
        sim_temp[o] = I_tc / tc

    sim_nc_RO = pd.DataFrame.from_dict(sim_nc_RO)
    sim_nc_RO.index = rgRR.index
    sim_nc_Q = pd.DataFrame.from_dict(sim_nc_Q)
    sim_nc_Q.index = rgRR.index
    sim_nc_IT = pd.DataFrame.from_dict(sim_nc_IT)
    sim_nc_IT.index = rgRR.index
    sim_nc_ITC = pd.DataFrame.from_dict(sim_nc_ITC)
    sim_nc_ITC.index = rgRR.index
    sim_o_T = pd.DataFrame.from_dict(sim_o_T)
    sim_o_T.index = rgRR.index
    sim_temp = pd.DataFrame.from_dict(sim_temp)
    sim_temp.index = rgRR.index
    return sim_nc_RO, sim_nc_Q, sim_nc_IT, sim_nc_ITC, sim_o_T, sim_temp
