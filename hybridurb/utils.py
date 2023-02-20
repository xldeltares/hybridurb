
# io functions
import pickle
def read_gpickle(filename):
    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
    return p

# write_gpickle when encountering compatibility issues in python3
def write_gpickle(G, filename):
    import pickle
    with open(filename, 'wb') as f:
        p = pickle.dump(G, f, protocol=2)
    return p

# read_pickle when encountering compatibility issues
def read_pickle(filename):
    import pickle
    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
    return p

# export to fews
import networkx as nx
import momepy
import geopandas as gpd
from pathlib import Path
def export_to_fews(G: nx.Graph, out_dir:Path):

    # export locations for fews
    # relabeling using xy
    mapping = {}
    for n in G.nodes:
        G.nodes[n]['node_name'] = n
        mapping[n] = G.nodes[n]['geo'][:2]

    _G = nx.relabel_nodes(G,mapping, copy=True)

    _nodes = momepy.nx_to_gdf(_G, points = True, lines = False, spatial_weights=False)
    _nodes[['node_name', 'nodeID', 'geometry']].to_file(Path(out_dir).joinpath('network_nodes.shp'))
    return _nodes



# drawing functions
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
def drawHazardMap(X, y_pred, ft = None):
    # ft: flooding time
    if ft is None:
        y_pred_multi = pd.cut(y_pred.values, [-1, 0.25, 0.5, 0.75, 1], labels = [0, 0.25,0.5,0.75]).astype(float) # categorized
        y_pred_multi = pd.DataFrame(data = y_pred_multi[..., np.newaxis], index = y_pred.index)
        y_pred_multi = y_pred_multi[y_pred_multi>0]
        # plot
        plt.figure(figsize = (5,5.5))
        pos_dict = dict(X.nodes(data = 'geo')); pos = {k: pos_dict[k][:2] for k in pos_dict}
        nx.draw_networkx_nodes(X.nodes,pos,node_size=1,node_color='grey')
        nx.draw_networkx_nodes(y_pred_multi[y_pred_multi>0].index,pos,
                node_color=y_pred_multi[y_pred_multi>0],
                node_size=10,  cmap = 'Reds', vmin = 0, vmax = 1)
        sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin = 0, vmax=1))
        sm._A = []
        clb =  plt.colorbar(sm)
        clb.ax.set_ylabel('probability')
        plt.title("Flood hazard map")
    else:
        y_pred_multi = pd.cut(y_pred.values, [-1, 0.25, 0.5, 0.75, 1], labels = [0, 0.25,0.5,0.75]).astype(float) # categorized #FIXME new: the mean is fixed for y_pred, meaning that confidence intercL IA NOR AUPPOER
        y_pred_multi = pd.DataFrame(data=y_pred_multi[..., np.newaxis], index=y_pred.index)
        y_pred_multi = y_pred_multi[y_pred_multi > 0]
        ft_pred_multi =  pd.cut(ft.values, [-1, 30, 60, 90, 120], labels = [0.5, 1, 1.5, 2]).astype(float)
        ft_pred_multi = pd.DataFrame(data=ft_pred_multi[..., np.newaxis], index=ft.index)
        ft_pred_multi = ft_pred_multi[y_pred_multi>0]
        # plot
        plt.figure(figsize = (11,5.5))
        plt.subplot(1,2,1)
        pos_dict = dict(X.nodes(data = 'geo')); pos = {k: pos_dict[k][:2] for k in pos_dict}
        nx.draw_networkx_nodes(X.nodes,pos,node_size=1,node_color='grey')
        nx.draw_networkx_nodes(y_pred_multi[y_pred_multi>0].index,pos,
                node_color=y_pred_multi[y_pred_multi>0],
                node_size=10,  cmap = 'Reds', vmin = 0, vmax = 1)
        sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin = 0, vmax=1))
        sm._A = []
        clb =  plt.colorbar(sm)
        clb.ax.set_ylabel('probability')
        clb.set_ticks([0, 0.25,0.5,0.75, 1])
        plt.title("Flood hazard map")
        plt.subplot(1,2,2)
        pos_dict = dict(X.nodes(data = 'geo')); pos = {k: pos_dict[k][:2] for k in pos_dict}
        nx.draw_networkx_nodes(X.nodes,pos,node_size=1,node_color='grey')
        nx.draw_networkx_nodes(X, pos, nodelist = list(ft_pred_multi.index),
            node_color=ft_pred_multi,cmap = 'jet_r', vmin = 0, vmax = 2,
            node_size=10)
        sm = plt.cm.ScalarMappable(cmap='jet_r', norm=plt.Normalize(vmin = 0, vmax=2))
        sm._A = []
        clb = plt.colorbar(sm)
        clb.ax.set_ylabel('Leadtime (+hr)')
        clb.set_ticks([0, 0.5, 1, 1.5, 2])
        plt.title("Flood urgency map")


# hydrology functions
import numpy as np
import sympy
def T_idf(intensity,duration):
    # input: intensity in mm/hr
    #        Duration in min
    # output:T in year
    i = float(intensity)
    D = float(duration)/1440.
    beta_a = 10.**(-0.05-0.58*np.log10(D))
    beta_b = 10.**(-0.55-0.58*np.log10(D))
    p_a = 10.**(-1.35-0.58*np.log10(D))
    i_0 = 10.**(-0.05-0.58*np.log10(D))
    p_1 = 10.**(-1.12-0.5*np.log10(D))
    n = 27.
    m = 1.+54.*p_1/(p_1-p_a)
    C = 0.93
    T = n/(m*(p_a*np.exp((i_0-i*C)/beta_a)+(1-p_a)*np.exp((i_0-i*C)/beta_b)))
    return T


def I_idf(T, tc):
    # solve intensity using numerical solver
    # tc in minutes, T in jears
    # D in days
    T = float(T)
    D  = tc/60.0/24.0
    beta_a = 10.0**(-0.05-0.58*np.log10(D))
    beta_b = 10.0**(-0.55-0.58*np.log10(D))
    p_a    = 10.0**(-1.35-0.58*np.log10(D))
    i_0    = 10.0**(-0.05-0.58*np.log10(D))
    p_1    = 10.0**(-1.12-0.50*np.log10(D))
    n      = 27.0
    m      = 1.0 + 54.0*p_1/(p_1-p_a)
    if D < 1.0: C = 0.93
    else: C = 1.0
    x = sympy.Symbol('x')
    func = T - n/(m*(p_a*sympy.exp((i_0-x*C)/beta_a)+(1.0-p_a)*sympy.exp((i_0-x*C)/beta_b)))
    if tc <= 30:
        return sympy.nsolve(func,x,150.0)
    elif tc <= 60:
        return sympy.nsolve(func,x,50.0)
    elif tc <= 120:
        return sympy.nsolve(func,x,30.0)
    else:
        return sympy.nsolve(func,x,10.0)


