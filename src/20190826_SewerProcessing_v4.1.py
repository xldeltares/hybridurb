"""
Created on 27/08/2019 
Modified from FloodMAPPING\20190213_SewerProcessing1_v3.1.py

@author: Xiaohan Li

version: python 3

Objective: sewer graph processing to be ready for calibration

Input: 
    - picked graph 
    - Water level file when applicable

Output: save as gpickle (X = nx.read_gpickle(sewer_dir + "sewer_graph_networkx_v4.1.gpickle"))

Methodology: optimise until criteria is satisfied
    
X - MultiDiGraph 
    NODE related
        'chambvolume_m3', 'chambarea_m2', 'cont_area_m2',
    DS pipe related 
        'Qo_m3ds', 'Qcap_m3ds', 'Qmax_m3ds',  'Qmin_m3ds'

"""


import sys

sys.path.append("./FloodMAPPING")
from floodMAPPING_funcs_p3 import *

##########################################################
#  path
##########################################################
# raw network
path = "J:\\LocalWorkspace\\hybridtest\\"
X = read_gpickle(path + "sewer\\" + "sewer_graph_networkx_v4.0.gpickle")

# add boundry condition
# outfall_level_filename = path + "sewer\\" + "waterhoogten-normaal.csv"
# levels = readOutfallLevels(outfall_level_filename, hours = 6)
# levels = {k.strip():levels[k] for k in levels.keys()}
# outfalls = [x for x in X.nodes if X.nodes[x]['type'].lower() in ['outfall']]
X = reverseOutfall(X)  # , levels = levels)

# -------------------- prepare for optimisation --> DAG

# 1. start
X1 = X.copy()

# 2. compute cost
X2 = calcCost(X1)

# 3. multiDiGraph 2 uniDiGraph
X3 = multi2uniDiGraph(X2)

# 4. make DAG based on Zloss1 + Zloss2 + Zloss3
X4 = makeDAG(X3)

# ------------------------ optimisation of DAG

count = 0
while count < 10:
    # 5. split multi pathways based on minimum cost path
    X5 = makeTREE(X4)

    # 6. recover leftover leaves
    X6 = recoverLEAF(X5, X3)  # NOTE! using X3 not X4

    X_optimised = X6.copy()
    if len(nx.difference(X4, X_optimised).edges()) != 0:
        X4 = X_optimised.copy()
        count += 1
    else:
        print(f"OPTIMISATION FINISHED: number of loop {count+1}")
        break
    if count == 9:
        print(
            f"OPTIMISATION FINISHED (Maixmum iteration reached): number of loop {count+1}"
        )

# 7. add outfall
X7 = addOutfall(X6)

# ------------------------ computation of flow params

# 8. compute tc
X8 = calcDAGTc(X7, method="CW_cap")

#  9. calculate static Qo
X9 = calcDAGStaticFI(X8)
X9 = fillNegCapacity_nodes(X9, param="Qo_m3ds")

# pickle
nx.write_gpickle(X9, path + "sewer\\" + "sewer_graph_networkx_v4.1.gpickle")
print("Graph created: " + path + "sewer\\" + "sewer_graph_networkx_v4.1.gpickle")
del [X1, X2, X3, X4, X5, X6, X7, X8]
##########################################################
#  graph figures
##########################################################
G = drawGraphTree(X9)

# nx.draw(G)

# save all the figures
fig_folder = (
    path + "FloodMAPPING\\Figures_" + datetime.datetime.now().strftime("%d-%m-%y")
)
figures = [
    manager.canvas.figure
    for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()
]

if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

for i, figure in enumerate(figures):
    figure.savefig(fig_folder + "\\figureA%d.png" % i)

print("Figures saved at: " + fig_folder)

##########################################################
#  save csv for QGIS
##########################################################
# all nodes
df1 = pd.DataFrame.from_dict(
    dict(X9.nodes(data="Outfall")), orient="index", columns=["Outfall"]
)
df2 = pd.DataFrame.from_dict(
    dict(X9.nodes(data="tc_min")), orient="index", columns=["tc_min"]
)
df3 = pd.DataFrame.from_dict(
    dict(X9.nodes(data="us_area_m2")), orient="index", columns=["us_area_m2"]
)
df = pd.concat([df1, df2, df3], axis=1)
df.to_csv(path + "sewer\\" + "sewer_graph_networkx_v4.1.csv")
# outfall
df = pd.DataFrame(
    columns=[
        "Number of nodes",
        "Effective upstream area (ha)",
        "Concentraiton time (min)",
    ]
)
outfalls = [x for x in X9.nodes if X9.nodes[x]["type"].lower() in ["outfall"]]
XX = X9.reverse()
for o in outfalls:
    df.loc[o, :] = (
        len(nx.dfs_tree(XX, o).nodes),
        round(X9.nodes[o]["us_area_m2"] / 10000, 1),
        X9.nodes[o]["tc_min"],
    )

df = df[df.iloc[:, 1] > 0]
df["ID"] = np.arange(len(df))
df.to_csv(path + "sewer\\" + "sewer_graph_networkx_v4.1_outfalls.csv")
print(sys.path)


# plot xy
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Flow")
pos = {xy: X9.nodes[xy]["geo"][0:2] for xy in X9.nodes()}

# base
nx.draw(
    X,
    pos=pos,
    node_size=0,
    with_labels=False,
    arrows=False,
    node_color="gray",
    edge_color="silver",
    width=1,
)

nodelist_types = X9.nodes.data("type")
node_names = []
type_list = []
colour_list = []
for a, b in nodelist_types:
    node_names.append(a)
    type_list.append(b)
    if b == "outfall":
        colour_list.append("green")
    else:
        colour_list.append("red")


# nodes dag
nx.draw_networkx_nodes(
    X9, pos=pos, nodelist=G.nodes(), node_size=40, node_color=colour_list
)
# edges dag
edge_width = [X9.nodes[d[0]]["us_count"] / 100 for d in X9.edges()]
nx.draw_networkx_edges(
    X9,
    pos=pos,
    edgelist=X9.edges(),
    arrows=False,
    width=[float(i) / (max(edge_width) + 0.1) * 20 + 0.5 for i in edge_width],
)
# nx.draw_networkx_nodes(X9, pos=pos, nodelist=X9.nodes(), node_size=50, node_shape="*", node_color=colour_list)

plt.savefig("J:\\Plotspace\\MapThicknesses.png", bbox_inches="tight")
