"""
Created om 26/11/2018
Updated 12/02/2019
Updated 04/09/2016 for the v4.0: added slope of subcatchment and corrected cont_area_m2 as abs1+abs2
Updated 21/09/2021 - 28/09/2021 to v4.1 for use with Delft3D (Mees Radema)
Updated 14/10/2021 - Started reworking for removal of GPD, replacement with Geopandas

@author: Xiaohan Li

version: python 3

Objective: Create graph using sewer networkfiles 
(NEW!)

Input: 
    - Nodes 
    - Conduits 
    - Strucures (weirs, pumps, user control, etc merged into one)
    - Subcatchment
    - Subcatchment-RGs in csv having informaiton of node_id, subcatid, rg_id (created by intersect RG polygons with subcat)


Output: save as png and gpickle (X = nx.read_gpickle(sewer_dir + "sewer_graph_networkx_v2.0.gpickle"))

X - MultiDiGraph 
graph contains:
node:  
    - type: outfall/manhole/storage (lower letters)
    - geo: x,y cordinates and z (m) for chamber floor elevation
    - chambarea_m2: manhole chamber area (m2)
    - chambfloor_m: manhole chamber floor (m)
    - chambroof_m: manhole chamber roof (m) == ground level (ground_lev) == DEM
    - chambvolume_m3: manhole chamber volume (m3)
    - subcat_id: subcatchment id in list (multiple subcatments to one node)
    - RG_id: Rain gauge id  
    - cont_area_m2: ubcatchment runoff area into each node (m2) (multiple subcatments to one node)
    calculated by a_abs_1 + a_abs_2 + 0.8 * a_abs_3
    - (!NOTE NEW) gross_area_m2: ubcatchment area (all do not consider runoff) into each node (m2) (multiple subcatments to one node)


edges: 
    - direction upstream node --> donwstream node
    - id: link id
    - sys_type: combined/storm...
    - type: cond/weir/pump/...
    - us_invert_m: m (hydraulic structure depends on types)
    - ds_invert_m: m (hydraulic structure depends on types)
    - length_m: pipe length - m (hydraulic structure 0)
    - gradient_mdm: -m/m (hydraulic structure 0)
    - rcw_mm: rought ness colebrook white (bot_rcw) (hydraulic structure None)
    - capacity_m3ds: pipe capacity  - m3/s (hydraulic structure None)
    - shape: shape of the pipe (shape_1) (hydraulic structure None)
    - width_m: shape of the pipe (condeidth) (hydraulic structure None)
    - height_m: shape of the pipe (condheight) (hydraulic structure None)
    - sediment_m: depth of sediment deposite in the sewer (sed_depth) (hydraulic structure None)

! NOTE: for sluice, if height exist, then treat as a normal link, if height == 0, then delete the edge (no water come through)
"""


import sys

sys.path.append("./FloodMAPPING")
from floodMAPPING_funcs_p3 import *

# import arcpy


##########################################################
#  path
##########################################################
path = "D:\\LocalWorkspace\\hybridtest"
sewer_dir = path + "\\sewer\\"

# network files
shp1_filename = sewer_dir + "Compartment.shp"  # compartment
shp1b_filename = sewer_dir + "Compartment_Outlet.shp"  # new
shp2_filename = sewer_dir + "Pipe.shp"  # used to be conduit
shp3a_filename = sewer_dir + "Pumps.shp"  # formerly "Structures.shp"
shp3b_filename = sewer_dir + "Weirs.shp"
shp4_filename = sewer_dir + "Subcatchment.shp"
rgid_filename = (
    sewer_dir + "Subcatchment_RGs.csv"
)  # must contain 0,rg_id,node_id, subcat_id

conn_filename = (
    sewer_dir + "Connections.shp"
)  # Sewer connections shapefile needed for Delft3D use

##########################################################
# sewer network
##########################################################

# nodes
points_gpd = gpd.read_file(shp1_filename)
featureCount = len(points_gpd)
x = []
y = []
kw = []
kw_alt = []
node_chambfloor = []
node_chambroof = []
node_chambarea = []
node_type = []
node_level_overview = {}

for i in range(featureCount):
    kw.append(
        points_gpd["ManholeNam"][i]
    )  # used to be .lower() (and with a different package but I digress)
    node_chambfloor.append(points_gpd["BottomLeve"][i])
    node_chambroof.append(points_gpd["SurfaceLev"][i])
    node_chambarea.append(points_gpd["FloodableA"][i])
    node_type.append(points_gpd["node_type"][i])
    # .lower()) # Maybe this needs to be either normal or Outfall == outlet compartment (merge compartments+outletcomp)

    x.append(points_gpd[i : i + 1].geometry.x[i])
    y.append(points_gpd[i : i + 1].geometry.y[i])

    node_level_overview[points_gpd["ManholeNam"][i]] = points_gpd["BottomLeve"][i]

nodes = {kk: (xx, yy) for kk, xx, yy in zip(kw, x, y)}
node_volumn = (np.array(node_chambroof) - np.array(node_chambfloor)) * np.array(
    node_chambarea
)
node_volumn = {kk: (vv) for kk, vv in zip(kw, node_volumn)}
node_type = {kk: (tt) for kk, tt in zip(kw, node_type)}
node_geo = {kk: ((xx, yy, zz)) for kk, xx, yy, zz in zip(kw, x, y, node_chambfloor)}
node_chambarea = {kk: (vv) for kk, vv in zip(kw, node_chambarea)}
node_chambfloor = {kk: (vv) for kk, vv in zip(kw, node_chambfloor)}
node_chambroof = {kk: (vv) for kk, vv in zip(kw, node_chambroof)}
# node rain gauges  < -- FIX THIS SHIT LATER, it's with the merged stuff and shit.
# df_rgid = pd.read_csv(rgid_filename, index_col=[0])
# df_rgid = df_rgid.groupby('node_id')['rg_id'].apply(lambda x: x.mode().iloc[0])
# node_rgid = {kk:df_rgid.loc[kk] for kk in kw if kk in df_rgid.index}

# links & structures
driver = ogr.GetDriverByName("ESRI Shapefile")
# roughness = 0.003 # This is for D-flow Waardenburg
dataSource = driver.Open(shp2_filename, 0)
layer = dataSource.GetLayer()
featureCount = layer.GetFeatureCount()
lid = []
us = []
ds = []
sys_type = []
link_type = []
link_shape = []
capacity = []  # m3/s
gradient = []  # m/m
length = []  # m
width = []  # m
height = []  # m
sediment = []  # m
roughness = []  # mm
us_invert = []
ds_invert = []

fieldlist = []
layerdefn = layer.GetLayerDefn()

for i in range(layerdefn.GetFieldCount()):
    fdefn = layerdefn.GetFieldDefn(i)
    fieldlist.append(fdefn.name)

for feature in layer:
    link_type.append(feature.GetField("DefName"))  # .lower())
    sys_type.append(feature.GetField("Sewer type"))  # .lower())
    link_shape.append(
        "Circle"
    )  # feature.GetField("shape_1")) < Should probably be expanded to use DefName
    capacity.append(None)  # feature.GetField("capacity"))
    gradient.append(
        (feature.GetField("Level sourc") - feature.GetField("Level targe"))
        / feature.GetField("Length")
    )  # calculate gradient in same line
    length.append(feature.GetField("Length"))
    width.append(feature.GetField("Width"))
    height.append(feature.GetField("Width"))  # assume Width == height
    sediment.append(
        None
    )  # feature.GetField("sed_depth")/1000.0) #Dflow does nothing with this sediment bs
    roughness.append(
        3.0
    )  # feature.GetField("bot_rcw")) # For some reason doesn't export and has to be in mm instead of m. Assume concrete value given in D-Flow, look at again later.
    us.append(feature.GetField("From compar"))
    ds.append(feature.GetField("To compartm"))
    lid.append(feature.GetField("Name"))
    us_invert.append(feature.GetField("Level sourc"))
    ds_invert.append(feature.GetField("Level targe"))


str_invert = {
    "compnd": "start_lev",
    "fixpmp": "on_level",
    "flap": "invert",
    "orific": "invert",
    "rotpmp": "on_level",
    "screen": "crest",
    "scrpmp": "on_level",
    "sluice": "invert",
    "weir": "crest",
    "vortex": "start_lev",
    "vsgate": "invert",
}

# Structure index
connections = gpd.read_file(conn_filename)
driver = ogr.GetDriverByName("ESRI Shapefile")
dataSource = driver.Open(conn_filename, 0)
layer = dataSource.GetLayer()
featureCount = layer.GetFeatureCount()

fieldlist = []
layerdefn = layer.GetLayerDefn()

for i in range(layerdefn.GetFieldCount()):
    fdefn = layerdefn.GetFieldDefn(i)
    fieldlist.append(fdefn.name)

structure_overview = {}
for feature in layer:
    name = feature.GetField("Name")
    type = feature.GetField("Sewer Speci")  # 0
    upstream_id = feature.GetField("From compar")  # 1
    downstream_id = feature.GetField("To compartm")  # 2
    feature_length = feature.GetField("GeomLength")  # 3
    upstream_level = node_level_overview[upstream_id]  # 4
    downstream_level = node_level_overview[downstream_id]  # 5

    structure_overview[name] = [
        type,
        upstream_id,
        downstream_id,
        feature_length,
        upstream_level,
        downstream_level,
    ]

# Pumps
driver = ogr.GetDriverByName("ESRI Shapefile")
dataSource = driver.Open(shp3a_filename, 0)
layer = dataSource.GetLayer()
featureCount = layer.GetFeatureCount()

fieldlist = []
layerdefn = layer.GetLayerDefn()

for i in range(layerdefn.GetFieldCount()):
    fdefn = layerdefn.GetFieldDefn(i)
    fieldlist.append(fdefn.name)

for feature in layer:
    name = feature.GetField("Name")
    link_type_h = "fixpmp"
    # link_invert_h = feature.GetField(str_invert[link_type_h])
    # if link_type_h == 'sluice':
    #     sluice_h = feature.GetField("opening")
    #     if sluice_h != 0:
    #         pass
    #     else:
    #         continue
    link_type.append(link_type_h)
    sys_type.append("pump")  # feature.GetField("systemtype").lower())
    us.append(structure_overview[name][1])
    ds.append(structure_overview[name][2])
    lid.append(feature.GetField("Name"))
    us_invert.append(structure_overview[name][4])
    ds_invert.append(structure_overview[name][5])
    length.append(structure_overview[name][3])
    gradient.append(None)
    capacity.append("Capacity")
    width.append(None)
    height.append(None)
    sediment.append(None)
    roughness.append(None)
    link_shape.append(None)

# Weirs
driver = ogr.GetDriverByName("ESRI Shapefile")
dataSource = driver.Open(shp3b_filename, 0)
layer = dataSource.GetLayer()
featureCount = layer.GetFeatureCount()

fieldlist = []
layerdefn = layer.GetLayerDefn()

for i in range(layerdefn.GetFieldCount()):
    fdefn = layerdefn.GetFieldDefn(i)
    fieldlist.append(fdefn.name)

for feature in layer:
    name = feature.GetField("Name")
    link_type_h = "weir"
    # link_invert_h = feature.GetField(str_invert[link_type_h])
    # if link_type_h == 'sluice':
    #     sluice_h = feature.GetField("opening")
    #     if sluice_h != 0:
    #         pass
    #     else:
    #         continue
    link_type.append(link_type_h)
    sys_type.append("weir")
    us.append(structure_overview[name][1])
    ds.append(structure_overview[name][2])
    lid.append(feature.GetField("Name"))
    us_invert.append(structure_overview[name][4])
    ds_invert.append(structure_overview[name][5])
    length.append(structure_overview[name][3])
    gradient.append(0)
    capacity.append(None)
    width.append(feature.GetField("CrestWidth"))
    height.append(None)
    sediment.append(None)
    roughness.append(None)
    link_shape.append(None)

links = {
    k: [u, d, t, st, s, c, g, l, w, h, sed, r, ui, di]
    for k, u, d, t, st, s, c, g, l, w, h, sed, r, ui, di in zip(
        lid,
        us,
        ds,
        link_type,
        sys_type,
        link_shape,
        capacity,
        gradient,
        length,
        width,
        height,
        sediment,
        roughness,
        us_invert,
        ds_invert,
    )
}


# subcatchments
driver = ogr.GetDriverByName("ESRI Shapefile")
dataSource = driver.Open(shp4_filename, 0)
layer = dataSource.GetLayer()
featureCount = layer.GetFeatureCount()
sub_id = []  # subcatchment node id
subcatid = []  # subcatchment id
sub_slope = []
sub_cont_area = []
sub_runoff_area = []

fieldlist = []
layerdefn = layer.GetLayerDefn()

for i in range(layerdefn.GetFieldCount()):
    fdefn = layerdefn.GetFieldDefn(i)
    fieldlist.append(fdefn.name)

for feature in layer:
    sub_id.append(feature.GetField("Name"))
    subcatid.append(feature.GetField("Input_FID"))
    sub_slope.append(
        0.00006666666
    )  # sub_slope.append(feature.GetField("cat_slope")) #NOTE: obvs not really 0
    sub_cont_area.append(feature.GetField("Shape_Area"))
    sub_runoff_area.append(feature.GetField("Shape_Area"))  # NOTE! NEW (04/09/2019)

subcat_id = defaultdict(list)
for kk, ii in zip(sub_id, subcatid):
    subcat_id[kk].append(ii)

subcat_slope = {kk: (xx) for kk, xx in zip(subcatid, sub_slope)}
node_subcatslope = {
    kk: (
        sum(
            subcat_slope[kkk] if subcat_slope[kkk] is not None else 0
            for kkk in subcat_id[kk]
        )
    )
    for kk in subcat_id
}


subcat_area = {kk: (xx) for kk, xx in zip(subcatid, sub_runoff_area)}
node_area = {
    kk: (
        sum(
            subcat_area[kkk] if subcat_area[kkk] is not None else 0
            for kkk in subcat_id[kk]
        )
    )
    for kk in subcat_id
}

subcat_grossarea = {kk: (xx) for kk, xx in zip(subcatid, sub_cont_area)}
node_grossarea = {
    kk: (
        sum(
            subcat_grossarea[kkk] if subcat_grossarea[kkk] is not None else 0
            for kkk in subcat_id[kk]
        )
    )
    for kk in subcat_id
}

##########################################################
# Graph of sewer network
##########################################################

X = nx.MultiDiGraph()
# add nodes with data as contributing area of the subcatchment
for k in nodes:
    t = 0 if k not in node_type or node_type[k] is None else node_type[k]
    v = 0 if k not in node_volumn or node_volumn[k] is None else node_volumn[k]
    ca = (
        0 if k not in node_chambarea or node_chambarea[k] is None else node_chambarea[k]
    )
    cf = (
        0
        if k not in node_chambfloor or node_chambfloor[k] is None
        else node_chambfloor[k]
    )
    cr = (
        0 if k not in node_chambroof or node_chambroof[k] is None else node_chambroof[k]
    )
    s = (
        0
        if k not in node_subcatslope or node_subcatslope[k] is None
        else node_subcatslope[k]
    )
    a = (
        0 if k not in node_area or node_area[k] is None else node_area[k]
    )  # EDITED: to m2 instead of ha
    ga = (
        0 if k not in node_grossarea or node_grossarea[k] is None else node_grossarea[k]
    )  # ha to square meters
    g = (0, 0, 0) if k not in node_geo or node_geo[k] is None else node_geo[k]
    i = None if k not in subcat_id or subcat_id[k] is None else subcat_id[k]
    # irg = (None if k not in node_rgid or node_rgid[k] is None else node_rgid[k])
    # X.add_node(k, type = t, chambfloor_m = cf,chambroof_m = cr, chambarea_m2 = ca, chambvolume_m3 = v, subcatslope = s, cont_area_m2 = a, gross_area_m2 = ga, geo = g, subcat_id = i, rg_id = irg)
    X.add_node(
        k,
        type=t,
        chambfloor_m=cf,
        chambroof_m=cr,
        chambarea_m2=ca,
        chambvolume_m3=v,
        subcatslope=s,
        cont_area_m2=a,
        gross_area_m2=ga,
        geo=g,
        subcat_id=i,
    )

for k in links:
    X.add_edge(
        links[k][0],
        links[k][1],
        id=k,
        type=links[k][2],
        sys_type=links[k][3],
        shape=links[k][4],
        capacity_m3ds=links[k][5],
        gradient_mdm=links[k][6],
        length_m=links[k][7],
        width_m=links[k][8],
        height_m=links[k][9],
        sediment_m=links[k][10],
        rcw_mm=links[k][11],
        us_invert_m=links[k][12],
        ds_invert_m=links[k][13],
    )

# draw graph
nx.draw_networkx_nodes(X, nodes, node_size=2, node_color="r")
nx.draw_networkx_edges(X, nodes)
plt.title("Sewer network in Graph")
plt.axis("equal")
plt.savefig(sewer_dir + "sewer_graph_networkx_v4.0.png")
# plt.show()
# pickle
nx.write_gpickle(X, sewer_dir + "sewer_graph_networkx_v4.0.gpickle")
# read pickle
# X = nx.read_gpickle(sewer_dir + "sewer_graph_networkx.gpickle")
print("Graph created: " + sewer_dir + "sewer_graph_networkx_v4.0.gpickle")
