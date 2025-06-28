"""
Created om 26/11/2018
Updated 12/02/2019
Updated 04/09/2016 for the v4.0: added slope of subcatchment and corrected cont_area_m2 as abs1+abs2

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
sys.path.append('./FloodMAPPING')
from floodMAPPING_funcs_p3 import *


##########################################################
#  path
##########################################################
path = r"I:\FloodMAPPING\Antwerp\\"
sewer_dir = path + "\\sewer\\"

# network files
shp1_filename = sewer_dir + "Node.shp"
shp2_filename = sewer_dir + "Conduit.shp"
shp3_filename = sewer_dir + "Structures.shp"
shp4_filename = sewer_dir + "Subcatchment.shp"
rgid_filename = sewer_dir + "Subcatchment_RGs.csv" # must contain 0,rg_id,node_id, subcat_id

##########################################################
# sewer network
##########################################################

# nodes
driver = ogr.GetDriverByName("ESRI Shapefile")
dataSource = driver.Open(shp1_filename, 0)
layer = dataSource.GetLayer()
featureCount = layer.GetFeatureCount()
x = []
y = []
kw = []
node_chambfloor = []
node_chambroof = []
node_chambarea = []
node_type = []
for feature in layer:
    kw.append(feature.GetField("node_id"))
    node_chambfloor.append(feature.GetField("chambfloor"))
    node_chambroof.append(feature.GetField("ground_lev"))
    node_chambarea.append(feature.GetField("chambarea"))
    node_type.append(feature.GetField("node_type").lower())
    geom = feature.GetGeometryRef()
    x.append(geom.GetX())
    y.append(geom.GetY())

nodes = {kk:(xx,yy) for kk,xx,yy in zip(kw,x,y)}
node_volumn = (np.array(node_chambroof) - np.array(node_chambfloor)) * np.array(node_chambarea)
node_volumn = {kk:(vv) for kk,vv in zip(kw,node_volumn)}
node_type = {kk:(tt) for kk,tt in zip(kw,node_type)}
node_geo = {kk:((xx,yy,zz)) for kk,xx,yy,zz in zip(kw,x,y, node_chambfloor)}
node_chambarea = {kk:(vv) for kk,vv in zip(kw,node_chambarea)}
node_chambfloor = {kk:(vv) for kk,vv in zip(kw,node_chambfloor)}
node_chambroof = {kk:(vv) for kk,vv in zip(kw,node_chambroof)}
# node rain gauges
df_rgid = pd.read_csv(rgid_filename, index_col=[0])
df_rgid = df_rgid.groupby('node_id')['rg_id'].apply(lambda x: x.mode().iloc[0])
node_rgid = {kk:df_rgid.loc[kk] for kk in kw if kk in df_rgid.index}

# links & structures
driver = ogr.GetDriverByName("ESRI Shapefile")
dataSource = driver.Open(shp2_filename, 0)
layer = dataSource.GetLayer()
featureCount = layer.GetFeatureCount()
lid = []
us = []
ds = []
sys_type = []
link_type = []
link_shape = []
capacity = [] #m3/s
gradient = [] #m/m
length = [] #m
width = [] #m
height = [] #m
sediment = [] #m
roughness = [] #mm
us_invert = []
ds_invert = []
for feature in layer:
    link_type.append(feature.GetField("link_type").lower())
    sys_type.append(feature.GetField("systemtype").lower())
    link_shape.append(feature.GetField("shape_1"))
    capacity.append(feature.GetField("capacity"))
    gradient.append(feature.GetField("gradient"))
    length.append(feature.GetField("condlen"))
    width.append(feature.GetField("condwidth")/1000.0)
    height.append(feature.GetField("condheight")/1000.0)
    sediment.append(feature.GetField("sed_depth")/1000.0)
    roughness.append(feature.GetField("bot_rcw"))
    us.append(feature.GetField("us_node_id"))
    ds.append(feature.GetField("ds_node_id"))
    lid.append(feature.GetField("id"))
    us_invert.append(feature.GetField("us_invert"))
    ds_invert.append(feature.GetField("ds_invert"))


str_invert = { 
    "compnd":"start_lev",
    "fixpmp":"on_level",
    "flap":"invert",
    "orific":"invert",
    "rotpmp":"on_level",
    "screen":"crest",
    "scrpmp":"on_level",
    "sluice":"invert",
    "weir":"crest",
    "vortex":"start_lev",
    "vsgate":"invert"
}
driver = ogr.GetDriverByName("ESRI Shapefile")
dataSource = driver.Open(shp3_filename, 0)
layer = dataSource.GetLayer()
featureCount = layer.GetFeatureCount()
for feature in layer:
    link_type_h = feature.GetField("link_type").lower()
    link_invert_h = feature.GetField(str_invert[link_type_h])
    if link_type_h == 'sluice':
        sluice_h = feature.GetField("opening")
        if sluice_h != 0:
            pass
        else:
            continue
    link_type.append(link_type_h)
    sys_type.append(feature.GetField("systemtype").lower())
    us.append(feature.GetField("us_node_id"))
    ds.append(feature.GetField("ds_node_id"))
    lid.append(feature.GetField("id"))
    us_invert.append(link_invert_h)
    ds_invert.append(link_invert_h)
    length.append(0)
    gradient.append(0)
    capacity.append(None)
    width.append(None)
    height.append(None)
    sediment.append(None)
    roughness.append(None)
    link_shape.append(None)

    
links = {k:[u,d,t,st,s, c,g,l,w,h,sed, r,ui,di] \
        for k,u,d,t,st,s,c,g,l,w,h,sed, r,ui,di in \
        zip(lid, us, ds, link_type, sys_type, link_shape, capacity, gradient, length, width, height, sediment, roughness, us_invert, ds_invert) }


# subcatchments
driver = ogr.GetDriverByName("ESRI Shapefile")
dataSource = driver.Open(shp4_filename, 0)
layer = dataSource.GetLayer()
featureCount = layer.GetFeatureCount()
sub_id = [] # subcatchment node id
subcatid = [] # subcatchment id
sub_slope = []
sub_cont_area = []
sub_runoff_area = []
for feature in layer:
    sub_id.append(feature.GetField("node_id"))
    subcatid.append(feature.GetField("subcatid"))
    sub_slope.append(feature.GetField("cat_slope")) #NOTE! NEW 04/09/2019
    sub_cont_area.append(feature.GetField("cont_area"))
    sub_runoff_area.append(feature.GetField("a_abs_1") + feature.GetField("a_abs_2")) #NOTE! NEW (04/09/2019)

subcat_id = defaultdict(list)
for kk,ii in zip(sub_id, subcatid): subcat_id[kk].append(ii)

subcat_slope = {kk:(xx) for kk,xx in zip(subcatid, sub_slope)}
node_subcatslope = {kk:(sum(subcat_slope[kkk] if subcat_slope[kkk] is not None else 0 for kkk in subcat_id[kk])) for kk in subcat_id }


subcat_area = {kk:(xx) for kk,xx in zip(subcatid, sub_runoff_area)}
node_area = {kk:(sum(subcat_area[kkk] if subcat_area[kkk] is not None else 0 for kkk in subcat_id[kk])) for kk in subcat_id }

subcat_grossarea = {kk:(xx) for kk,xx in zip(subcatid, sub_cont_area)}
node_grossarea = {kk:(sum(subcat_grossarea[kkk] if subcat_grossarea[kkk] is not None else 0 for kkk in subcat_id[kk])) for kk in subcat_id }

##########################################################
# Graph of sewer network
##########################################################

X=nx.MultiDiGraph() 
# add nodes with data as contributing area of the subcatchment
for k in nodes:
    t = (0 if k not in node_type or node_type[k] is None else node_type[k])
    v = (0 if k not in node_volumn or node_volumn[k] is None else node_volumn[k])
    ca = (0 if k not in node_chambarea or node_chambarea[k] is None else node_chambarea[k])
    cf = (0 if k not in node_chambfloor or node_chambfloor[k] is None else node_chambfloor[k])
    cr = (0 if k not in node_chambroof or node_chambroof[k] is None else node_chambroof[k])
    s = (0 if k not in node_subcatslope or node_subcatslope[k] is None else node_subcatslope[k])
    a = (0 if k not in node_area or node_area[k] is None else node_area[k]*10000)  # ha to square meters
    ga = (0 if k not in node_grossarea or node_grossarea[k] is None else node_grossarea[k]*10000)  # ha to square meters
    g = ( (0,0,0) if k not in node_geo or node_geo[k] is None else node_geo[k])
    i = (None if k not in subcat_id or subcat_id[k] is None else subcat_id[k])
    irg = (None if k not in node_rgid or node_rgid[k] is None else node_rgid[k])
    X.add_node(k, type = t, chambfloor_m = cf,chambroof_m = cr, chambarea_m2 = ca, chambvolume_m3 = v, subcatslope = s, cont_area_m2 = a, gross_area_m2 = ga, geo = g, subcat_id = i, rg_id = irg)

for k in links:
    X.add_edge(links[k][0], links[k][1], id = k, type = links[k][2],  sys_type = links[k][3], shape = links[k][4], \
      capacity_m3ds = links[k][5] , gradient_mdm = links[k][6], length_m = links[k][7], \
      width_m = links[k][8], height_m = links[k][9], sediment_m = links[k][10],\
      rcw_mm = links[k][11], us_invert_m = links[k][12], ds_invert_m = links[k][13])

# draw graph
nx.draw_networkx_nodes(X,nodes,node_size=2,node_color='r')
nx.draw_networkx_edges(X,nodes)
plt.title('Sewer network in Graph')
plt.axis('equal')
plt.savefig(sewer_dir + "sewer_graph_networkx_v4.0.png")
# plt.show()
# pickle
nx.write_gpickle(X,sewer_dir + "sewer_graph_networkx_v4.0.gpickle")
# read pickle
# X = nx.read_gpickle(sewer_dir + "sewer_graph_networkx.gpickle")
print('Graph created: ' + sewer_dir + "sewer_graph_networkx_v4.0.gpickle")