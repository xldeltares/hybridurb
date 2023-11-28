# -*- coding: utf-8 -*-
import logging
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString
import momepy
import os
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
import random
from networkx.drawing.nx_agraph import graphviz_layout
import community
import itertools
import numpy as np
from pyproj import CRS
from hybridurb.gis_utils import *

from hydromt_delft3dfm.workflows import find_nearest_branch

logger = logging.getLogger(__name__)


__all__ = [
    # utils
    "validate_graph",
    "preprocess_graph",
    "graph_to_network",
    "network_to_graph",
    "read_graph",
    "write_graph",
    "get_network_edges_and_nodes_from_graph",
    # workflows
    "add_edges_with_id",
    "add_nodes_with_id",
    "update_edges_attributes",
    "update_nodes_attributes",
    "query_graph_edges_attributes",
    "query_graph_nodes_attributes",
]


def validate_graph(graph: nx.Graph) -> None:
    """
    Validate the graph.

    The method checks that the graph has the required attribtues.

    Parameters
    ----------
    nx.Graph
        The prepared graph.
        graph properties ['crs']
        edge properties ['edgeid', 'geometry','node_start', 'node_end']
        node properties ['nodeid', 'geometry']
    """
    _exist_attribute = "crs" in graph.graph
    assert _exist_attribute, "CRS does not exist in graph."
    _correct_attribute = isinstance(graph.graph["crs"], int)
    assert _correct_attribute, "crs must be epsg int."

    _exist_attribute = any(nx.get_node_attributes(graph, "nodeid"))
    assert _exist_attribute, "Missing nodeid in nodes"
    _exist_attribute = any(nx.get_node_attributes(graph, "geometry"))
    assert _exist_attribute, "Missing geometries in nodes"

    _exist_attribute = any(nx.get_edge_attributes(graph, "edgeid"))
    assert _exist_attribute, "Missing edgeid in edges"
    _exist_attribute = any(nx.get_edge_attributes(graph, "node_start"))
    assert _exist_attribute, "Missing node_start in edges"
    _exist_attribute = any(nx.get_edge_attributes(graph, "node_end"))
    assert _exist_attribute, "Missing node_end in edges"
    _exist_attribute = any(nx.get_edge_attributes(graph, "geometry"))
    assert _exist_attribute, "Missing geometries in edges"


def preprocess_graph(graph: nx.Graph, to_crs: CRS = None) -> nx.Graph:
    """
    Preprocess graph geometry and indexes.

    Parameters
    ----------
    graph: nx.Graph
        The input graph. Must provide crs in global properties
    to_crs: CRS
        The desired CRS for the graph's geometry (if reprojecting is desired).

    Returns
    -------
    nx.Graph
        The prepared graph.
        graph properties ['crs']
        edge properties ['edgeid', 'geometry','node_start', and 'node_end']
        node properties ['nodeid', 'geometry']
    """
    crs = graph.graph.get("crs")
    if not crs:
        raise ValueError("must provide crs in graph.")
    crs = CRS.from_user_input(crs)

    # assign graph geometry
    graph = assign_graph_geometry(graph)

    # If CRS is provided, reproject the graph
    if to_crs and to_crs != crs:
        graph = reproject_graph_geometry(graph, crs, to_crs)
        graph.graph["crs"] = graph.graph["crs"].to_epsg()

    # Assign node and edge indexes
    graph = assign_graph_index(graph)

    validate_graph(graph)

    return graph


def assign_graph_geometry(graph: nx.Graph) -> nx.Graph:
    """Add geometry to graph nodes and edges.

    Note that this function does not add missing geometries one by one.
    """
    _exist_node_geometry = any(nx.get_node_attributes(graph, "geometry"))
    _exist_edge_geometry = any(nx.get_edge_attributes(graph, "geometry"))

    if _exist_node_geometry and _exist_edge_geometry:
        return graph

    if (not _exist_node_geometry) and _exist_edge_geometry:
        if any(nx.get_node_attributes(graph, "x")):
            # use xy from nodes
            geometry = {n: Point(a["x"], a["y"]) for n, a in graph.nodes(data=True)}
            nx.set_node_attributes(graph, geometry, name="geometry")
        else:
            # add node from egde geometry
            assert not graph.is_multigraph()  # FIXME support multigraph
            geometry = {}
            for s, t, e in graph.edges(data="geometry"):
                geometry.update({s: Point(e.coords[0]), t: Point(e.coords[-1])})
            nx.set_node_attributes(graph, geometry, name="geometry")
        return graph

    if _exist_node_geometry and (not _exist_edge_geometry):
        # add edge from node geometry
        assert not graph.is_multigraph()  # FIXME support multigraph
        geometry = {}
        for s, t, e in graph.edges(data="geometry"):
            geometry.update(
                {
                    (s, t): LineString(
                        [graph.nodes[s]["geometry"], graph.nodes[t]["geometry"]]
                    )
                }
            )
        nx.set_edge_attributes(graph, geometry, name="geometry")
        return graph

    if not (_exist_node_geometry or _exist_edge_geometry):
        # no geometry, check if the geometry can be derived from node tuple
        assert not graph.is_multigraph()  # FIXME support multigraph
        assert len(list(graph.nodes)[0]) == 2  # must have tuple (x,y) as nodes
        geometry = {n: Point(n[0], n[1]) for n in graph.nodes}
        nx.set_node_attributes(graph, geometry, name="geometry")
        return graph


def reproject_graph_geometry(graph: nx.Graph, crs_from: CRS, crs_to: CRS) -> nx.Graph:
    """
    Reprojects the geometry attributes of the nodes and edges of the graph.

    Parameters
    ----------
    graph: nx.Graph
        The input graph with geometry attributes.
    crs_from: CRS
        The current CRS of the graph's geometry.
    crs_to: CRS
        The desired CRS for the graph's geometry.

    Returns
    -------
    nx.Graph
        A new graph with reprojected geometry attributes.
    """
    # Create a new graph instance to avoid modifying the original graph
    reprojected_graph = graph.copy()

    # Initialize the transformer
    transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)

    # Reproject the nodes
    for node, attributes in graph.nodes(data=True):
        if "geometry" in attributes:
            reprojected_geometry = transform(
                transformer.transform, attributes["geometry"]
            )
            reprojected_graph.nodes[node]["geometry"] = reprojected_geometry

    # Reproject the edges
    if graph.is_multigraph():
        for u, v, key, attributes in graph.edges(keys=True, data=True):
            reprojected_geometry = transform(
                transformer.transform, attributes["geometry"]
            )
            reprojected_graph[u][v][key]["geometry"] = reprojected_geometry
    else:
        for u, v, attributes in graph.edges(data=True):
            if "geometry" in attributes:
                reprojected_geometry = transform(
                    transformer.transform, attributes["geometry"]
                )
                reprojected_graph[u][v]["geometry"] = reprojected_geometry

    reprojected_graph.graph["crs"] = crs_to

    return reprojected_graph


def assign_graph_index(graph: nx.Graph) -> nx.Graph:
    """
    Assign unique indexes to the nodes and edges of the graph.

    Parameters
    ----------
    graph: nx.Graph
        The input graph.

    Returns
    -------
    nx.Graph
        The prepared graph.
        edge properties ['edgeid', 'geometry','node_start', and 'node_end']
        node properties ['nodeid', 'geometry']
    """
    # Create a new graph instance to avoid modifying the original graph
    indexed_graph = graph.copy()

    # Assign nodeids
    for index, node in enumerate(graph.nodes()):
        indexed_graph.nodes[node]["nodeid"] = index

    # Assign edgeids
    if graph.is_multigraph():
        for u, v, key in graph.edges(keys=True):
            node_start = str(indexed_graph.nodes[u]["nodeid"])
            node_end = str(indexed_graph.nodes[v]["nodeid"])
            edge_id = node_start + "_" + node_end
            indexed_graph[u][v][key]["edgeid"] = edge_id
            indexed_graph[u][v][key]["node_start"] = node_start
            indexed_graph[u][v][key]["node_end"] = node_end
            edge_id_multi = node_start + "_" + node_end + "_" + str(key)
            indexed_graph[u][v][key]["edgeid_multi"] = edge_id_multi
    else:
        for u, v in graph.edges():
            node_start = str(indexed_graph.nodes[u]["nodeid"])
            node_end = str(indexed_graph.nodes[v]["nodeid"])
            edge_id = node_start + "_" + node_end
            indexed_graph[u][v]["edgeid"] = edge_id
            indexed_graph[u][v]["node_start"] = node_start
            indexed_graph[u][v]["node_end"] = node_end
    return indexed_graph


def network_to_graph(
    edges: gpd.GeoDataFrame,
    nodes: Union[gpd.GeoDataFrame, None] = None,
    create_using=nx.DiGraph,
) -> nx.Graph:
    """
    Convert the network edges and/or nodes to a graph.

    Parameters
    ----------
    edges: gpd.GeoDataFrame
        The network lines.
    nodes: gpd.GeoDataFrame or None
        The network nodes, optional.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    nx.Graph
        The prepared graph.
        global properties ['crs']
        edge properties ['edgeid', 'geometry','node_start', and 'node_end']
        node properties ['nodeid', 'geometry']
    """
    # use memopy convention but not momepy.gdf_to_nx
    # create graph from edges
    graph = nx.from_pandas_edgelist(
        edges,
        source="node_start",
        target="node_end",
        create_using=create_using,
        edge_attr=True,
    )

    # add node attribtues
    if nodes is not None:
        nx.set_node_attributes(graph, {nid: {"nodeid": nid} for nid in nodes["nodeid"]})
        nx.set_node_attributes(graph, nodes.set_index("nodeid").to_dict("index"))

    graph.graph["crs"] = edges.crs
    return graph


def graph_to_network(
    graph: nx.Graph, crs: CRS = None
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Convert the graph to network edges and nodes.

    Parameters
    ----------
    graph : nx.Graph
        The input graph.
        Required variables: ["geometry"] for graph.edges
        Optional variables: ["geometry"] for graph.nodes

        If "geometry" is not present, graph nodes should have "x" and "y" as attributes.
        Optional global property crs (pyproj.CRS).
    crs: CRS
        The crs of the output network.
        Optional if graph.graph["crs"] exsit.

    Returns
    -------
    Tuple
        gpd.GeoDataFrame
            edges of the graph.
            Contains attributes ['edgeid', 'geometry', node_start', 'node_end']
        gpd.GeoDataFrame
            nodes of the graph.
            Contains attributes ['nodeid', 'geometry']

    See Also
    --------
    momepy.nx_to_gdf
    """
    # get crs
    crs = crs if crs is not None else graph.graph.get("crs")
    if not crs:
        raise ValueError("must provide crs.")

    # assign geometry
    graph = assign_graph_geometry(graph)

    # assign ids
    nodes, edges = momepy.nx_to_gdf(graph, nodeID="nodeid")
    edges["edgeid"] = (
        edges["node_start"].astype(str) + "_" + edges["node_end"].astype(str)
    )

    # assign crs
    crs = CRS.from_user_input(crs)
    edges = edges.to_crs(crs)
    nodes = nodes.to_crs(crs)

    return edges, nodes


def read_graph(graph_fn: str) -> nx.Graph:
    """
    Read the graph from a file.

    Parameters
    ----------
    graph_fn: str or Path
        The path to the file where the graph is stored.
        Supported extentions: gml, graphml, gpickle, geojson

    Returns
    -------
    nx.Graph
    """
    filepath = Path(graph_fn)
    extension = filepath.suffix.lower()

    if extension == ".gml":
        return nx.read_gml(filepath)
    elif extension == ".graphml":
        return nx.read_graphml(filepath)
    elif extension == ".gpickle":
        with open(filepath, "rb") as f:
            return pickle.load(f)
    elif extension == ".geojson":
        edges = gpd.read_file(filepath, driver="GeoJSON")
        return network_to_graph(edges)
    else:
        raise ValueError(f"Unsupported file format: {extension}")


def _pop_geometry_from_graph(graph: nx.Graph) -> nx.Graph:
    graph_copy = graph.copy()

    x = {}
    y = {}

    for node, attrs in graph_copy.nodes(data=True):
        geometry = attrs.pop("geometry", None)
        if geometry:
            x[node] = geometry.x
            y[node] = geometry.y

    for _, _, attrs in graph_copy.edges(data=True):
        attrs.pop("geometry", None)

    nx.set_node_attributes(graph_copy, x, name="x")
    nx.set_node_attributes(graph_copy, y, name="y")

    return graph_copy


def _pop_list_from_graph(graph: nx.Graph) -> nx.Graph:
    graph_copy = graph.copy()
    for u, v, attrs in graph_copy.edges(data=True):
        for key, value in list(
            attrs.items()
        ):  # Convert to list to avoid "dictionary size changed during iteration" error
            if isinstance(value, list):
                attrs.pop(key)
    for node, attrs in graph_copy.nodes(data=True):
        for key, value in list(
            attrs.items()
        ):  # Convert to list to avoid "dictionary size changed during iteration" error
            if isinstance(value, list):
                attrs.pop(key)
    return graph_copy


def write_graph(graph: nx.Graph, graph_fn: str) -> None:
    """
    Write the graph to a file.

    Parameters
    ----------
    graph: nx.Graph
        The graph to be written.
    graph_fn: str or Path
        The path to the file where the graph will be written.
        Supported extentions: gml, graphml, gpickle, geojson
        To write complete information, use gpickle.
        To write graph without geometry info, use gml.
        To write graph without geometry and any lists, use graphml.
        To write graph with geometry and index only, use geojson.

    Returns
    -------
    None
    """
    filepath = Path(graph_fn)
    extension = filepath.suffix.lower()

    if extension == ".gml":
        nx.write_gml(_pop_geometry_from_graph(graph), filepath)
    elif extension == ".graphml":
        nx.write_graphml(
            _pop_list_from_graph(_pop_geometry_from_graph(graph)), filepath
        )
    elif extension == ".gpickle":
        with open(filepath, "wb") as f:
            pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
    elif extension == ".geojson":
        edges, _ = graph_to_network(graph)
        edges = edges[["edgeid", "node_start", "node_end", "geometry"]]
        edges.to_file(filepath, driver="GeoJSON")
    else:
        raise ValueError(f"Unsupported file format: {extension}")


def get_network_edges_and_nodes_from_graph(
    graph: nx.Graph,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Sets up network nodes and edges from a given graph.

    Args:
        graph (nx.Graph): Input graph with geometry for nodes and edges.
            Must contain "crs" as graph property.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame representing the network edges with "edge_fid", "node_start", and "node_end".
        gpd.GeoDataFrame: GeoDataFrame representing the network nodes with "node_fid".
    """
    DeprecationWarning("will be replaced by graph_to_network")

    nodes, edges = momepy.nx_to_gdf(graph, nodeID="node_fid")
    edges["edge_fid"] = (
        edges["node_start"].astype(str) + "_" + edges["node_end"].astype(str)
    )
    if not nodes.crs:
        nodes.crs = graph.graph["crs"]
    if not edges.crs:
        edges.crs = graph.graph["crs"]
    return edges, nodes


# create graph


def add_edges_with_id(
    G: nx.Graph, edges: gpd.GeoDataFrame, id_col: str, snap_offset: float = 1e-6
) -> nx.Graph():
    """Return graph with edges and edges ids"""

    for index, row in edges.iterrows():
        from_node = row.geometry.coords[0]
        to_node = row.geometry.coords[-1]

        G.add_edge(from_node, to_node, id=row[id_col])

    return G


def add_nodes_with_id(
    G: nx.Graph, nodes: gpd.GeoDataFrame, id_col: str, snap_offset: float = 1e-6
) -> nx.Graph():
    """return graph with nodes and nodes ids"""

    # derived nodes and user nodes
    G_nodes = gpd.GeoDataFrame(
        {
            "geometry": [Point(p) for p in G.nodes],
            "id": [f"{p[0]:.6f}_{p[1]:.6f}" for p in G.nodes],
            "tuple": G.nodes,
        }
    ).set_index("id")
    if "id" in nodes.columns:
        logger.error("Abort: nodes contains id columns. Please remove the column.")
    F_nodes = nodes.rename(columns={id_col: "id"}).set_index("id")

    # map user nodes to derived nodes
    if set(F_nodes.index).issubset(set(G_nodes.index)):
        # check if 1-to-1 match
        G_nodes_new = G_nodes.join(F_nodes.drop(columns="geometry"))

    else:
        G_nodes_new = snap_nodes_to_nodes(F_nodes, G_nodes, snap_offset)
        logger.debug("performing snap nodes to graph nodes")

    # assign nodes id to graph
    dict = {row.tuple: i for i, row in G_nodes_new.iterrows()}
    nx.set_node_attributes(G, dict, "id")

    return G


def update_edges_attributes(
    G: nx.Graph,
    edges: gpd.GeoDataFrame,
    id_col: str,
) -> nx.Graph():
    """This function updates the graph by adding new edges attributes specified in edges"""

    # graph df
    _graph_df = nx.to_pandas_edgelist(G).set_index("id")

    # check if edges id in attribute df
    if edges.index.name == id_col:
        edges.index.name = "id"
    elif id_col in edges.columns:
        edges = edges.set_index(id_col)
        edges.index.name = "id"
    else:
        raise ValueError(
            "attributes could not be updated to graph: could not perform join"
        )

    # last item that isnt NA
    _graph_df = _graph_df.reindex(
        columns=_graph_df.columns.union(edges.columns, sort=False)
    )
    graph_df = pd.concat([_graph_df, edges]).groupby(level=0).last()
    graph_df = graph_df.loc[_graph_df.index]

    G_updated = nx.from_pandas_edgelist(
        graph_df.reset_index(),
        source="source",
        target="target",
        edge_attr=True,
        create_using=type(G),
    )

    return G_updated


def find_edge_ids_by_snapping(
    G: nx.Graph,
    edges: gpd.GeoDataFrame,
    snap_offset: float = 1,
    snap_method: str = "overall",
) -> gpd.GeoDataFrame:
    """This function adds "id" to edges GeoDataFrame"""

    # graph
    _ = gpd.GeoDataFrame(nx.to_pandas_edgelist(G).set_index("id"))

    # wrapper to use delft3dfmpy function to find "branch_id"
    _ = _.rename({"id": "branch_id"}).assign(branchType=None)
    find_nearest_branch(
        _,
        edges,
        method=snap_method,
        maxdist=snap_offset,
        move_geometries=True,
    )

    # rename "branch_id" to "edge_id"
    edges_with_ids = edges.rename(columns={"branch_id": "_id"})

    return edges_with_ids


def find_node_ids_by_snapping(
    G: nx.Graph,
    nodes: gpd.GeoDataFrame,
    snap_offset: float = 1,
    snap_method: str = "overall",
) -> gpd.GeoDataFrame:
    """This function adds "id" to nodes GeoDataFrame"""

    # graph
    G_nodes = gpd.GeoDataFrame(
        {
            "geometry": [Point(p) for p in G.nodes],
            "node_id": [f"{p[0]:.6f}_{p[1]:.6f}" for p in G.nodes],
            "_id": G.nodes(data="id"),
        }
    ).set_index("node_id")

    # nodes
    nodes.loc[:, "node_id"] = [
        f"{x:.6f}_{y:.6f}" for x, y in zip(nodes.geometry.x, nodes.geometry.y)
    ]
    nodes = nodes.set_index("node_id")

    # map user nodes to derived nodes
    if set(nodes.index).issubset(set(G_nodes.index)):
        # check if 1-to-1 match
        G_nodes_new = G_nodes.join(nodes.drop(columns="geometry"))
    else:
        # use snap_nodes_to_nodes function to find "node_id"
        G_nodes_new = snap_nodes_to_nodes(nodes, G_nodes, snap_offset)
        logger.debug("performing snap nodes to graph nodes")

    # assign id from graph to nodes
    nodes = nodes.join(G_nodes_new["_id"])

    return nodes


def update_nodes_attributes(
    G: nx.Graph,
    nodes: gpd.GeoDataFrame,
    id_col: str,
) -> nx.Graph():
    """This function updates the graph by adding new edges attributes specified in edges"""

    # graph df
    _graph_df = pd.DataFrame(G.nodes(data="id"), columns=["tuple", "id"]).set_index(
        "id"
    )

    # check if edges id in attribute df
    if nodes.index.name == id_col:
        nodes.index.name = "id"
    elif id_col in nodes.columns:
        nodes = nodes.set_index(id_col)
        nodes.index.name = "id"
    else:
        raise ValueError(
            "attributes could not be updated to graph: could not perform join"
        )

    # last item that isnt NA
    _graph_df = _graph_df.reindex(
        columns=_graph_df.columns.union(nodes.columns, sort=False)
    )
    graph_df = pd.concat([_graph_df, nodes]).groupby(level=0).last()
    graph_df = graph_df.loc[_graph_df.index]

    # add each attribute
    for c in nodes.columns:
        dict = {row.tuple: row[c] for i, row in graph_df.iterrows()}
        nx.set_node_attributes(G, dict, c)

    return G


# process graph


def query_graph_edges_attributes(G, id_col: str = "id", edge_query: str = None):
    """This function queries the graph by selecting only the edges specified in edge_query"""

    if edge_query is None:
        G_query = G.copy()
    else:
        _graph_df = nx.to_pandas_edgelist(G).set_index(id_col)
        keep_df = _graph_df.query(edge_query)

        if len(keep_df) > 0:
            G_query = G.edge_subgraph(
                [(row.source, row.target) for row in keep_df.itertuples()]
            ).copy()

        else:
            raise ValueError("edges_query results in nothing left")

    return G_query


def query_graph_nodes_attributes(G, id_col: str = "id", node_query: str = None):
    """This function queries the graph by selecting only the nodes specified in node_query"""

    if node_query is None:
        G_query = G

    else:
        _graph_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index")
        graph_df = _graph_df.query(node_query)

        if len(graph_df) != 0:
            G_query = G.subgraph(graph_df.index.tolist()).copy()
        else:
            raise ValueError("node_query results in nothing left")

    return G_query


def contract_graph_nodes(G, nodes, to_node=None):
    """This function contract the nodes into one node in G"""

    G_contracted = G.copy()
    node_contracted = []

    if len(nodes) > 1:
        nodes = sorted(nodes)
        if not to_node:  # use the first node
            to_node = nodes[0]
        nodes = [n for n in nodes if n != to_node]
        node_contracted.append(to_node)
        for node in nodes:
            G_contracted = nx.contracted_nodes(
                G_contracted, to_node, node, self_loops=False, copy=False
            )

    return G_contracted, node_contracted


def louvain_partition(G: nx.Graph):
    """This function is a wrapper around best partiton method in community
    See :py:meth:`~community.best_partition()` for more information.
    """
    return community.best_partition(G)


def sort_ids(G: nx.Graph):
    """Function to sort the ids of the graph.
    if there are no ids for the nodes, the ids will be generated based on the geometry
    """
    if set(dict(G.nodes(data="id")).values()) == {None}:
        nx.set_node_attributes(G, {p: f"{p[0]:.6f}_{p[1]:.6f}" for p in G.nodes}, "id")

    return G


def sort_ends(G: nx.Graph):
    """Function to sort the ends of the graph.

    Arguments
    ---------
    G: nx.Graph
        Networkx Graph

    Returns
    -------
    G: nx.Graph
        Networkx Graph with node attribute '_type'
    """
    if isinstance(G, nx.DiGraph):
        endnodes = {
            n: "endnode" for n in G.nodes if (G.degree[n] == 1 and G.out_degree[n] == 0)
        }
        startnodes = {
            n: "startnode"
            for n in G.nodes
            if (G.degree[n] == 1 and G.in_degree[n] == 0)
        }
        nx.set_node_attributes(G, endnodes, "_type")
        nx.set_node_attributes(G, startnodes, "_type")
    elif isinstance(G, nx.Graph):
        endnodes = {n: "endnode" for n in G.nodes if G.degree[n] == 1}
        nx.set_node_attributes(G, endnodes, "_type")
    return G


def sort_direction(G: nx.DiGraph) -> nx.DiGraph:
    """Function sort the start end direction of the graph and obtain start and end nodes.

    Arguments
    ---------
    G: nx.DiGraph
        Directional Graph

    Returns
    -------
    G: nx.DiGraph
        Directional Graph with node attributes endnodes and startnodes"""

    endnodes = {
        n: "endnode" for n in G.nodes if (G.degree[n] == 1 and G.out_degree[n] == 0)
    }
    startnodes = {
        n: "startnode" for n in G.nodes if (G.degree[n] == 1 and G.in_degree[n] == 0)
    }
    nx.set_node_attributes(G, endnodes, "_type")
    nx.set_node_attributes(G, startnodes, "_type")
    return G


def get_predecessors(G: nx.DiGraph, n, inclusive=True):
    """Function to find the predecessors of a node n
    See :py:meth:`~nx.bfs_predecessors()` for more information.

    Arguments
    ---------
    G: nx.DiGraph
        Directional Graph
    n:
        Directional Graph node that are used as target to find predecessors
    inclusive: bool
        Whether to include the input node in the results
    """

    RG = G.reverse()
    predecessors = list(dict(nx.bfs_predecessors(RG, n)).keys())
    if inclusive:
        predecessors = predecessors + [n]
    return predecessors


def find_difference(G, H):
    """function to find the difference between G and H (G-H) based on edges
    replace :py:meth:`~nx.difference()`"""
    c = G.copy()
    c.remove_edges_from(H.edges)
    c.remove_nodes_from(list(nx.isolates(c)))
    return c


def contract_graph(G: nx.Graph, partition, tonodes):
    """contract based on partition --> needs further improvements
    TODO: harmonize with setup partition
    """
    ind = G.copy()
    nx.set_node_attributes(ind, {n: {"ind_size": 1} for n in ind.nodes})
    for part in np.unique(list(partition.values())):
        part_nodes = [n for n in partition if partition[n] == part]
        if part == -1:
            # do not contract
            pass
        else:
            for to_node in [n for n in part_nodes if n in tonodes]:
                ind, targets = contract_graph_nodes(ind, part_nodes, to_node)
                ind.nodes[to_node]["ind_size"] = len(part_nodes)
    return ind


def make_dag(
    G: nx.DiGraph,
    targets: list,
    weight: str = None,
    algorithm="dijkstra",
    drop_unreachable=False,
    logger=logger,
):
    """dag making for digraph --> needs further improvements
    TODO: add to setup_dag
    # test
    # G = nx.DiGraph()
    # G.add_edge(0,1)
    # G.add_edge(1,2)
    # G.add_edge(1,3)
    # nx.draw(G)
    # targets = [3, 2]
    # G.add_edges_from ([(o, -1) for o in targets])
    # nx.draw(G)
    # X_new = nx.DiGraph()
    # for n in G.nodes:
    #     path = nx.shortest_path(G, n, -1,
    #              method = algorithm)
    #     nx.add_path(X_new, path)
    # X_new.remove_nodes_from([-1])
    """

    # copy
    X = G.copy()

    X_new = nx.DiGraph()
    X.add_edges_from([(o, -1) for o in targets])
    # for nodes that are reachable from target
    if drop_unreachable:
        XX = X.reverse()
        sub_network = X.subgraph(nx.dfs_tree(XX, -1).nodes)
        X = sub_network.copy()
        logger.debug("drop unreachable nodes")

    for n in X.nodes:
        path = nx.shortest_path(X, n, -1, weight=weight, method=algorithm)
        nx.add_path(X_new, path)

    X_new.remove_nodes_from([-1])

    # add back weights
    for u, v, weight in X_new.edges(data=True):
        # get the old data from X
        xdata = X[u].get(v)
        non_shared = set(xdata) - set(weight)
        if non_shared:
            # add old weights to new weights if not in new data
            weight.update(dict((key, xdata[key]) for key in non_shared))

    return X_new


# plot graph


def make_graphplot_for_targetnodes(
    G: nx.DiGraph,
    target_nodes: list,
    target_nodes_labeldict: dict = None,
    layout="xy",
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # layout graphviz
    if layout == "graphviz":
        # get position
        pos = graphviz_layout(G, prog="dot", args="")

        # draw network
        nx.draw(G, pos, node_size=20, alpha=0.5, node_color="blue", with_labels=False)

        if target_nodes_labeldict is not None:
            # draw labels
            nx.draw_networkx_labels(
                G, pos, target_nodes_labeldict, font_size=16, font_color="k"
            )

    # layout xy
    elif layout == "xy":
        # get position
        pos = {xy: xy for xy in G.nodes()}
        # make plot for each target node
        RG = G.reverse()

        for target in target_nodes:
            c = random_color()

            # make target upstream a graph
            target_G = G.subgraph(
                list(dict(nx.bfs_predecessors(RG, target)).keys()) + [target]
            )

            # draw graph
            nx.draw_networkx(
                target_G,
                pos,
                node_size=10,
                node_color=[c],
                width=2,
                edge_color=[c],
                with_labels=False,
                ax=ax,
            )

            # draw outlets
            nx.draw_networkx_nodes(
                target_G,
                pos,
                nodelist=[target],
                node_size=100,
                node_color="k",
                edgecolors=c,
                ax=ax,
            )

            # draw labels
            if target_nodes_labeldict is not None:
                nx.draw_networkx_labels(
                    target_G, pos, target_nodes_labeldict, font_size=16, font_color="k"
                )
    return fig, ax


def plot_xy(G: nx.DiGraph, plot_outfall=False):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    pos_G = {xy: xy for xy in G.nodes()}
    nx.draw_networkx_nodes(G, pos=pos_G, node_size=1, node_color="k")
    nx.draw_networkx_edges(G, pos=pos_G, edge_color="k", arrows=False)

    if plot_outfall:
        if isinstance(G, nx.DiGraph):
            endnodes = [n for n in G.nodes if G.out_degree[n] == 0 and G.degree[n] == 1]
            startnodes = [
                n for n in G.nodes if G.in_degree[n] == 0 and G.degree[n] == 1
            ]
            nx.draw_networkx_nodes(
                endnodes, pos=pos_G, node_size=10, node_color="r", node_shape="o"
            )
            nx.draw_networkx_nodes(
                startnodes, pos=pos_G, node_size=10, node_color="b", node_shape="v"
            )
        else:
            pass  # can only be used in case of digraph
    return


def plot_graphviz(G: nx.DiGraph):
    """This function makes plots for grahviz layout"""

    # convert to undirected graph
    UG = G.to_undirected()

    # find connected components in undirected graph
    outlets = []
    for i, SG in enumerate(nx.connected_components(UG)):
        # make components a subgraph
        SG = G.subgraph(SG)

        # find outlets of the subgraph
        outlets.append([n for n in SG.nodes() if G.out_degree(n) == 0])

    outlets = sum(outlets, [])
    outlet_ids = {}

    fig1, ax1 = make_graphplot_for_targetnodes(
        G, outlets, outlet_ids, layout="graphviz"
    )
    return (fig1, ax1)


def plot_graph(G: nx.DiGraph):
    """This function makes plots for two different layout"""

    # convert to undirected graph
    UG = G.to_undirected()

    # find connected components in undirected graph
    outlets = []
    for i, SG in enumerate(nx.connected_components(UG)):
        # make components a subgraph
        SG = G.subgraph(SG)

        # find outlets of the subgraph
        outlets.append([n for n in SG.nodes() if G.out_degree(n) == 0])

    outlets = sum(outlets, [])
    outlet_ids = {}

    fig1, ax1 = make_graphplot_for_targetnodes(
        G, outlets, outlet_ids, layout="graphviz"
    )
    fig2, ax2 = make_graphplot_for_targetnodes(G, outlets, outlet_ids, layout="xy")

    return (fig1, ax1), (fig2, ax2)


def random_color():
    return tuple(
        [
            random.randint(0, 255) / 255,
            random.randint(0, 255) / 255,
            random.randint(0, 255) / 255,
        ]
    )


# validate graph - old


def validate_1dnetwork_connectivity(
    branches: gpd.GeoDataFrame,
    plotit=False,
    ax=None,
    exportpath=os.getcwd(),
    logger=logging,
):
    """Function to validate the connectivity of provided branch"""

    # affirm datatype
    branches = gpd.GeoDataFrame(branches)

    # create digraph
    G = create_graph_from_branches(branches)
    pos = {xy: xy for xy in G.nodes()}

    # convert to undirected graph
    UG = G.to_undirected()

    # find connected components in undirected graph
    outlets = []
    for i, SG in enumerate(nx.connected_components(UG)):
        # make components a subgraph
        SG = G.subgraph(SG)

        # find outlets of the subgraph
        outlets.append([n for n in SG.nodes() if G.out_degree(n) == 0])

    outlets = sum(outlets, [])
    outlet_ids = {
        p: [li for li, l in branches.geometry.iteritems() if l.intersects(Point(p))]
        for p in outlets
    }

    # report
    if i == 0:
        logger.info(
            "Validation results: the 1D network are fully connected.  Supress plotit function."
        )
    else:
        logger.info(
            f"Validation results: the 1D network are disconnected have {i+1} connected components"
        )

    if plotit:
        ax = make_graphplot_for_targetnodes(G, outlets, outlet_ids, layout="graphviz")
        ax.set_title(
            "Connectivity of the 1d network, with outlets"
            + "(connectivity outlets, not neccessarily network outlets due to bi-directional flow, please check these)",
            wrap=True,
        )
        plt.savefig(exportpath.joinpath("validate_1dnetwork_connectivity"))

    return None


def validate_1dnetwork_flowpath(
    branches: gpd.GeoDataFrame,
    branchType_col="branchType",
    plotit=False,
    ax=None,
    exportpath=os.getcwd(),
    logger=logging,
):
    """function to validate flowpath (flowpath to outlet) for provided branch"""

    # affirm datatype
    branches = gpd.GeoDataFrame(branches)

    # create digraph
    G = gpd_to_digraph(branches)
    pos = {xy: xy for xy in G.nodes()}

    # create separate graphs for pipes and branches
    pipes = branches.query(f"{branchType_col} == 'Pipe'")
    channels = branches.query(f"{branchType_col} == 'Channel'")

    # validate 1d network based on pipes -> channel logic
    if len(pipes) > 0:
        # create graph
        PG = gpd_to_digraph(pipes)
        # pipes outlets
        pipes_outlets = [n for n in PG.nodes() if G.out_degree(n) == 0]
        pipes_outlet_ids = {
            p: [li for li, l in pipes.geometry.iteritems() if l.intersects(Point(p))]
            for p in pipes_outlets
        }
        logger.info(
            f"Validation result: the 1d network has {len(pipes_outlets)} pipe outlets."
        )

    if len(channels) > 0:
        # create graph
        CG = gpd_to_digraph(channels)
        # pipes outlets
        channels_outlets = [n for n in CG.nodes() if G.out_degree(n) == 0]
        channels_outlet_ids = {
            p: [li for li, l in channels.geometry.iteritems() if l.intersects(Point(p))]
            for p in channels_outlets
        }
        logger.info(
            f"Validation result: the 1d network has {len(channels_outlets)} channel outlets."
        )

    if (len(channels) > 0) and (len(pipes) > 0):
        isolated_outlets = [
            p
            for p in pipes_outlets
            if not any(Point(p).intersects(l) for _, l in channels.geometry.iteritems())
        ]
        isolated_outlet_ids = {}
        for p in isolated_outlets:
            isolated_outlet_id = [
                li for li, l in pipes.geometry.iteritems() if l.intersects(Point(p))
            ]
            isolated_outlet_ids[p] = isolated_outlet_id
            logger.warning(
                f"Validation result: downstream of {isolated_outlet_id} are not located on channels. Please double check. "
            )

    # plot
    if plotit:
        ax = make_graphplot_for_targetnodes(
            G,
            target_nodes={**isolated_outlet_ids, **channels_outlet_ids}.keys(),
            target_nodes_labeldict={**isolated_outlet_ids, **channels_outlet_ids},
        )
        ctx.add_basemap(
            ax=ax, url=ctx.providers.OpenStreetMap.Mapnik, crs=branches.crs.to_epsg()
        )
        ax.set_title(
            "Flow path of the 1d network, with outlets"
            + "(flowpath outlets, not neccessarily network outlets due to bi-directional flow , please check these)",
            wrap=True,
        )
        plt.savefig(exportpath.joinpath("validate_1dnetwork_flowpath"))

    return None


def get_arborescence(G: nx.DiGraph):
    """function to get arborescence from Digraph
    This function will loop through all bifurcation node and check if its predecessors forms a arborescence.
    If yes, _arborescence = True is assigned to nodes and edges.
    See :py:meth:`networkx.algorithms.tree.recognition.is_arborescence` for more.
    """

    if not isinstance(G, nx.DiGraph):
        raise TypeError("Must be a DiGraph")

    # prepare
    G_neutral = G.to_undirected()
    G_positive = G.copy()
    G_negative = G_positive.reverse()

    # get all bifurcation node
    bifurcations = [n for n in G.nodes if G.degree[n] > 2]

    # get edges that can be pruned and its load
    for n in bifurcations:
        # get its upstream as a subgraph (based on edges not nodes)
        n_predecessors = nx.dfs_predecessors(G_negative, n)
        if len(n_predecessors) > 1:
            subgraph_edges = []
            for nn in n_predecessors:
                _ = list(itertools.chain(*list(G_neutral.edges(nn))))
                subgraph_edges.append(_)
            subgraph_nodes = list(set(sum(subgraph_edges, [])))

            subgraph = G_negative.subgraph(subgraph_nodes)
            subgraph_edges = list(subgraph.edges())

            # check if its upstream subgraph is arborescence
            if nx.is_arborescence(subgraph):
                nx.set_node_attributes(
                    G, {k: True for k in subgraph_nodes}, "_arborescence"
                )
                nx.set_edge_attributes(
                    G, {e: True for e in subgraph_edges}, "_arborescence"
                )

    return G
