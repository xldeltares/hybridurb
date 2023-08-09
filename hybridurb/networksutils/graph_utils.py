# -*- coding: utf-8 -*-
# adapted from https://github.com/Deltares/ra2ce/blob/master/ra2ce/graph/network_wrappers/vector_network_wrapper.py
import logging
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString, MultiLineString
import momepy

logger = logging.getLogger(__name__)


__all__ = ["get_network_edges_and_nodes_from_graph"]


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

    nodes, edges = momepy.nx_to_gdf(graph, nodeID="node_fid")
    edges["edge_fid"] = (
        edges["node_start"].astype(str) + "_" + edges["node_end"].astype(str)
    )
    if not nodes.crs:
        nodes.crs = graph.graph["crs"]
    if not edges.crs:
        edges.crs = graph.graph["crs"]
    return edges, nodes
