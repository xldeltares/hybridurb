# adapted from https://github.com/Deltares/ra2ce/blob/master/ra2ce/graph/network_wrappers/vector_network_wrapper.py

import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString, MultiLineString
import momepy


class GraphUtils:
    """A class for handling and manipulating vector files.

    Provides methods for reading vector data, cleaning it, and setting up graph and
    network.
    """

    # def get_network(
    #     self,
    # ) -> tuple[nx.MultiGraph, gpd.GeoDataFrame]:
    #     """Gets a network built from vector files.

    #     Returns:
    #         nx.MultiGraph: MultiGraph representing the graph.
    #         gpd.GeoDataFrame: GeoDataFrame representing the network.
    #     """
    #     gdf = self._read_vector_to_project_region_and_crs()
    #     gdf = self.clean_vector(gdf)
    #     if self.directed:
    #         graph = self.get_direct_graph_from_vector(gdf)
    #     else:
    #         graph = self.get_indirect_graph_from_vector(gdf)
    #     edges, nodes = self.get_network_edges_and_nodes_from_graph(graph)
    #     graph_complex = nut.graph_from_gdf(edges, nodes, node_id="node_fid")
    #     return graph_complex, edges

    @staticmethod
    def get_direct_graph_from_vector(gdf: gpd.GeoDataFrame) -> nx.DiGraph:
        """Creates a simple directed graph with node and edge geometries based on a given GeoDataFrame.

        Args:
            gdf (gpd.GeoDataFrame): Input GeoDataFrame containing line geometries.
                Allow both LineString and MultiLineString.

        Returns:
            nx.DiGraph: NetworkX graph object with "crs", "approach" as graph properties.
        """

        # simple geometry handeling
        gdf = GraphUtils.explode_and_deduplicate_geometries(gdf)

        # to graph
        digraph = nx.DiGraph(crs=gdf.crs, approach="primal")
        for _, row in gdf.iterrows():
            from_node = row.geometry.coords[0]
            to_node = row.geometry.coords[-1]
            digraph.add_node(from_node, geometry=Point(from_node))
            digraph.add_node(to_node, geometry=Point(to_node))
            digraph.add_edge(
                from_node,
                to_node,
                geometry=row.pop(
                    "geometry"
                ),  # **row TODO: check if we do need all columns
            )

        return digraph

    @staticmethod
    def get_indirect_graph_from_vector(gdf: gpd.GeoDataFrame) -> nx.Graph:
        """Creates a simple undirected graph with node and edge geometries based on a given GeoDataFrame.

        Args:
            gdf (gpd.GeoDataFrame): Input GeoDataFrame containing line geometries.
                Allow both LineString and MultiLineString.

        Returns:
            nx.Graph: NetworkX graph object with "crs", "approach" as graph properties.
        """
        digraph = GraphUtils.get_direct_graph_from_vector(gdf)
        return digraph.to_undirected()

    @staticmethod
    def get_network_edges_and_nodes_from_graph(
        graph: nx.Graph,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Sets up network nodes and edges from a given graph.

        Args:
            graph (nx.Graph): Input graph with geometry for nodes and edges.
                Must contain "crs" as graph property.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame representing the network edges with "edge_fid", "node_A", and "node_B".
            gpd.GeoDataFrame: GeoDataFrame representing the network nodes with "node_fid".
        """

        # TODO ths function use conventions. Good to make consistant convention with osm
        nodes, edges = momepy.nx_to_gdf(graph, nodeID="node_fid")
        edges["edge_fid"] = (
            edges["node_start"].astype(str) + "_" + edges["node_end"].astype(str)
        )
        edges.rename(
            {"node_start": "node_A", "node_end": "node_B"}, axis=1, inplace=True
        )
        if not nodes.crs:
            nodes.crs = graph.graph["crs"]
        if not edges.crs:
            edges.crs = graph.graph["crs"]
        return edges, nodes

    @staticmethod
    def clean_vector(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Cleans a GeoDataFrame.

        Args:
            gdf (gpd.GeoDataFrame): Input GeoDataFrame.

        Returns:
            gpd.GeoDataFrame: Cleaned GeoDataFrame.
        """

        gdf = GraphUtils.explode_and_deduplicate_geometries(gdf)

        return gdf

    @staticmethod
    def explode_and_deduplicate_geometries(gpd: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Explodes and deduplicates geometries a GeoDataFrame.

        Args:
            gpd (gpd.GeoDataFrame): Input GeoDataFrame.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with exploded and deduplicated geometries.
        """
        gpd = gpd.explode()
        gpd = gpd[
            gpd.index.isin(
                gpd.geometry.apply(lambda geom: geom.wkb).drop_duplicates().index
            )
        ]
        return gpd
