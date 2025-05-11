import os
import pandas as pd
import networkx as nx
import xugrid as xu
import geopandas as gpd
from typing import Union, Literal
import numpy as np
import pickle


class Delft3dfmDatasetWrapper:
    """
    A class to manage, process, and visualize 1D Delft3D-FM simulation data as a network graph.

    This class converts UGRID-1D NetCDF outputs into a `networkx.Graph`, attaches both dynamic 
    (time series) and static (spatial) attributes, and provides utilities to integrate shapefile 
    metadata and multiple simulations.

    Args:
        None on initialization. Use classmethods to load data.

    Attributes:
        G (networkx.Graph): Graph representing the 1D network with node and edge properties.
        mesh1d (xugrid.Ugrid1d): UGRID-1D mesh extracted from the Delft3D-FM NetCDF file.
        dataset_list (list[dict]): A list of simulation datasets, each containing:
            - heads_raw_data (pandas.DataFrame): Node water levels over time.
            - runoff_raw_data (pandas.DataFrame): Node runoff volumes over time.
            - flowrate_raw_data (pandas.DataFrame): Edge flowrates over time.

    Class Methods:
        from_netcdf(path): Load mesh and build graph from a single NetCDF file.
        load_from_pickle(path): Restore a saved wrapper instance.

    Instance Methods:
        graph_from_netcdf(path): Create graph from NetCDF (legacy use).
        load_simulation_from_netcdf(path): Load simulation results and set time series attributes.
        load_simulations_from_folder(folder): Load multiple simulations from a folder.
        add_graph_attributes_from_shapefile(...): Attach spatial attributes from shapefiles.
        save_to_pickle(path): Save full object state to disk.
    """
    
    def __init__(self):
        """
        Initializes an empty wrapper instance.
        """
        self.G = None  # NetworkX graph of the mesh1d
        self.mesh1d = None  # xugrid mesh1d object
        self.dataset_list = []  # List of simulation results (dict of DataFrames)


    def graph_from_netcdf(self, path: str) -> nx.Graph:
        """
        Create a NetworkX graph from a 1D UGRID Delft3D-FM NetCDF file.

        Parameters:
            path (str): Path to the NetCDF file.

        Returns:
            nx.Graph: Graph with node/edge properties from the mesh.
        """
        ds_xugrid = xu.open_dataset(path)
        mesh1d = [g for g in ds_xugrid.grids if g.name == "mesh1d"][0]
        ds = ds_xugrid

        node_coords = mesh1d.node_coordinates
        edge_connectivity = mesh1d.edge_node_connectivity
        edge_coords = mesh1d.edge_coordinates

        node_ids = [s.decode('utf-8').strip() if isinstance(s, bytes) else str(s).strip()
                        for s in ds["mesh1d_node_id"].values]

        G = nx.Graph()
        for i, (x, y) in enumerate(node_coords):
            G.add_node(
                node_ids[i],
                x=float(x),
                y=float(y),
                branch=int(ds["mesh1d_node_branch"].values[i]),
                offset=float(ds["mesh1d_node_offset"].values[i])
            )

        for i, (start_idx, end_idx) in enumerate(edge_connectivity):
            start_key = node_ids[start_idx]
            end_key = node_ids[end_idx]
            x, y = edge_coords[i][0], edge_coords[i][1]

            G.add_edge(
                start_key, end_key,
                x=float(x),
                y=float(y),
                branch=int(ds["mesh1d_edge_branch"].values[i]),
                offset=float(ds["mesh1d_edge_offset"].values[i]),
                length=float(ds["Network_edge_length"].values[i])
            )

        self.G = G
        self.mesh1d = mesh1d
    
    
    def load_simulation_from_netcdf(self, path: str) -> dict:
        """
        Load simulation result from a single NetCDF file and update graph attributes.

        Parameters:
            path (str): Path to the NetCDF file.

        Returns:
            dict: Dictionary containing DataFrames for heads, flowrate, and runoff.
        """
        ds_xugrid = xu.open_dataset(path)
        mesh1d = [g for g in ds_xugrid.grids if g.name == "mesh1d"][0]
        ds = ds_xugrid

        node_keys = [s.decode('utf-8').strip() if isinstance(s, bytes) else str(s).strip()
                     for s in ds["mesh1d_node_id"].values]

        edge_keys = []
        for start_idx, end_idx in mesh1d.edge_node_connectivity:
            start_key = node_keys[start_idx]
            end_key = node_keys[end_idx]
            edge_keys.append((start_key, end_key))

        time_index = pd.to_datetime(ds["time"].values)
        heads_df = pd.DataFrame(ds["mesh1d_s1"].values, columns=node_keys, index=time_index)
        runoff_df = pd.DataFrame(ds["mesh1d_current_total_net_inflow_lateral"].values, columns=node_keys, index=time_index)
        flowrate_df = pd.DataFrame(ds["mesh1d_q1"].values, columns=edge_keys, index=time_index)

        # Set full time series as graph attributes
        if self.G is not None:
            h_x_dict = {col: heads_df[col] for col in heads_df.columns}
            ro_x_dict = {col: runoff_df[col] for col in runoff_df.columns}
            q_x_dict = {col: flowrate_df[col] for col in flowrate_df.columns}

            nx.set_node_attributes(self.G, h_x_dict, name="h_x")
            nx.set_node_attributes(self.G, ro_x_dict, name="runoff")
            nx.set_edge_attributes(self.G, q_x_dict, name="q_x")
        
        return {
            "heads_raw_data": heads_df,
            "flowrate_raw_data": flowrate_df,
            "runoff_raw_data": runoff_df
        }

    def add_graph_attributes_from_shapefile(
        self,
        shp: Union[str, gpd.GeoDataFrame],
        attribute_cols: list[str],
        target: Literal["node", "edge"] = "node",
        search_radius: float = None,
        col_mapping: dict[str, str] = None,
    ):
        """
        Attach spatial attributes from a shapefile to nodes or edges of the graph.

        Parameters:
            shp (str | GeoDataFrame): File path or GeoDataFrame of spatial features.
            attribute_cols (list[str]): Column names in shapefile to use.
            target (str): Whether to apply attributes to 'node' or 'edge'.
            search_radius (float): Optional max search distance in projection units.
            col_mapping (dict[str, str]): Optional mapping from shapefile column names
                                        to graph attribute names.
        """
        G = self.G
        mesh1d = self.mesh1d

        if isinstance(shp, str):
            gdf = gpd.read_file(shp)
        else:
            gdf = shp.copy()

        if col_mapping:
            gdf = gdf.rename(columns=col_mapping)
            attribute_cols = [col_mapping.get(col, col) for col in attribute_cols]

        node_keys = list(G.nodes)
        node_coords = mesh1d.node_coordinates
        graph_node_coords = np.array([[G.nodes[key]['x'], G.nodes[key]['y']] for key in node_keys])
        assert np.allclose(node_coords, graph_node_coords), "Node coordinates do not match between mesh1d and graph."

        if target == "node":
            coords = np.array([[geom.x, geom.y] for geom in gdf.geometry])
            matched = mesh1d.locate_nearest_node(coords, max_distance=search_radius)
            keys = node_keys
        elif target == "edge":
            coords = np.array([geom.interpolate(0.5, normalized=True).coords[0][:2] for geom in gdf.geometry])
            matched = mesh1d.locate_nearest_edge(coords, max_distance=search_radius)
            edge_conn = mesh1d.edge_node_connectivity
            keys = [(node_keys[i0], node_keys[i1]) for i0, i1 in edge_conn]
        else:
            raise ValueError("target must be 'node' or 'edge'")

        for gidx, match_idx in enumerate(matched):
            if match_idx == -1:
                continue
            graph_key = keys[match_idx]
            for col in attribute_cols:
                if target == "node":
                    G.nodes[graph_key][col] = gdf.iloc[gidx][col]
                else:
                    G.edges[graph_key][col] = gdf.iloc[gidx][col]

        self.G = G


    def load_simulations_from_folder(self, folder: str, name_list: list[str] = None):
        """
        Load multiple simulations from a folder of NetCDF files.

        Parameters:
            folder (str): Folder path containing NetCDF files.
            name_list (list[str], optional): List of filenames (without extension) to include.
        """
        dataset_list = []

        for fname in sorted(os.listdir(folder)):
            if not fname.endswith(".nc"):
                continue
            name_sim = os.path.splitext(fname)[0]
            if name_list and name_sim not in name_list:
                continue

            path = os.path.join(folder, fname)
            data = self.load_simulation_from_netcdf(path)
            dataset_list.append(data)

        self.dataset_list = dataset_list


    def save_to_pickle(self, path: str):
        """
        Save the entire wrapper state to a pickle file.

        Parameters:
            path (str): Output path for the .pkl file.
        """
        to_save = {
            "graph": self.G,
            "mesh1d": self.mesh1d,  # Warning: not all mesh1d may be serializable
            "dataset_list": self.dataset_list
        }

        try:
            with open(path, "wb") as f:
                pickle.dump(to_save, f)
            print(f"Wrapper state saved to {path}")
        except Exception as e:
            print(f"Failed to save: {e}")

    @classmethod
    def load_from_pickle(cls, path: str) -> "Delft3dfmDatasetWrapper":
        """
        Load a Delft3dfmDatasetWrapper object from a pickle file.

        Parameters:
            path (str): Path to the .pkl file.

        Returns:
            Delft3dfmDatasetWrapper: The restored instance.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        wrapper = cls()
        wrapper.G = data["graph"]
        wrapper.mesh1d = data.get("mesh1d", None)
        wrapper.dataset_list = data["dataset_list"]

        print(f"Wrapper loaded from {path}")
        return wrapper

    @classmethod
    def from_netcdf(cls, path: str) -> "Delft3dfmDatasetWrapper":
        """
        Create a wrapper instance from a 1D UGRID Delft3D-FM NetCDF (map) file,
        including both the graph and simulation time series.

        Parameters:
            path (str): Path to the NetCDF file.

        Returns:
            Delft3dfmDatasetWrapper: New instance with graph, mesh, and time series.
        """
        wrapper = cls()
        wrapper.graph_from_netcdf(path)
        sim_data = wrapper.load_simulation_from_netcdf(path)
        wrapper.dataset_list = [sim_data]
        return wrapper
