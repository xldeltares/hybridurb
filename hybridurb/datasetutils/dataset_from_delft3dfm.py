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
        dataset_list (list[dict]): A list of simulation datasets, each containing time series data 
            for nodes and edges.

    Class Methods:
        graph_from_netcdf(path): Create graph from NetCDF.
        graph_from_pickle(path): Restore a saved wrapper instance from a pickle file.

    Instance Methods:
        add_graph_attributes_from_shapefile(...): Attach spatial attributes from shapefiles.
        load_dataset_from_folder(folder, node_vars, edge_vars): Load multiple simulations from a folder.
        load_dataset_from_netcdf(path, node_vars, edge_vars): Load a single simulation dataset.
        add_dynamic_attributes_from_dataset(...): Add dynamic attributes to the graph.
        save_to_pickle(path): Save full object state to disk.
    """
    
    def __init__(self):
        """
        Initializes an empty wrapper instance.
        """
        self.G = None  # NetworkX graph of the mesh1d
        self.mesh1d = None  # xugrid mesh1d object
        self.dataset_list = []  # List of simulation results (dict of DataFrames)

    @ classmethod
    def graph_from_netcdf(cls, path: str) -> "Delft3dfmDatasetWrapper":
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

        wrapper = cls()
        wrapper.G = G
        wrapper.mesh1d = mesh1d
        return wrapper

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

    def load_dataset_from_folder(self, folder: str, node_vars: list = ["mesh1d_s1"], edge_vars: list = ["mesh1d_q1"], name_list: list = None):
        """
        Load multiple simulations from a folder of NetCDF files.

        Parameters:
            folder (str): Folder path containing NetCDF files.
            node_vars (list, optional): List of node variable names to extract. Defaults to ["mesh1d_s1"].
            edge_vars (list, optional): List of edge variable names to extract. Defaults to ["mesh1d_q1"].
            name_list (list[str], optional): List of filenames (without extension) to include.
        """
        dataset_list = []

        for fname in sorted(os.listdir(folder)):
            if not fname.endswith(".nc"):
                continue
            if name_list and (fname not in name_list):
                continue
            path = os.path.join(folder, fname)
            data = self.load_dataset_from_netcdf(path, node_vars=node_vars, edge_vars=edge_vars)
            dataset_list.append(data)

        self.dataset_list = dataset_list

    def load_dataset_from_netcdf(self, path: str, node_vars: list = ["mesh1d_s1"], edge_vars: list = ["mesh1d_q1"]) -> dict:
        
        """	
        This function loads a single NetCDF file and returns a dictionary of DataFrames
        containing the time series data for the specified node and edge variables.  
        The function also checks if the mesh1d in the dataset is identical to the existing mesh1d
        in the wrapper. If not, it raises a ValueError.

        Parameters:
            path (str): Path to the NetCDF file.
            node_vars (list, optional): List of node variable names to extract. Defaults to ["mesh1d_s1"].
            edge_vars (list, optional): List of edge variable names to extract. Defaults to ["mesh1d_q1"].
        
        Returns:
            dict: Dictionary containing DataFrames for node and edge variables.
        """	
        
        ds_xugrid = xu.open_dataset(path)
        mesh1d = [g for g in ds_xugrid.grids if g.name == "mesh1d"][0]

        if self.mesh1d is not None:
            if not mesh1d.equals(self.mesh1d):
                raise ValueError("Mesh1D in the dataset does not match the existing mesh1d.")
            
        ds = ds_xugrid

        node_keys = [s.decode('utf-8').strip() if isinstance(s, bytes) else str(s).strip()
                     for s in ds["mesh1d_node_id"].values]

        edge_keys = []
        for start_idx, end_idx in mesh1d.edge_node_connectivity:
            start_key = node_keys[start_idx]
            end_key = node_keys[end_idx]
            edge_keys.append((start_key, end_key))

        time_index = pd.to_datetime(ds["time"].values)
        node_vars = node_vars or ["mesh1d_s1"]
        edge_vars = edge_vars or ["mesh1d_q1"]

        dataset = {}
        for node_var in node_vars:
            if node_var not in ds.variables:
                raise ValueError(f"Node variable {node_var} not found in dataset.") 
            dataset[node_var] = pd.DataFrame(ds[node_var].values, columns=node_keys, index=time_index)
        for edge_var in edge_vars:
            if edge_var not in ds.variables:
                raise ValueError(f"Edge variable {edge_var} not found in dataset.") 
            dataset[edge_var] = pd.DataFrame(ds[edge_var].values, columns=edge_keys, index=time_index)    
        
        return dataset

    def add_dynamic_attributes_from_dataset(self,
        dataset: dict,
        attribute_cols: list[str],
        target: Literal["node", "edge"] = "node",
        col_mapping: dict[str, str] = None):
        """
        Add dynamic attributes from a dataset to the graph.

        Parameters:
            dataset (dict): Dictionary containing time series data for nodes and edges. Read from `load_dataset_from_netcdf` or 
                item of `load_dataset_from_folder`.  
            attribute_cols (list[str]): List of attribute names to add.
            target (str): Whether to apply attributes to 'node' or 'edge'.
            col_mapping (dict[str, str]): Optional mapping from dataset keys to graph attribute names.
        """
        if self.G is None:
            raise ValueError("Graph is not initialized.")

        col_mapping = col_mapping or {}
        for attr in attribute_cols:
            mapped_attr = col_mapping.get(attr, attr)
            if target == "node":
                if isinstance(dataset[attr], pd.DataFrame):
                    data_dict = {node: dataset[attr][node] for node in dataset[attr].columns}
                    nx.set_node_attributes(self.G, data_dict, name=mapped_attr)
                else:
                    raise ValueError(f"Dataset attribute {attr} is not a DataFrame.")
            elif target == "edge":
                if isinstance(dataset[attr], pd.DataFrame):
                    data_dict = {edge: dataset[attr][edge] for edge in dataset[attr].columns}
                    nx.set_edge_attributes(self.G, data_dict, name=mapped_attr)
                else:
                    raise ValueError(f"Dataset attribute {attr} is not a DataFrame.")
            else:
                raise ValueError("Target must be 'node' or 'edge'.")
        

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
    def graph_from_pickle(cls, path: str) -> "Delft3dfmDatasetWrapper":
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

