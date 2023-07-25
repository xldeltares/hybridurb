from pathlib import Path
from typing import Optional
import logging
import geopandas as gpd
from hydromt_delft3dfm import DFlowFMModel
from pyproj import CRS
import numpy as np

logger = logging.getLogger()


class Delft3dFMReader:
    """Class for reading a D-Flow FM model and extracting its geometries."""

    def __init__(self, mdu_path: Path, crs: int):
        """
        Initialize a Delft3dFMReader with an MDU file and a CRS.

        Parameters:
        mdu_path (Path): Path to the MDU file. This file is required to initialize the DFlowFM model.
        crs (int): Coordinate Reference System (CRS) as an EPSG code for the geometries.

        Returns:
        None
        """
        self.mdu_path = Path(mdu_path)
        self.crs = CRS.from_user_input(crs)
        self.model = None

    def read_model(self):
        """Reads the DFlowFM model."""
        self.model = DFlowFMModel(self.mdu_path, crs=self.crs)

    def extract_geometries(self, region: Optional[gpd.GeoDataFrame]):
        """
        Extracts geometries from the DFlowFM model. If a region is provided, the geometries will be clipped to this region.

        Parameters:
        region (gpd.GeoDataFrame, optional): A GeoPandas DataFrame containing the region to which the geometries will be clipped.
                                              If no region is provided, all geometries from the model will be returned.

        Returns:
        dict: A dictionary where the keys are the names of the geometries and the values are GeoPandas DataFrames containing the geometries.
              Each DataFrame has a 'geometry' column containing the geometry objects, and the rest of the columns contain attributes of the geometries.
        """
        if self.model is None:
            raise ValueError("Model has not been read. Please run read_model() first.")

        self.model.geoms["network_edges"] = self.geoms["branches"].rename(
            columns={"branchid": "edgeid"}
        )
        self.model.geoms["network_nodes"] = self.model.network1d_nodes

        geometries = {}
        for name, gdf in self.model.geoms.items():
            if self.crs is not None:
                geometries[name] = gdf.set_crs(self.crs, allow_override=True)
            else:
                geometries[name] = gdf

        if region is not None:
            for name, gdf in geometries.items():
                geometries[name] = gpd.clip(gdf, region)

        return geometries


class Delft3DFMConverter:
    """Class for converting D-Flow FM model geometries and calculating additional attributes."""

    def __init__(self):
        """Initialize a Delft3DFMConverter."""
        pass

    def convert(self, geometries: dict) -> dict:
        """
        Converts geometries and calculates additional attributes.

        Parameters:
        geometries (dict): A dictionary where the keys are the names of the geometries and the values are GeoPandas DataFrames containing the geometries.
                           Each DataFrame should have a 'geometry' column containing the geometry objects, and the rest of the columns contain attributes of the geometries.

        Returns:
        dict: A dictionary containing the converted geometries.
        """
        for name, gdf in geometries.items():
            if name == "network_edges":
                geometries[name] = self.convert_edges(gdf)
            elif name == "network_nodes":
                geometries[name] = self.convert_nodes(gdf)
            elif name == "branches":
                geometries[name] = self.convert_branches(gdf)
            elif name == "manholes":
                geometries[name] = self.convert_manholes(gdf)
            elif name == "pumps":
                geometries[name] = self.convert_pumps(gdf)
            elif name == "weirs":
                geometries[name] = self.convert_weirs(gdf)
            else:
                # not implemented
                pass

        return geometries

    def convert_edges(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Convert edges geometries.

        Parameters:
        gdf (gpd.GeoDataFrame): A GeoDataFrame containing the edges geometries to be converted. It must include 'edgeid' and 'geometry' columns.

        Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the converted edges geometries.
        """
        return gdf[["edgeid", "geometry"]]

    def convert_nodes(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Convert nodes geometries.

        Parameters:
        gdf (gpd.GeoDataFrame): A GeoDataFrame containing the nodes geometries to be converted. It must include 'nodeid' and 'geometry' columns.

        Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the converted nodes geometries.
        """
        return gdf[["nodeid", "geometry"]]

    def convert_branches(
        self,
        branches: Optional[gpd.GeoDataFrame],
        crosssections: Optional[gpd.GeoDataFrame],
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Converts branches geometries and calculates additional attributes.

        Parameters:
        branches (gpd.GeoDataFrame, optional): A GeoDataFrame containing the branches geometries to be converted.
        crosssections (gpd.GeoDataFrame, optional): A GeoDataFrame containing the crosssections.

        Returns:
        gpd.GeoDataFrame, optional: GeoDataFrame containing the converted branches geometries.
        """
        _supported_branch_types = ["pipe", "sewerconnection"]
        if not branches["branchtype"].isin(_supported_branch_types).any():
            logger.warning(f"branches only support {_supported_branch_types}")
            branches = branches.loc[
                branches["branchtype"].isin(_supported_branch_types), :
            ]

        if branches.empty or crosssections is None:
            return None

        crosssections.rename(
            columns={c: c[7:] for c in crosssections.columns if c.startswith("crs")},
            inplace=True,
        )
        _supported_crossection_shapes = ["circle", "rectangle", "zw"]
        if not set(crosssections["type"].unique()).issubset(
            set(_supported_crossection_shapes)
        ):
            raise NotImplementedError(
                f"crossections other than {_supported_crossection_shapes} are not supported."
            )

        # Add crosssections to branches
        branches = branches[["branchid", "geometry"]].merge(
            crosssections, on="branchid"
        )

        def _calculate_flow_area(row):
            if row.shape == "rectangle":
                return row.width * row.length
            elif row.shape == "circle":
                return np.pi * row.diameter**2
            elif row.shape == "zw":
                return Delft3DFMUtils.calculate_zw_flow_area(row.levels, row.flowwidths)

        branches["area"] = branches.apply(_calculate_flow_area, axis=1)
        branches["length"] = branches.geometry.length
        branches.loc[branches["chainage"] == 0, "invlev_up"] = branches.loc[
            branches["chainage"] == 0, "shift"
        ]
        branches.loc[branches["chainage"] != 0, "invlev_dn"] = branches.loc[
            branches["chainage"] != 0, "shift"
        ]
        branches["gradient"] = (
            branches["invlev_dn"] - branches["invlev_up"]
        ) / branches["length"]

        # TODO: calculate capacity
        # TODO: calculate roughness
        # TODO: add branchtype

        branches["edge_id"] = branches["branchid"]

        return branches[
            [
                "edgeid",
                "branchid",
                "branchtype",
                "shape",
                "diameter",
                "width",
                "height",
                "area",
                "length",
                "geometry",
            ]
        ]

    def convert_manholes(self, manholes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Convert manholes geometries.

        Parameters:
        manholes (gpd.GeoDataFrame): A GeoDataFrame containing the manholes geometries to be converted. It must include 'nodeid' and 'geometry' columns.

        Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the converted nodes geometries.
        """

        manholes["depth"] = manholes["streetlevel"] - manholes["bedlevel"]
        manholes["volume"] = manholes["depth"] * manholes["area"]

        return manholes[
            ["bedlevel", "streetlevel", "depth", "area", "volume", "geometry"]
        ]

    def convert_pumps(self, pumps: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Convert pumps geometries."""
        return pumps

    def convert_weirs(self, weirs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Convert weirs geometries."""
        return weirs


class Delft3DFMUtils:
    """Class for utility functions related to D-Flow FM model."""

    @staticmethod
    def calculate_zw_flow_area(levels, flow_widths):
        """
        Calculate the flow area for 'zw' type crosssections.

        Parameters:
        levels (numpy.ndarray): Array of levels.
        flow_widths (numpy.ndarray): Array of flow widths.

        Returns:
        numpy.ndarray: Array of flow areas.
        """
        return np.trapz(flow_widths, x=levels)

    @staticmethod
    def calculate_zw_wet_radius(levels, flow_widths, closed):
        """
        Calculate the wet radius for 'zw' type crosssections.

        Parameters:
        levels (numpy.ndarray): Array of levels.
        flow_widths (numpy.ndarray): Array of flow widths.
        closed (bool): Whether the crosssection is closed.

        Returns:
        numpy.ndarray: Array of wet radii.
        """
        return np.trapz(flow_widths, x=levels) / (
            (flow_widths[-1] - flow_widths[0])
            if closed
            else np.trapz(flow_widths / levels, x=levels)
        )
