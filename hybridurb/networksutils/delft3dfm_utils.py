import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import hydromt_delft3dfm
import numpy as np
import pandas as pd
from hydromt_delft3dfm import DFlowFMModel
from pyproj import CRS

from hybridurb.hydraulics_utils import HydraulicUtils

logger = logging.getLogger(__name__)


__all__ = [
    "get_geometries_from_model",
    "extract_geometries",
    "convert_geometries",
    "save_geometries",
]


def get_geometries_from_model(
    model_root: Path,
    mdu_fn: str,
    crs: CRS,
    clip_region: Optional[gpd.GeoDataFrame] = None,
    geoms_dir: Optional[str] = "geoms_dir",
):
    """
    Retrieves geometries from a given model directory.

    :param model_root: The model root directory.
    :param mdu_fn: The mdu file as relative path to model_root.
    :param crs: The coordinate reference system.
    :param clip_region: The region to clip geoms. Defaults to None.
    :param geoms_dir: the dir to save the geoms.
    :return: The geometries from the model directory.
    """

    # read model
    model = DFlowFMModel(root=model_root, config_fn=mdu_fn, mode="r+", crs=crs)
    # extract geometries
    geometries = extract_geometries(model, clip_region)
    # convert geometries into hybridurb convention
    geometries = convert_geometries(geometries)
    # save the geometries to original data folder
    save_geometries(geometries, geoms_dir)
    return geometries


def extract_geometries(
    model: DFlowFMModel, clip_region: Optional[gpd.GeoDataFrame] = None
):
    """
    Extracts geometries from the DFlowFM model. If a region is provided, the geometries will be clipped to this region.

    Parameters:
    model (DFlowFMModel): A DFlowFMModel instance containing the model whoes geometries will be extracted.
    clip_region (gpd.GeoDataFrame, optional): A GeoPandas DataFrame containing the region to which the geometries will be clipped.
                                            If no region is provided, all geometries from the model will be returned.

    Returns:
    dict: A dictionary where the keys are the names of the geometries and the values are GeoPandas DataFrames containing the geometries.
            Each DataFrame has a 'geometry' column containing the geometry objects, and the rest of the columns contain attributes of the geometries.
    """
    if model is None:
        raise ValueError("Model has not been read. Please run read_model() first.")

    model.geoms["network_edges"] = model.geoms["branches"].rename(
        columns={"branchid": "edgeid"}
    )
    model.geoms[
        "network_nodes"
    ] = hydromt_delft3dfm.mesh_utils.network1d_nodes_geodataframe(
        model.mesh_datasets["network1d"]
    )

    geometries = {}
    for name, gdf in model.geoms.items():
        if clip_region is not None:
            geometries[name] = gpd.clip(gdf, clip_region)
        else:
            geometries[name] = gdf

    return geometries


def convert_geometries(geometries) -> dict:
    """
    Converts geometries and calculates additional attributes.

    Returns:
    dict: A dictionary containing the converted geometries.
    """
    _geometries = geometries.copy()
    geometries = {}
    _crosssections = _geometries.pop("crosssections")  # part of branches
    for name, gdf in _geometries.items():
        if name == "network_edges":
            geometries[name] = convert_edges(gdf)
        elif name == "network_nodes":
            geometries[name] = convert_nodes(gdf)
        elif name == "branches":
            geometries[name] = convert_branches(
                branches=gdf,
                crosssections=_crosssections,
            )
        elif name == "manholes":
            geometries[name] = convert_manholes(gdf)
        elif name == "pumps":
            geometries[name] = convert_pumps(gdf)
        elif name == "weirs":
            geometries[name] = convert_weirs(gdf)
        else:
            # not implemented
            pass

    return geometries


def save_geometries(geometries, out_dir: Path):
    """save geometries to dir"""

    out_dir.mkdir(exist_ok=True)

    for name, gdf in geometries.items():
        _fn = out_dir.joinpath(f"{name}.geojson")
        gdf.to_file(_fn)


def convert_edges(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert edges geometries.

    Parameters:
    gdf (gpd.GeoDataFrame): A GeoDataFrame containing the edges geometries to be converted. It must include 'edgeid' and 'geometry' columns.

    Returns:
    gpd.GeoDataFrame: A GeoDataFrame containing the converted edges geometries.
    """
    return gdf[["edgeid", "geometry"]]


def convert_nodes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert nodes geometries.

    Parameters:
    gdf (gpd.GeoDataFrame): A GeoDataFrame containing the nodes geometries to be converted. It must include 'nodeid' and 'geometry' columns.

    Returns:
    gpd.GeoDataFrame: A GeoDataFrame containing the converted nodes geometries.
    """
    return gdf[["nodeid", "geometry"]]


def convert_branches(
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

    def _split_string_to_array(s):
        return np.array([float(value) for value in s.split()])

    def _calculate_area(row):
        if row.type == "rectangle":
            return HydraulicUtils.calculate_rectangle_area(row.height, row.width)
        elif row.type == "circle":
            return HydraulicUtils.calculate_circle_area(row.diameter)
        elif row.type == "zw":
            levels = _split_string_to_array(row.levels)
            flowwidths = _split_string_to_array(row.flowwidths)
            return HydraulicUtils.calculate_zw_area(levels, flowwidths)

    def _calculate_perimeter(row):
        if row.type == "rectangle":
            return HydraulicUtils.calculate_rectangle_perimeter(row.height, row.width)
        elif row.type == "circle":
            return HydraulicUtils.calculate_circle_perimeter(row.diameter)
        elif row.type == "zw":
            levels = _split_string_to_array(row.levels)
            flowwidths = _split_string_to_array(row.flowwidths)
            closed = bool(row.closed)
            return HydraulicUtils.calculate_zw_perimeter(levels, flowwidths, closed)

    def _get_shift_at_chainage_begin(df):
        # first, find the index of the minimum chainage for each branchid
        idx = df.groupby("branchid")["chainage"].idxmin()

        # then, create a lookup dataframe with minimum chainage shift values for each branchid
        lookup_df = df.loc[idx, ["branchid", "shift"]].set_index("branchid")
        return df["branchid"].map(lookup_df["shift"])

    def _get_shift_at_chainage_end(df):
        # first, find the index of the maximum chainage for each branchid
        idx = df.groupby("branchid")["chainage"].idxmax()

        # then, create a lookup dataframe with maximum chainage shift values for each branchid
        lookup_df = df.loc[idx, ["branchid", "shift"]].set_index("branchid")
        return df["branchid"].map(lookup_df["shift"])

    def _parse_crosssections(crosssections):
        # get locations
        crslocs = crosssections[
            [c for c in crosssections.columns if c.startswith("crsloc")]
        ]
        # get definition
        crsdef = crosssections[
            [
                c
                for c in crosssections.columns
                if (c.startswith("crsdef") or c.startswith("friction"))
                and not c.startswith("crsdef_friction")
            ]  # exclude friction
        ]
        crs = pd.concat([crslocs, crsdef], axis=1)
        crs.rename(
            columns={c: c[7:] for c in crs.columns if c.startswith("crs")},
            inplace=True,
        )
        crs["geometry"] = crosssections["geometry"]
        return crs

    _supported_branch_types = ["pipe"]
    if not branches["branchtype"].isin(_supported_branch_types).any():
        logger.warning(f"branches only support {_supported_branch_types}")
        branches = branches.loc[branches["branchtype"].isin(_supported_branch_types), :]
        crosssections = crosssections[
            crosssections["crsloc_branchid"].isin(branches["branchid"])
        ]

    _supported_crossection_shapes = ["circle", "rectangle", "zw"]
    if not set(crosssections["crsdef_type"].unique()).issubset(
        set(_supported_crossection_shapes)
    ):
        logger.warning(
            f"crossections other than {_supported_crossection_shapes} are not supported."
        )
        crosssections = crosssections.loc[
            crosssections["crsdef_type"].isin(_supported_crossection_shapes), :
        ]

    _supported_roughness_types = ["WhiteColebrook"]
    if not set(crosssections["frictiontype"].unique()).issubset(
        set(_supported_roughness_types)
    ):
        logger.warning(
            f"crossections other than {_supported_roughness_types} are not supported."
        )
        crosssections = crosssections.loc[
            crosssections["frictiontype"].isin(_supported_roughness_types),
            :,
        ]
    branches = branches[branches["branchid"].isin(crosssections["crsloc_branchid"])]

    if branches.empty or crosssections.empty:
        return None

    # preprocess crossections
    crosssections = _parse_crosssections(crosssections)
    # FIXME: assumes always at the beginning and end
    crosssections["invlev_up"] = _get_shift_at_chainage_begin(crosssections)
    crosssections["invlev_dn"] = _get_shift_at_chainage_end(crosssections)

    # Add crosssections to branches
    branches = branches[["branchid", "branchtype", "geometry"]].merge(
        crosssections.drop(columns="geometry"), on="branchid"
    )
    # FIXME: do I need to remove the crossections that are none for friction?
    # add static parameters
    branches["length"] = branches.geometry.length
    branches["area"] = branches.apply(_calculate_area, axis=1)
    branches["perimeter"] = branches.apply(_calculate_perimeter, axis=1)
    branches["gradient"] = (branches["invlev_dn"] - branches["invlev_up"]) / branches[
        "length"
    ]

    # add edgeid
    branches["edgeid"] = branches["branchid"]

    return branches[
        [
            "edgeid",
            "branchid",
            "branchtype",
            "area",
            "perimeter",
            "length",
            "gradient",
            "invlev_up",
            "invlev_dn",
            "frictiontype",
            "frictionvalue",
            "geometry",
        ]
    ]


def convert_manholes(manholes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert manholes geometries.

    Parameters:
    manholes (gpd.GeoDataFrame): A GeoDataFrame containing the manholes geometries to be converted. It must include 'nodeid' and 'geometry' columns.

    Returns:
    gpd.GeoDataFrame: A GeoDataFrame containing the converted nodes geometries.
    """

    manholes["depth"] = manholes["streetlevel"] - manholes["bedlevel"]
    manholes["volume"] = manholes["depth"] * manholes["area"]

    return manholes[
        ["nodeid", "bedlevel", "streetlevel", "depth", "area", "volume", "geometry"]
    ]


def convert_pumps(pumps: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert pumps geometries."""
    pumps["structureid"] = pumps["id"]
    pumps["structuretype"] = pumps["type"]
    pumps["edgeid"] = pumps["branchid"]
    # FIXME: support only single capacity, suction side control
    pumps["invlev"] = pumps["startlevelsuctionside"].apply(
        lambda x: x[0] if len(x) > 0 else None
    )
    return pumps[
        ["edgeid", "structureid", "structuretype", "invlev", "capacity", "geometry"]
    ]


def convert_weirs(weirs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert weirs geometries."""
    weirs["structureid"] = weirs["id"]
    weirs["structuretype"] = weirs["type"]
    weirs["edgeid"] = weirs["branchid"]
    weirs["invlev"] = weirs["crestlevel"]
    # FIXME: support only simple weir
    return weirs[["edgeid", "structureid", "structuretype", "invlev", "geometry"]]
