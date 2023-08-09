# -*- coding: utf-8 -*-

import logging
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import shapely
import geopandas as gpd
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
import logging
import networkx as nx
import contextily as ctx
import random
from networkx.drawing.nx_agraph import graphviz_layout
from scipy.spatial import distance
from scipy.spatial import KDTree
import numpy as np

logger = logging.getLogger(__name__)


__all__ = [
    "multipage",
    "reduce_gdf_precision",
    "snap_branch_ends",
    "snap_nodes_to_nodes",
]


# IO
def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format="pdf")
    pp.close()


def reduce_gdf_precision(gdf: gpd.GeoDataFrame, rounding_precision=8):
    if isinstance(gdf.geometry[0], LineString):
        branches = gdf.copy()
        for i_branch, branch in enumerate(branches.itertuples()):
            points = shapely.wkt.loads(
                shapely.wkt.dumps(
                    branch.geometry, rounding_precision=rounding_precision
                )
            ).coords[:]
            branches.at[i_branch, "geometry"] = LineString(points)

    elif isinstance(gdf.geometry[0], Point):
        points = gdf.copy()
        for i_point, point in enumerate(points.itertuples()):
            new_point = shapely.wkt.loads(
                shapely.wkt.dumps(point.geometry, rounding_precision=rounding_precision)
            ).coords[:]
            points.at[i_point, "geometry"] = Point(new_point)

    else:
        raise NotImplementedError

    return gdf


def snap_branch_ends(
    branches: gpd.GeoDataFrame,
    offset: float = 0.01,
    subsets=[],
    max_points: int = np.inf,
    id_col="BRANCH_ID",
):
    """
    Helper to snap branch ends to other branch ends within a given offset.


    Parameters
    ----------
    branches : gpd.GeoDataFrame
    offset : float [m]
        Maximum distance between end points. If the distance is larger, they are not snapped.
    subset : list
        A list of branch id subset to perform snapping (forced snapping)
    max_points: int
        maximum points allowed in a group.
        if snapping branch ends only, use max_points = 2
        if not specified, branch intersections will also be snapped
    Returns
    branches : gpd.GeoDataFrame
        Branches updated with snapped geometry
    """
    # Collect endpoints
    _endpoints = []
    for branch in branches.itertuples():
        _endpoints.append((branch.geometry.coords[0], branch.Index, 0))
        _endpoints.append((branch.geometry.coords[-1], branch.Index, -1))

    # determine which branches should be included
    if len(subsets) > 0:
        _endpoints = [[i for i in _endpoints if i[1] in subsets]]
    else:
        _endpoints = _endpoints

    # # group branch ends based on off set
    groups = {}
    coords = [i[0] for i in _endpoints]
    dist = distance.squareform(distance.pdist(coords))
    bdist = dist <= offset
    for row_i, row in enumerate(bdist):
        groups[_endpoints[row_i]] = []
        for col_i, col in enumerate(row):
            if col:
                groups[_endpoints[row_i]].append(_endpoints[col_i])

    # remove duplicated group, group that does not satisfy max_points in groups. Assign endpoints
    endpoints = {
        k: list(set(v))
        for k, v in groups.items()
        if (len(set(v)) >= 2) and (len(set(v)) <= max_points)
    }
    logger.debug(
        "Limit snapping to allow a max number of {max_points} contact points. If max number == 2, it means 1 to 1 snapping."
    )

    # Create a counter
    snapped = 0

    # snap each group (list) in endpoints together, by using the coords from the first point
    for point_reference, points_to_snap in endpoints.items():
        # get the point_reference coords as reference point
        ref_crd = point_reference[0]
        # for each of the rest
        for j, (endpoint, branchid, side) in enumerate(points_to_snap):
            # Change coordinates of branch
            crds = branches.at[branchid, "geometry"].coords[:]
            if crds[side] != ref_crd:
                crds[side] = ref_crd
                branches.at[branchid, "geometry"] = LineString(crds)
                snapped += 1
    logger.debug(f"Snapped {snapped} points.")

    return branches


def snap_nodes_to_nodes(
    nodes: gpd.GeoDataFrame, nodes_prior: gpd.GeoDataFrame, offset: float
):
    """
    Method to snap nodes to nodes_prior.
    index of nodes_prior will be overwritten by index of nodes.
    index column will be retained as in nodes_prior.

    Parameters
    offset : float
        Maximum distance between end points. If the distance is larger, they are not snapped.
    """

    id_col = str(nodes_prior.index.name)

    # Collect points_prior in nodes_prior
    points_prior = [
        (node.geometry.x, node.geometry.y) for node in nodes_prior.itertuples()
    ]
    points_prior_index = nodes_prior.index.to_list()

    # Collect points in nodes
    points = [(node.geometry.x, node.geometry.y) for node in nodes.itertuples()]
    points_index = nodes.index.to_list()

    # Create KDTree of points_prior
    snapped = 0

    # For every points determine the distance to the nearest points_prior
    for pointid, point in zip(points_index, points):
        mindist, minidx = KDTree(points_prior).query(point)

        # snapped if dist is smaller than offset (also include 0.0).
        if mindist <= offset:
            # Change index columns of nodes_prior
            points_prior_index[minidx] = pointid
            snapped += 1

    logger.info(f"Snapped {snapped} points.")
    # reindex
    nodes_prior_ = nodes_prior.copy()
    nodes_prior_["ORIG_" + id_col] = list(nodes_prior.index)
    nodes_prior_[id_col] = points_prior_index
    nodes_prior_.index = nodes_prior_[id_col]
    return nodes_prior_
