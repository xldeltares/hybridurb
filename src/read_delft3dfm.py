from pathlib import Path

import logging
import geopandas as gpd
from hydromt_delft3dfm import DFlowFMModel
from pyproj import CRS


logger = logging.Logger


# define a FM model (hybridurb)
class FMModel(DFlowFMModel):
    """Represents the D-Flow Flexible Mesh (FM) model.

    Attributes:
    model (DFlowFMModel): The DFlowFMModel object.
    root (Path): The root directory for the model.

    Methods:
    write_geoms(region, outdir): Writes the geometries of the model to GeoJSON files.

    NOTE:
    for now only support pipe network
    """

    model: DFlowFMModel = None
    root: Path = None
    crs: CRS = None
    geoms: dict = None

    def __init__(self, config_fn, crs):
        """
        Initializes the FMModel with the specified configuration file.

        Args:
        config_fn (Path): The configuration file for the DFlowFMModel.
        """
        self.root = config_fn.parent.parent
        self.model = DFlowFMModel(root=self.root, config_fn=config_fn, mode="r+")
        self.crs = CRS.from_user_input(crs)
        self.geoms = {}

    def _prepare_geoms(self):
        """
        Prepare geometries by updating the coordinate reference system (CRS) if necessary.
        """
        self.model.geoms["network_nodes"] = self.model.network1d_nodes

        for name, gdf in self.model.geoms.items():
            """
            Iterate through each geometry in the model and check its CRS.
            If the CRS is None, set it to the specified CRS.
            If the CRS is different from the specified CRS, convert it to the specified CRS.
            """
            if gdf.crs is None:
                self.geoms[name] = gdf.set_crs(self.crs)
            elif gdf.crs != self.crs:
                self.geoms[name] = gdf.to_crs(self.crs)

    def write_geoms(self, region: gpd.GeoDataFrame = None, outdir: str = "geoms"):
        """
        Writes the geometries of the model to GeoJSON files.

        If a region is provided, the geometries are clipped to this region before writing.

        The output directory is created if it doesn't exist.

        Args:
        region (gpd.GeoDataFrame, optional): The region to clip the geometries to.
            If None (default), the geometries are not clipped.
        outdir (str, optional): The output directory to write the GeoJSON files to.
            Defaults to "geoms".

        Returns:
        None
        """

        if region is not None:
            for name, gdf in self.model.geoms.items():
                self.model.geoms[name] = gpd.clip(gdf, region)

        if not self.root.joinpath(outdir).is_dir():
            self.root.joinpath(outdir).mkdir()

        # write edges
        self.write_network_edges(
            fn=self.root.joinpath(f"{outdir}/network_edges.geojson")
        )

        # write nodes
        self.write_network_nodes(
            fn=self.root.joinpath(f"{outdir}/network_edges.geojson")
        )

        # write branches
        self.write_branches(fn=self.root.joinpath(f"{outdir}/branches.geojson"))

        # write manholes
        self.write_manholes(fn=self.root.joinpath(f"{outdir}/manholes.geojson"))

        # # write pumps
        # self.write_pumps(fn=self.root.joinpath(f"{outdir}/pumps.geojson"))

        # # write weirs
        # self.write_weirs(fn=self.root.joinpath(f"{outdir}/weirs.geojson"))

    def write_network_edges(self, fn: Path):
        """write network edges, with edgeid and geometry only"""

        assert (
            "branches" in self.geoms
        ), "cannot find branches, network edges cannot be created"
        edges = self.geoms["branches"]

        _supported_edge_types = ["pipe", "sewerconnection"]
        if not set(edges["branchtype"].unique()).issubset(set(_supported_edge_types)):
            logger.warning(
                f"network edges can only be created from {_supported_edge_types} "
            )
        edges = edges.loc[edges["branchtype"].isin(_supported_edge_types), :]

        assert len(edges) > 0, "no pipes found, network edges cannot be created"

        edges.rename(columns={"branchid": "edgeid"}, inplace=True)
        edges[["edgeid", "geometry"]].to_file(fn)

    def write_network_nodes(self, fn: Path):
        """write network nodes, with nodeid and geometry only"""

        assert (
            "network_nodes" in self.geoms
        ), "cannot find network_nodes, network cannot be created"
        nodes = self.geoms["network_nodes"]
        nodes[["nodeid", "geometry"]].to_file(fn)

    def write_branches(self, fn: Path):
        """write branches with crossections, with edgeid and attributes [optional]"""

        if "branches" in self.geoms:
            branches = self.geoms["branches"]
            branches = branches[["branchid", "geometry"]]

        if "crosssections" in self.geoms:
            crosssections = self.geoms["crosssections"].drop(columns="geometry")
            branches = branches.merge(
                crosssections, left_on="branchid", right_on="crsloc_branchid"
            )
            # assign upstream and downstream
            _up = branches["crsloc_chainage"] == 0
            _dn = branches["crsloc_chainage"] != 0
            branches.loc[_up, "bedlevel_up"] = branches.loc[_up, "crsloc_shift"]
            branches.loc[_dn, "bedlevel_dn"] = branches.loc[_dn, "crsloc_shift"]
            # rename crsloc
            branches.rename(
                columns={"crsloc_branchid": "edgeid"},
                inplace=True,
            )
            branches.rename(
                columns={
                    "crsdef_type": "shape",
                    "crsdef_diameter": "diameter",
                    "crsdef_width": "width",
                    "crsdef_height": "height",
                },
                inplace=True,
            )

            _supported_crossection_shapes = ["circle", "rectangle"]
            if not set(branches["shape"].unique()).issubset(
                set(_supported_crossection_shapes)
            ):
                NotImplementedError(
                    f"crossections other than {_supported_crossection_shapes} are not supported."
                )

            def _calculate_area(x):
                if x.shape == "rectangle":
                    return x.width * x.length
                elif x.shape == "circle":
                    return 3.1415926 * x.diameter**2

            branches["area"] = branches.apply(_calculate_area)
            branches["length"] = branches.geometry.length

            branches[
                [
                    "edgeid",
                    "shape",
                    "diameter",
                    "width",
                    "height",
                    "area",
                    "length",
                    "geometry",
                ]
            ].to_file(fn)

    def write_manholes(self, fn: Path):
        """write manholes, with nodeid and attributes [optional]"""

        def _calculate_depth(row):
            return row.streetlevel - row.bedlevel

        if "manholes" in self.geoms:
            manholes = self.geoms["manholes"]

            manholes["depth"] = manholes.apply(_calculate_depth, axis=1)
            manholes[["bedlevel", "streetlevel", "depth", "area", "geometry"]].to_file(
                fn
            )


# Define the root path where the project files are located
root = Path(r"c:\Projects\2023\SITO urban\delft3dfm\dimr_export")

# Define the filename of the Rainfall Runoff Model
mdu_fn = "dflowfm\FlowFM.mdu"
mdu = root / mdu_fn

# Parse the Rainfall Runoff Model from the file
model = FMModel(mdu, crs=28992)

# Read region
region = gpd.read_file(
    r"c:\Projects\2023\SITO urban\delft3dfm\dimr_export\geoms\clip_region.shp"
)

# write geoms in region
model.write_geoms(region=region, outdir="clip_geoms")
