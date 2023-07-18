from hydromt_delft3dfm import DFlowFMModel
import geopandas as gpd
from pathlib import Path


# define a FN model (hybridurb)
class FMModel(DFlowFMModel):
    """Represents the D-Flow Flexible Mesh (FM) model.

    Attributes:
    model (DFlowFMModel): The DFlowFMModel object.
    root (Path): The root directory for the model.

    Methods:
    write_geoms(region, outdir): Writes the geometries of the model to GeoJSON files.
    """

    model: DFlowFMModel = None
    root: Path = None

    def __init__(self, config_fn):
        """
        Initializes the FMModel with the specified configuration file.

        Args:
        config_fn (Path): The configuration file for the DFlowFMModel.
        """
        self.root = config_fn.parent.parent
        self.model = DFlowFMModel(root=self.root, config_fn=config_fn, mode="r+")

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
        if not self.root.joinpath(outdir).is_dir():
            self.root.joinpath(outdir).mkdir()

        self.model.geoms["network_nodes"] = self.model.network1d_nodes

        for name, gdf in self.model.geoms.items():
            if region is not None:
                gdf = gpd.clip(gdf, region)
            _fn = self.root.joinpath(f"{outdir}/{name}.geojson")
            gdf.to_file(_fn)


# Define the root path where the project files are located
root = Path(r"c:\Projects\2023\SITO urban\delft3dfm\dimr_export")

# Define the filename of the Rainfall Runoff Model
mdu_fn = "dflowfm\FlowFM.mdu"
mdu = root / mdu_fn

# Parse the Rainfall Runoff Model from the file
model = FMModel(mdu)

# Read region
region = gpd.read_file(
    r"c:\Projects\2023\SITO urban\delft3dfm\dimr_export\geoms\clip_region.shp"
)

# write geoms in region
model.write_geoms(region=region, outdir="clip_geoms")
