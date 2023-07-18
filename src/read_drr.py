from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from hydrolib.core.rr import RainfallRunoffModel
from hydrolib.core.rr.topology.parser import NetworkTopologyFileParser


# define customised parser for parsing nwrw location model (hydrolib-core)
class NwrwLocationParser(NetworkTopologyFileParser):
    # overwtite the _parse_line function for reading nwrw area blocks
    def _parse_line(self, line: str) -> dict:
        parts = line.split()

        record = {}

        index = 0
        while index < len(parts) - 1:
            key = parts[index]
            if key in ["ar"]:
                # `ar` has 12 fields
                for i in range(12):
                    index += 1
                    key = "ar" + f"_{i}"
                    value = parts[index].strip("'")
                    record[key] = value
            else:
                if key == "mt" and parts[index + 1] == "1":
                    # `mt 1` is one keyword, but was parsed as two separate parts.
                    index += 1

                index += 1
                value = parts[index].strip("'")
                record[key] = value

            index += 1

        return record


# define customised parser for parsing nwrw general model (hydrolib-core)
class NwrwGeneralParser(NetworkTopologyFileParser):
    # overwtite the _parse_line function for reading nwrw area blocks
    def _parse_line(self, line: str) -> dict:
        parts = line.split()

        record = {}

        # Define a dictionary that maps the keys to the number of fields
        fields_dict = {"rf": 12, "ms": 12, "ix": 4, "im": 4, "ic": 4, "dc": 4, "wh": 24}

        index = 0
        while index < len(parts) - 1:
            key = parts[index]
            _key = key
            num_fields = fields_dict.get(key, 1)
            for i in range(num_fields):
                index += 1
                key = f"{_key}_{i}" if num_fields > 1 else _key
                value = parts[index].strip("'")
                record[key] = value

            index += 1

        return record


# define a RR model (hydromt)
class RRModel(RainfallRunoffModel):

    """Represents the RR model.

    This class encapsulates a RainfallRunoffModel and adds additional functionality
    for reading in RR files and converting their data to GeoDataFrames.

    Attributes:
        data (dict): A dictionary mapping file tags to their corresponding DataFrames.
        geoms (dict): A dictionary mapping geometry names to their corresponding GeoDataFrames.
    """

    data: dict = None
    geoms: dict = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = dict()
        self.geoms = dict()

        self.add_data(
            "nwrw_topology",
            self._read_file(self.filepath.with_name("3brunoff.tp"), "node"),
        )
        self.add_data(
            "nwrw_data", self._read_file(self.filepath.with_name("pluvius.3b"), "nwrw")
        )
        self.add_data(
            "nwrw_general",
            self._read_file(self.filepath.with_name("pluvius.alg"), "plvg"),
        )
        self.add_data(
            "nwrw_dwa", self._read_file(self.filepath.with_name("pluvius.dwa"), "dwa")
        )

        self.add_geom("subcatchments", self._get_subcatchments())

    def add_data(self, name: str, df: pd.DataFrame):
        self.data[name] = df

    def add_geom(self, name: str, geom: gpd.GeoDataFrame):
        self.geoms[name] = geom

    def _read_file(self, fn: Path, tag: str):
        """
        Reads and parses a file, converting it to a DataFrame.

        Depending on the provided tag ("node" or "nwrw"), a location parser is used
        to process the data. All other tags use a general parser.

        Args:
            fn (Path): The path to the file to be read and parsed.
            tag (str): The tag indicating the type of data in the file.

        Returns:
            pd.DataFrame: The parsed data as a DataFrame.
        """
        if tag in ["node", "nwrw"]:
            _parser = NwrwLocationParser(enclosing_tag=tag)
            data = _parser.parse(fn)[tag]
        else:
            _parser = NwrwGeneralParser(enclosing_tag=tag)
            data = _parser.parse(fn)[tag]

        data = pd.DataFrame(data)

        return self._convert_strings_to_floats(data)

    def _convert_strings_to_floats(self, df):
        """
        Converts string values in a DataFrame to floats, if possible.

        Iterates over all columns in the DataFrame. If a column contains string data,
        attempts are made to convert the strings to floats. If a string cannot be converted,
        it is left as is.

        Args:
            df (pd.DataFrame): The DataFrame to process.

        Returns:
            pd.DataFrame: The DataFrame with strings converted to floats where possible.
        """
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors="ignore")

        return df

    def _get_subcatchments(self):
        """
        Merges the nwrw topology and data, converting it to a GeoDataFrame.

        Also calculates the area for each geometry and buffers the geometries based on the area.

        Returns:
            gpd.GeoDataFrame: The processed data as a GeoDataFrame.
        """
        # Merge the nwrw topology and data
        nwrw_df = pd.merge(
            self.data["nwrw_topology"],
            self.data["nwrw_data"],
            on="id",
            how="inner",
        )

        # Convert to GeoDataFrame
        nwrw_gdf = gpd.GeoDataFrame(
            nwrw_df, geometry=gpd.points_from_xy(x=nwrw_df["px"], y=nwrw_df["py"])
        )

        # Calculate area
        nwrw_gdf["area"] = nwrw_gdf[
            [c for c in nwrw_gdf.columns if c.startswith("ar")]
        ].sum(axis=1)

        # Buffer geometries based on area
        nwrw_gdf["geometry"] = nwrw_gdf["geometry"].buffer(
            2 * np.sqrt(nwrw_gdf["area"] / np.pi)
        )

        return nwrw_gdf


# define a NWRW model (hybridurb)
class NwrwModel:
    """Represents the Delft3D Rainfall-Runoff NWRW model.

    Attributes:
    model (DFlowFMModel): The RRModel object.
    root (Path): The root directory for the model.

    Methods:
    write_geoms(region, outdir): Writes the geometries of the model to GeoJSON files.
    """

    model: RRModel = None
    root: Path = None

    def __init__(self, config_fn: Path):
        """
        Initializes the NwrwModel with the specified configuration file.

        The configuration file is used to initialize the underlying RainfallRunoffModel.
        Additional NWRW-specific files are also read in and converted to DataFrames.

        Args:
            config_fn (Path): The path to the configuration file (.fnm).
        """

        self.root = config_fn.parent.parent
        self.model = RRModel(config_fn)

    def write_geoms(self, region: gpd.GeoDataFrame = None, outdir: str = "geoms"):
        """
        Writes the geometries of the NWRW data to GeoJSON files in the specified directory.

        If a region is provided, the geometries are clipped to that region before being written.
        The output directory is created if it does not already exist.

        Args:
            region (gpd.GeoDataFrame, optional): The region to clip the geometries to.
                If None (default), the geometries are not clipped.
            outdir (str, optional): The output directory to write the GeoJSON files to.
                Defaults to "geoms".

        Returns:
            None
        """
        # Ensure the output directory exists
        if not self.root.joinpath(outdir).is_dir():
            self.root.joinpath(outdir).mkdir()

        for name, gdf in self.model.geoms.items():
            if region is not None:
                gdf = gpd.clip(gdf, region)
            _fn = self.root.joinpath(f"{outdir}/{name}.geojson")
            gdf.to_file(_fn)


# Define the root path where the project files are located
root = Path(r"c:\Projects\2023\SITO urban\delft3dfm\dimr_export")

# Define the filename of the Rainfall Runoff Model
fnm_fn = "rr\Sobek_3b.fnm"
fnm = root / fnm_fn

# Parse the Rainfall Runoff Model from the file
model = NwrwModel(fnm)

# Read region
region = gpd.read_file(
    r"c:\Projects\2023\SITO urban\delft3dfm\dimr_export\geoms\clip_region.shp"
)

# write geoms in region
model.write_geoms(region=region, outdir="clip_geoms")

# TODO write other configurations (do I need this? yes if I do nwrw concept)

# TODO: to timeseries in xarray (needed for creating calibration dataset)
# model.bui_file
