import unittest
import tempfile
from pathlib import Path
import importlib.util
import geopandas as gpd
from shapely.geometry import Polygon

# Load the script directly
SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "gee"
    / "prepare_csd_boundary_asset.py"
)
spec = importlib.util.spec_from_file_location(
    "prepare_csd_boundary_asset", SCRIPT_PATH
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
prepare_csd_boundary_asset = module.prepare_csd_boundary_asset


def make_synthetic_gdf(duplicate=False, missing_uid=False):
    """Create a tiny GeoDataFrame with required and extra columns.
    Parameters allow us to trigger validation errors.
    """
    polys = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
             Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])]
    data = {
        "CSDUID": ["001", "002"],
        "CSDNAME": ["A", "B"],
        "PRUID": ["01", "02"],
        "PRNAME": ["ProvinceA", "ProvinceB"],
        "FORBIDDEN": [1, 2],  # column that should be dropped
        "geometry": polys,
    }
    if duplicate:
        data["CSDUID"] = ["001", "001"]
    if missing_uid:
        data["CSDUID"] = [None, "002"]
    return gpd.GeoDataFrame(data, crs="EPSG:4326")

class TestPrepareCsdBoundaryAsset(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for input shapefile and output geojson
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.input_shp = self.tmp_path / "test.shp"
        self.output_geojson = self.tmp_path / "out.geojson"

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_successful_preparation(self):
        gdf = make_synthetic_gdf()
        gdf.to_file(self.input_shp)
        prepare_csd_boundary_asset(
            input_shp=self.input_shp,
            output_geojson=self.output_geojson,
            expected_rows=2,
        )
        out_gdf = gpd.read_file(self.output_geojson)
        expected_cols = {"csd_uid", "csd_name", "province_uid", "province_name", "geometry"}
        self.assertEqual(set(out_gdf.columns), expected_cols)
        self.assertTrue(out_gdf["csd_uid"].apply(lambda x: isinstance(x, str)).all())

    def test_duplicate_csd_uid_raises(self):
        gdf = make_synthetic_gdf(duplicate=True)
        gdf.to_file(self.input_shp)
        with self.assertRaises(SystemExit) as cm:
            prepare_csd_boundary_asset(
                input_shp=self.input_shp,
                output_geojson=self.output_geojson,
                expected_rows=2,
            )
        self.assertIn("Duplicate csd_uid", str(cm.exception))

    def test_missing_csd_uid_raises(self):
        gdf = make_synthetic_gdf(missing_uid=True)
        gdf.to_file(self.input_shp)
        with self.assertRaises(SystemExit) as cm:
            prepare_csd_boundary_asset(
                input_shp=self.input_shp,
                output_geojson=self.output_geojson,
                expected_rows=2,
            )
        self.assertIn("Missing csd_uid", str(cm.exception))

    def test_forbidden_columns_dropped(self):
        # The function always drops columns not in the whitelist, so just ensure no error.
        gdf = make_synthetic_gdf()
        gdf.to_file(self.input_shp)
        prepare_csd_boundary_asset(
            input_shp=self.input_shp,
            output_geojson=self.output_geojson,
            expected_rows=2,
        )
        out_gdf = gpd.read_file(self.output_geojson)
        self.assertNotIn("FORBIDDEN", out_gdf.columns)

    def test_missing_source_columns_raises(self):
        """Missing required source columns should cause SystemExit."""
        # Create a GeoDataFrame missing the PRUID column
        polys = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
        data = {
            "CSDUID": ["001"],
            "CSDNAME": ["A"],
            # "PRUID" omitted intentionally
            "PRNAME": ["ProvinceA"],
            "FORBIDDEN": [1],
            "geometry": polys,
        }
        gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
        gdf.to_file(self.input_shp)
        with self.assertRaises(SystemExit) as cm:
            prepare_csd_boundary_asset(
                input_shp=self.input_shp,
                output_geojson=self.output_geojson,
                expected_rows=1,
            )
        self.assertIn("Missing required source columns", str(cm.exception))

    def test_missing_optional_prname_succeeds(self):
        """When PRNAME is absent, script should still succeed and create empty province_name."""
        polys = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
        data = {
            "CSDUID": ["001"],
            "CSDNAME": ["A"],
            "PRUID": ["01"],
            # PRNAME intentionally omitted
            "FORBIDDEN": [1],
            "geometry": polys,
        }
        gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
        gdf.to_file(self.input_shp)
        prepare_csd_boundary_asset(
            input_shp=self.input_shp,
            output_geojson=self.output_geojson,
            expected_rows=1,
        )
        out_gdf = gpd.read_file(self.output_geojson)
        self.assertIn("province_name", out_gdf.columns)
        self.assertEqual(out_gdf["province_name"].iloc[0], "")
        # Verify required columns are present
        expected_cols = {"csd_uid", "csd_name", "province_uid", "province_name", "geometry"}
        self.assertEqual(set(out_gdf.columns), expected_cols)



