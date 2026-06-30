import unittest

class TestImport(unittest.TestCase):
    def test_package_import(self):
        """Verify that the hpai_gnn package can be imported without errors."""
        try:
            import hpai_gnn
            print("Successfully imported hpai_gnn.")
            self.assertTrue(hasattr(hpai_gnn, "__version__"))
            print(f"hpai_gnn package version is: {hpai_gnn.__version__}")
        except ImportError as e:
            self.fail(f"Failed to import hpai_gnn. Error: {e}")

if __name__ == "__main__":
    unittest.main()
