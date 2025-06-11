import unittest
from data.processed.transformations import gasf_transform, gadf_transform

class TestDataProcessing(unittest.TestCase):

    def test_gasf_transform(self):
        # Test the GASF transformation
        input_data = [...]  # Replace with appropriate test data
        expected_output = [...]  # Replace with expected output
        output = gasf_transform(input_data)
        self.assertEqual(output, expected_output)

    def test_gadf_transform(self):
        # Test the GADF transformation
        input_data = [...]  # Replace with appropriate test data
        expected_output = [...]  # Replace with expected output
        output = gadf_transform(input_data)
        self.assertEqual(output, expected_output)

if __name__ == '__main__':
    unittest.main()