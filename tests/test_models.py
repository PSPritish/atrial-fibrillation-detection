import unittest
from models.model_factory import create_model
from config.default import get_default_config

class TestModelArchitectures(unittest.TestCase):

    def setUp(self):
        self.config = get_default_config()

    def test_resnet18(self):
        model = create_model('resnet18', self.config)
        self.assertIsNotNone(model)
        self.assertEqual(model.num_layers, 18)

    def test_resnet34(self):
        model = create_model('resnet34', self.config)
        self.assertIsNotNone(model)
        self.assertEqual(model.num_layers, 34)

    def test_resnet50(self):
        model = create_model('resnet50', self.config)
        self.assertIsNotNone(model)
        self.assertEqual(model.num_layers, 50)

    def test_complex_resnet50(self):
        model = create_model('complex_resnet50', self.config)
        self.assertIsNotNone(model)
        self.assertTrue(model.is_complex)

    def test_custom_architecture(self):
        model = create_model('custom_architecture', self.config)
        self.assertIsNotNone(model)
        self.assertEqual(model.custom_property, 'expected_value')

if __name__ == '__main__':
    unittest.main()