"""
tests/test_models.py
====================
Unit tests for app/models (ModelFactory, CNN, RNN, LSTM, PretrainedCNN).
"""

import unittest
from app.models import (
    ModelFactory,
    ConvBlock,
    SimpleCNN,
    FlexibleCNN,
    PretrainedCNN,
    SimpleRNN,
    SimpleLSTM,
)


class TestModelFactory(unittest.TestCase):
    """Test suite for ModelFactory and classical ML models."""

    def test_list_supported_models(self):
        models = ModelFactory.list_supported_models()
        self.assertIn("classification", models)
        self.assertIn("random_forest", models["classification"])
        self.assertIn("cnn", models["classification"])

    def test_create_classical_classifiers(self):
        rf = ModelFactory.create("classification", "random_forest", n_estimators=10)
        self.assertEqual(rf.n_estimators, 10)

        lr = ModelFactory.create("classification", "logistic_regression")
        self.assertIsNotNone(lr)

        svm = ModelFactory.create("classification", "svm")
        self.assertIsNotNone(svm)

    def test_create_classical_regressors(self):
        ridge = ModelFactory.create("regression", "ridge", alpha=0.5)
        self.assertEqual(ridge.alpha, 0.5)

    def test_create_classical_clustering(self):
        kmeans = ModelFactory.create("clustering", "kmeans", n_clusters=4)
        self.assertEqual(kmeans.n_clusters, 4)

    def test_unsupported_model_raises(self):
        with self.assertRaises(ValueError):
            ModelFactory.create("classification", "non_existent_model")


class TestCNNArchitectures(unittest.TestCase):
    """Test suite for PyTorch CNN architectures."""

    def test_simple_cnn_instantiation(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed in the environment.")

        model = SimpleCNN(in_channels=3, num_classes=5)
        self.assertEqual(model.in_channels, 3)
        self.assertEqual(model.num_classes, 5)

        # Test forward pass with dummy tensor (batch_size=2, C=3, H=32, W=32)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (2, 5))

        # Test parameter count
        params = model.count_parameters()
        self.assertIn("trainable", params)
        self.assertGreater(params["trainable"], 0)

    def test_flexible_cnn_instantiation(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed in the environment.")

        model = FlexibleCNN(
            in_channels=1,
            num_classes=3,
            channel_list=[16, 32],
            fc_dims=[64],
            activation="leaky_relu",
            dropout_rate=0.1,
        )
        self.assertEqual(model.in_channels, 1)

        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
