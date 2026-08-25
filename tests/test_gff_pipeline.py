import struct
import unittest

import numpy as np
import torch

from src.gff_data import (
    WEATHER_CHANNELS,
    _gpkg_geometry_bounds,
    mercator_bounds_to_lonlat,
    sunet_clahe_sar,
    sunet_sar_variant,
)
from src.gff_model import GFFHorizonFormer, GFFViTHorizonFormer
from src.train_gff import (
    balanced_sampler,
    heatmap_confidence,
    normalize_probability_maps,
    objective,
    postprocess_probability_maps,
)


class GFFPipelineTests(unittest.TestCase):
    def test_probability_maps_are_normalized_independently_over_valid_pixels(self):
        probability = torch.tensor(
            [
                [[[2.0, 3.0], [4.0, 9.0]]],
                [[[0.4, 0.4], [0.4, 0.4]]],
            ]
        )
        valid = torch.tensor(
            [
                [[[1, 1], [1, 0]]],
                [[[1, 1], [1, 1]]],
            ],
            dtype=torch.bool,
        )
        normalized = normalize_probability_maps(
            probability, valid, "per_heatmap_minmax"
        )
        expected = torch.tensor(
            [
                [[[0.0, 0.5], [1.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
            ]
        )
        torch.testing.assert_close(normalized, expected)
        self.assertIs(normalize_probability_maps(probability), probability)
        with self.assertRaises(ValueError):
            normalize_probability_maps(probability, valid, "unknown")

    def test_only_low_confidence_heatmaps_are_normalized(self):
        probability = torch.tensor(
            [
                [[[0.01, 0.02], [0.03, 0.04]]],
                [[[0.10, 0.20], [0.60, 0.80]]],
            ]
        )
        valid = torch.ones_like(probability, dtype=torch.bool)
        settings = {
            "probability_normalization": "adaptive_low_confidence_minmax",
            "confidence_quantile": 1.0,
            "confidence_gate": 0.1,
            "raw_threshold": 0.5,
            "normalized_threshold": 0.75,
        }
        result = postprocess_probability_maps(
            probability, valid, 0.5, settings
        )
        self.assertEqual(result["normalized"].flatten().tolist(), [True, False])
        torch.testing.assert_close(
            result["probability"][0],
            torch.tensor([[[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]]]),
        )
        torch.testing.assert_close(result["probability"][1], probability[1])
        self.assertEqual(result["threshold"].flatten().tolist(), [0.75, 0.5])
        self.assertEqual(
            result["prediction"].flatten().tolist(),
            [False, False, False, True, False, False, True, True],
        )
        torch.testing.assert_close(
            heatmap_confidence(probability, valid, 1.0).flatten(),
            torch.tensor([0.04, 0.8]),
        )
        with self.assertRaises(ValueError):
            heatmap_confidence(probability, valid, 1.1)
        with self.assertRaises(ValueError):
            postprocess_probability_maps(
                probability,
                valid,
                postprocessing={
                    **settings,
                    "confidence_statistic": "mean",
                },
            )

    def test_balanced_sampler_groups_all_horizons_per_tile(self):
        class DummyDataset:
            tiles = list(range(4))
            horizons = (1, 2, 3)
            tile_positive_flags = [True, False, True, False]

            def __len__(self):
                return len(self.tiles) * len(self.horizons)

        sampler = balanced_sampler(DummyDataset(), 0.5, seed=7)
        indexes = list(sampler)
        self.assertEqual(len(indexes), 12)
        for start in range(0, len(indexes), 3):
            group = indexes[start : start + 3]
            self.assertEqual(len({index // 3 for index in group}), 1)
            self.assertEqual({index % 3 for index in group}, {0, 1, 2})

    def test_gpkg_envelope_and_mercator_conversion(self):
        # GP, version 0, little-endian + XY envelope, EPSG:3857, then envelope.
        blob = b"GP" + bytes([0, 3]) + struct.pack("<i4d", 3857, 0.0, 2240.0, 0.0, 2240.0)
        self.assertEqual(_gpkg_geometry_bounds(blob), (0.0, 0.0, 2240.0, 2240.0))
        lonlat = mercator_bounds_to_lonlat((0.0, 0.0, 2240.0, 2240.0))
        self.assertAlmostEqual(lonlat[0], 0.0)
        self.assertAlmostEqual(lonlat[1], 0.0)
        self.assertGreater(lonlat[2], 0.0)
        self.assertGreater(lonlat[3], 0.0)

    def test_horizonformer_shapes_and_finite_loss(self):
        model = GFFHorizonFormer(
            weather_channels=WEATHER_CHANNELS,
            decoder_dim=64,
            temporal_dim=64,
            temporal_depth=1,
        )
        batch_size = 2
        batch = {
            "spatial": torch.randn(batch_size, 4, 64, 64),
            "weather": torch.randn(batch_size, 20, WEATHER_CHANNELS, 8, 8),
            "horizon": torch.tensor([1, 3]),
            "forecast_mask": torch.tensor(
                [[False] * 19 + [True], [False] * 17 + [True] * 3]
            ),
            "target": torch.zeros(batch_size, 1, 64, 64),
            "valid": torch.ones(batch_size, 1, 64, 64),
            "boundary": torch.zeros(batch_size, 1, 64, 64),
            "presence": torch.tensor([0.0, 1.0]),
        }
        batch["target"][1, :, 20:40, 18:42] = 1.0
        batch["boundary"][1, :, 18:42, 16:44] = 1.0
        outputs = model(
            batch["spatial"],
            batch["weather"],
            batch["horizon"],
            batch["forecast_mask"],
        )
        self.assertEqual(tuple(outputs["segmentation"].shape), (2, 1, 64, 64))
        self.assertEqual(tuple(outputs["boundary"].shape), (2, 1, 64, 64))
        self.assertEqual(tuple(outputs["presence"].shape), (2,))
        loss = objective(outputs, batch, {"pos_weight": 4.0})
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_sunet_clahe_returns_original_and_enhanced_polarisations(self):
        gradient = np.linspace(1e-4, 1.0, 224 * 224, dtype=np.float32).reshape(224, 224)
        values = np.stack([gradient, gradient[::-1] * 0.2])
        result = sunet_clahe_sar(values)
        self.assertEqual(result.shape, (4, 224, 224))
        self.assertTrue(np.isfinite(result).all())
        self.assertGreaterEqual(float(result.min()), -1.0)
        self.assertLessEqual(float(result.max()), 1.0)
        self.assertFalse(np.allclose(result[0], result[2]))
        self.assertFalse(np.allclose(result[1], result[3]))

    def test_sunet_preprocessing_ablation_variants(self):
        gradient = np.linspace(1e-4, 1.0, 224 * 224, dtype=np.float32).reshape(224, 224)
        values = np.stack([gradient, gradient[::-1] * 0.2])
        db = sunet_sar_variant(values, "sunet_db")
        clahe = sunet_sar_variant(values, "sunet_clahe_only")
        dual = sunet_sar_variant(values, "sunet_clahe")
        self.assertEqual(db.shape, (2, 224, 224))
        self.assertEqual(clahe.shape, (2, 224, 224))
        self.assertEqual(dual.shape, (4, 224, 224))
        np.testing.assert_allclose(db, dual[:2])
        np.testing.assert_allclose(clahe, dual[2:])
        with self.assertRaises(ValueError):
            sunet_sar_variant(values, "unknown")

    def test_vit_horizonformer_output_shapes(self):
        model = GFFViTHorizonFormer(
            spatial_channels=6,
            weather_channels=WEATHER_CHANNELS,
            temporal_dim=64,
            temporal_depth=1,
            pretrained=False,
            freeze_blocks=12,
        ).eval()
        with torch.no_grad():
            outputs = model(
                torch.randn(1, 6, 224, 224),
                torch.randn(1, 20, WEATHER_CHANNELS, 8, 8),
                torch.tensor([3]),
                torch.tensor([[False] * 17 + [True] * 3]),
            )
        self.assertEqual(tuple(outputs["segmentation"].shape), (1, 1, 224, 224))
        self.assertEqual(tuple(outputs["boundary"].shape), (1, 1, 224, 224))
        self.assertEqual(tuple(outputs["presence"].shape), (1,))


if __name__ == "__main__":
    unittest.main()
