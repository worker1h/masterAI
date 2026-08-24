import unittest
from pathlib import Path
import torch
from src.data import CHANNELS,ImpactMeshDataset
from src.losses import boundary_band, segmentation_loss
from src.metrics import binary_metrics
from src.model import DeepLabV3PlusMobileNet, HistoricalFloodForecastNet, HistoricalFloodResidualForecastNet, SegFormerB0, SiameseChangeUNet, SiameseChangeUNetPresenceRefine, SiameseChangeUNetWithPresence, SiameseSegFormerB0, UNet, build_model
from src.train import training_objective
from src.sampling import balanced_bin_weights, flood_fraction_bins
from scripts.evaluate_checkpoint import boundary_counts


class PipelineTests(unittest.TestCase):
    @unittest.skipUnless(Path("data/raw").exists(), "legacy ImpactMesh data was removed")
    def test_all_ablation_shapes(self):
        for mode,channels in CHANNELS.items():
            x,y,_=ImpactMeshDataset("data/raw","val",mode,limit=1)[0]
            self.assertEqual(tuple(x.shape),(channels,256,256)); self.assertEqual(tuple(y.shape),(1,256,256))
            self.assertEqual(tuple(UNet(channels,8)(x[None]).shape),(1,1,256,256))

    @unittest.skipUnless(Path("data/raw").exists(), "legacy ImpactMesh data was removed")
    def test_mask_is_binary_flood_only(self):
        _,y,_=ImpactMeshDataset("data/raw","val","e0",limit=1)[0]
        self.assertTrue(set(torch.unique(y).tolist()).issubset({0.0,1.0}))

    def test_metrics(self):
        m=binary_metrics(3,1,1,5); self.assertAlmostEqual(m["iou"],.6,places=6); self.assertAlmostEqual(m["dice"],.75,places=6)

    def test_boundary_loss_has_finite_gradient(self):
        target=torch.zeros(2,1,16,16); target[:,:,4:12,5:11]=1
        logits=torch.zeros_like(target,requires_grad=True)
        band=boundary_band(target,5)
        self.assertGreater(band.sum().item(),0); self.assertEqual(band.shape,target.shape)
        loss=segmentation_loss(logits,target,4.0,{"type":"focal_tversky","boundary_weight":2.0})
        loss.backward(); self.assertTrue(torch.isfinite(logits.grad).all())

    def test_fraction_bin_balancing(self):
        bins=flood_fraction_bins([0.0,0.001,0.02,0.2])
        self.assertEqual(bins.tolist(),[0,1,2,3])
        weights,counts,probs=balanced_bin_weights([0,0,1,2,3])
        self.assertEqual(counts,[2,1,1,1]); self.assertAlmostEqual(sum(probs),1.0)
        totals=[sum(w for w,b in zip(weights,[0,0,1,2,3]) if b==i) for i in range(4)]
        self.assertTrue(all(abs(v-.25)<1e-8 for v in totals))

    def test_boundary_metric_perfect_prediction(self):
        target=torch.zeros(1,1,16,16,dtype=torch.bool);target[:,:,4:12,4:12]=True
        mp,npred,mt,ntrue=boundary_counts(target,target,tolerance=2)
        self.assertEqual(mp,npred);self.assertEqual(mt,ntrue)

    def test_siamese_change_unet_shape_and_factory(self):
        model=build_model("siamese_change_unet",4,8)
        x=torch.randn(2,4,64,64)
        self.assertEqual(tuple(model(x).shape),(2,1,64,64))
        self.assertIsInstance(model,SiameseChangeUNet)

    def test_siamese_rejects_non_temporal_input(self):
        with self.assertRaises(ValueError):
            SiameseChangeUNet(2,8)

    def test_lightweight_model_shapes_and_factory(self):
        x=torch.randn(2,4,64,64)
        cases={
            "deeplabv3plus_mobilenet": DeepLabV3PlusMobileNet,
            "segformer_b0": SegFormerB0,
        }
        for name,expected_type in cases.items():
            with self.subTest(model=name):
                model=build_model(name,4,8)
                self.assertEqual(tuple(model(x).shape),(2,1,64,64))
                self.assertIsInstance(model,expected_type)

    def test_presence_model_auxiliary_loss_and_gate(self):
        model=build_model("siamese_change_unet_presence",4,8)
        x=torch.randn(2,4,64,64); y=torch.zeros(2,1,64,64); y[0,:,20:30,20:30]=1
        raw,presence=model.forward_with_aux(x)
        self.assertEqual(tuple(raw.shape),(2,1,64,64))
        self.assertEqual(tuple(presence.shape),(2,))
        self.assertTrue(torch.isfinite(training_objective(model,x,y,2.0,{"type":"bce_dice","presence_weight":.25})))
        self.assertIsInstance(model,SiameseChangeUNetWithPresence)

    def test_siamese_segformer_shape_and_factory(self):
        model=build_model("siamese_segformer_b0",4,8)
        self.assertEqual(tuple(model(torch.randn(2,4,64,64)).shape),(2,1,64,64))
        self.assertIsInstance(model,SiameseSegFormerB0)

    def test_presence_refine_auxiliary_outputs_and_loss(self):
        model=build_model("siamese_change_unet_presence_refine",4,8)
        x=torch.randn(2,4,64,64); y=torch.zeros(2,1,64,64); y[0,:,16:40,20:36]=1
        segmentation,presence,boundary=model.forward_with_aux(x)
        self.assertEqual(tuple(segmentation.shape),(2,1,64,64))
        self.assertEqual(tuple(presence.shape),(2,))
        self.assertEqual(tuple(boundary.shape),(2,1,64,64))
        config={"type":"bce_dice","presence_weight":.25,"boundary_aux_weight":.2}
        self.assertTrue(torch.isfinite(training_objective(model,x,y,2.0,config)))
        self.assertIsInstance(model,SiameseChangeUNetPresenceRefine)

    def test_historical_flood_forecast_model(self):
        model=build_model("historical_flood_forecast_net",4,8)
        output=model(torch.randn(2,4,64,64))
        self.assertEqual(tuple(output.shape),(2,1,64,64))
        self.assertIsInstance(model,HistoricalFloodForecastNet)

    def test_historical_flood_residual_forecast_starts_at_mapping_logits(self):
        model=build_model("historical_flood_residual_forecast_net",4,8)
        model.eval(); x=torch.randn(2,4,64,64)
        with torch.no_grad():
            mapped,_,_=SiameseChangeUNetPresenceRefine.forward_with_aux(model,x)
            forecast,_,_=model.forward_with_aux(x)
        self.assertTrue(torch.equal(mapped,forecast))
        self.assertIsInstance(model,HistoricalFloodResidualForecastNet)


if __name__=="__main__": unittest.main()
