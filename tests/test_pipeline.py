import unittest
import torch
from src.data import CHANNELS,ImpactMeshDataset
from src.losses import boundary_band, segmentation_loss
from src.metrics import binary_metrics
from src.model import UNet
from src.sampling import balanced_bin_weights, flood_fraction_bins
from scripts.evaluate_checkpoint import boundary_counts


class PipelineTests(unittest.TestCase):
    def test_all_ablation_shapes(self):
        for mode,channels in CHANNELS.items():
            x,y,_=ImpactMeshDataset("data/raw","val",mode,limit=1)[0]
            self.assertEqual(tuple(x.shape),(channels,256,256)); self.assertEqual(tuple(y.shape),(1,256,256))
            self.assertEqual(tuple(UNet(channels,8)(x[None]).shape),(1,1,256,256))

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


if __name__=="__main__": unittest.main()
