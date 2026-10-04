"""Small correctness checks that do not require DeepWeeds or a GPU."""
import unittest
import io
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

import benchmark
import dataset
import inference
import losses
import model
import train
import workflow


class PipelineChecks(unittest.TestCase):
    def test_load_split_accepts_official_two_column_csv(self):
        frame = pd.DataFrame({"Filename": ["example.jpg"], "Label": [0]})
        with patch.object(dataset.pd, "read_csv", return_value=frame) as read_csv:
            splits = dataset.load_split("labels", fold=0)
        self.assertEqual(len(splits), 3)
        self.assertTrue(all(list(split.columns) == ["Filename", "Label"] for split in splits))
        self.assertEqual(read_csv.call_count, 3)

    def test_focal_gamma_zero_matches_ce(self):
        torch.manual_seed(0)
        logits = torch.randn(7, 9)
        labels = torch.randint(0, 9, (7,))
        self.assertAlmostEqual(float(losses.FocalLoss(gamma=0)(logits, labels)),
                               float(F.cross_entropy(logits, labels)), places=6)

    def test_cutmix_lambda_uses_actual_replaced_area(self):
        np.random.seed(1)
        torch.manual_seed(1)
        x = torch.arange(4, dtype=torch.float32)[:, None, None, None].expand(4, 1, 20, 20).clone()
        y = torch.arange(4)
        mixed, (_, shuffled, lam) = losses.mix_batch(x, y, mode="cutmix")
        for i in range(4):
            if shuffled[i] != y[i]:
                area = (mixed[i] != x[i]).float().mean().item()
                self.assertAlmostEqual(lam, 1 - area, places=6)

    def test_fuse_conv_bn_preserves_eval_output(self):
        torch.manual_seed(0)
        net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4)).eval()
        x = torch.randn(2, 3, 8, 8)
        with torch.inference_mode():
            before = net(x)
            after = inference.fuse_conv_bn(net)(x)
        self.assertLess(float((before - after).abs().max()), 1e-5)

    def test_param_groups_exclude_frozen_and_decay_bias(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(3, 4)
                self.head = nn.Linear(4, 9)
            def get_classifier(self):
                return self.head
        net = Tiny()
        groups = model.param_groups(net, 1e-4, 1e-3, 0.05)
        self.assertEqual([len(group["params"]) for group in groups], [1, 1, 2])
        self.assertEqual([group["weight_decay"] for group in groups], [0.05, 0.0, 0.05])
        model.freeze_backbone(net)
        self.assertEqual(len(model.param_groups(net, 1e-4, 1e-3, 0.05)), 1)

    def test_parse_overrides_and_probabilities(self):
        parsed = train.parse_overrides(["seed=3", "amp=false", "ema_decay=0.99", "mix=none"])
        self.assertEqual(parsed, {"seed": 3, "amp": False, "ema_decay": 0.99, "mix": None})
        probs = inference.aggregate_views([np.zeros((2, 9)), np.ones((2, 9))])
        np.testing.assert_allclose(probs.sum(axis=1), 1)

    def test_official_split_audit(self):
        names = [f"image{i}.jpg" for i in range(17509)]
        frame = pd.DataFrame({"Filename": names, "Label": np.arange(17509) % 9})
        a, b, c = frame.iloc[:10505], frame.iloc[10505:14007], frame.iloc[14007:]
        with patch.object(Path, "is_file", return_value=True):
            with redirect_stdout(io.StringIO()):
                result = dataset.check_split(a, b, c, "unused")
            self.assertEqual(result["total"], 17509)
            with self.assertRaisesRegex(ValueError, "overlap"):
                dataset.check_split(a, b, pd.concat([c.iloc[:-1], a.iloc[:1]]), "unused")

    def test_train_epoch_and_evaluate(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(3 * 4 * 4, 9)
            def forward(self, x):
                return self.head(x.flatten(1))
            def get_classifier(self):
                return self.head
        net = Tiny()
        cfg = train.Config(epochs=1, batch_size=4, amp=False)
        loader = [(torch.randn(4, 3, 4, 4), torch.tensor([0, 1, 2, 3]),
                   [f"{i}.jpg" for i in range(4)])]
        optimizer = train.build_optimizer(net, cfg)
        scheduler = train.build_scheduler(optimizer, cfg, 1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        criterion = nn.CrossEntropyLoss()
        result = train.train_one_epoch(net, loader, criterion, optimizer, scheduler,
                                       scaler, cfg, torch.device("cpu"))
        self.assertTrue(np.isfinite(result["train_loss"]))
        filenames, labels, logits, loss = train.evaluate(net, loader, criterion, torch.device("cpu"))
        self.assertEqual(filenames, [f"{i}.jpg" for i in range(4)])
        self.assertEqual(logits.shape, (4, 9))
        self.assertTrue(np.isfinite(loss))

    def test_inference_methods_are_normalized(self):
        class PoolHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(3, 9)
            def forward(self, x):
                return self.head(x.mean(dim=(2, 3)))
        net = PoolHead().eval()
        images = torch.randn(3, 3, 224, 224)
        with torch.inference_mode():
            for method in ("I00", "I01", "I02", "I03"):
                probs = inference.forward_probs(net, images, method)
                self.assertEqual(tuple(probs.shape), (3, 9))
                torch.testing.assert_close(probs.sum(dim=1), torch.ones(3), atol=1e-6, rtol=0)
        loader = [(images, torch.tensor([0, 1, 2]), ["a.jpg", "b.jpg", "c.jpg"])]
        names, labels, probs = inference.predict_probs(net, loader, "cpu", "I01")
        self.assertEqual(names, ["a.jpg", "b.jpg", "c.jpg"])
        self.assertEqual(labels.tolist(), [0, 1, 2])
        np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-6)

    def test_ablation_changes_one_factor_and_includes_combination(self):
        def fake_run(cfg):
            score = {"T00": 0.5, "T01": 0.4, "T02": 0.3,
                     "T03": 0.55, "T04": 0.56, "T05": 0.53,
                     "T06": 0.51, "T07": 0.57}[cfg.exp_id]
            return {"macro_f1_val": score, "top1_val": score + 0.1}
        with patch.object(workflow, "run_cached", side_effect=fake_run), \
             patch.object(workflow, "_val_class_metrics", return_value={"f1": np.ones(9)}), \
             patch.object(workflow, "write_sheet"):
            frame = workflow.run_ablation("resnet50", "images", "labels")
        self.assertEqual(len(frame), 8)
        self.assertEqual(set(frame["axis"]), {"baseline", "A_init", "B_aug", "C_loss", "B_plus_C"})
        self.assertAlmostEqual(frame.loc[frame.exp_id == "T07", "delta_vs_T00"].iloc[0], 0.07)


if __name__ == "__main__":
    unittest.main()
