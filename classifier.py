"""Lightweight CPU-side classifier head.

The head pairs with a frozen feature-extractor backbone (`models/vibration_backbone*.tflite`):
the backbone turns a (WINDOW, 3) raw window into a 128-d float embedding; the head
classifies that embedding by cosine similarity against per-class prototypes.

Persisted as JSON next to the backbone — `classifier_head.json`:

    {
      "labels": ["circle", "shake", "steady"],
      "prototypes": [[...128 floats...], ...]
    }

"Training" = computing one mean embedding per label. Done by `trainer.py`.
"""

import json
import os
import threading

import numpy as np


DEFAULT_HEAD_PATH = "classifier_head.json"
FEATURE_DIM = 128
PALETTE_SIZE = 10           # number of distinct UI colour slots; trainer shuffles into this


class ClassifierHead:
    """Class-prototype + cosine-similarity classifier. No TF, pure numpy.

    Each label also gets a colour slot (0..PALETTE_SIZE-1) that the dashboard
    uses to style the label. Slots are re-assigned by `trainer.py` on every
    train so the same label can change colour between runs — by design,
    so you can visually tell when the model has been retrained.
    """

    def __init__(self, labels, prototypes, colors=None):
        labels = list(labels)
        prototypes = np.asarray(prototypes, dtype=np.float32)
        if prototypes.ndim != 2 or prototypes.shape[0] != len(labels):
            raise ValueError(
                f"prototypes shape {prototypes.shape} doesn't match {len(labels)} labels"
            )
        if colors is None:
            # Stable fallback for older head.json files: assign 0,1,2,... mod palette.
            colors = [i % PALETTE_SIZE for i in range(len(labels))]
        else:
            colors = [int(c) % PALETTE_SIZE for c in colors]
            if len(colors) != len(labels):
                raise ValueError(
                    f"colors length {len(colors)} doesn't match {len(labels)} labels"
                )
        self.labels = labels
        self.colors = colors
        self.prototypes = prototypes
        # Cache L2-normalised prototypes once so predict() is one dot product.
        norms = np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-9
        self._prototypes_normed = (prototypes / norms).astype(np.float32)
        self._lock = threading.Lock()   # used by InferenceWorker.reload_head swaps

    def label_color_map(self):
        """Dict {label_name: color_slot} — convenient for the snapshot API."""
        return dict(zip(self.labels, self.colors))

    @classmethod
    def load(cls, path=DEFAULT_HEAD_PATH):
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                blob = json.load(f)
            labels = blob["labels"]
            prototypes = np.array(blob["prototypes"], dtype=np.float32)
            colors = blob.get("colors")
            return cls(labels, prototypes, colors=colors)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"[classifier] failed to load {path}: {e}")
            return None

    def save(self, path=DEFAULT_HEAD_PATH):
        blob = {
            "labels": list(self.labels),
            "colors": list(self.colors),
            "prototypes": self.prototypes.tolist(),
        }
        # Write to a temp file + rename so a partial write can't corrupt the
        # head that the running InferenceWorker is about to reload.
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(blob, f, indent=2)
        os.replace(tmp, path)

    def predict(self, feat):
        """feat: (128,) float32. Returns (class_id, cosine_sim_in_[-1,1])."""
        f = np.asarray(feat, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(f) + 1e-9
        fn = f / n
        sims = self._prototypes_normed @ fn          # (N_labels,)
        i = int(np.argmax(sims))
        return i, float(sims[i])
