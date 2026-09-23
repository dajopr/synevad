"""Standalone defect generation: clean frame in, graded synthetic anomaly out.

Edits a clean frame with FLUX from a per-category prompt set, estimates the defect mask
that becomes pixel ground truth, composites it back, and logs *every* candidate —
accepted or not — so a run is reproducible from its manifests alone. Orchestration lives
in ``scripts/generate_standalone.py``; the pure building blocks live here, and the YAML
that drives them in ``synevad/synthesis/configs/``.

The eval half of the repo (``synevad.data``, ``synevad.eval``, ``synevad.metrics``) reads
the manifests this writes and never imports from here. Two imports go the other way,
both lazy: ``synevad.metrics.area`` measures each composite's mask as it is made, and
``synevad.backbones.DINO_MODELS`` keeps the mask estimator and the detector agreeing on
what a DINO backbone name means.
"""
