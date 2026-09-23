"""Versioned public-release metadata used by the API and audit scripts."""

from pathlib import Path


PROJECT_VERSION = "0.2.0"
MODEL_ID = "vqvae-ecofirebias-adapted-2026-09"
MODEL_STATUS = "experimental_no_go_for_sealed_test"
DEFAULT_CHECKPOINT_PATH = Path("models/checkpoints/vqvae_ecofirebias_train_adapted.pt")
DEFAULT_CHECKPOINT_SHA256 = "668cf5bda5c69b196d70306dbc6ed576c2675d372ffcfe2605e53a40286011f6"
