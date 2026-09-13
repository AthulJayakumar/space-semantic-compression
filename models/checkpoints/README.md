# Model Checkpoints

Large model checkpoint files are not stored in this public repository.

Expected default checkpoint:

```text
checkpoints/vqvae_s16k8.pt
```

or set:

```bash
COMPRESSAI_CHECKPOINT_PATH=path/to/checkpoint.pt
```

The code keeps checkpoint loading separate so the repository can remain lightweight for supervisors, reviewers, and GitHub visitors.
