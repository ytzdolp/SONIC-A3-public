# Release checklist

See [PUBLICATION_FOLLOWUP.md](PUBLICATION_FOLLOWUP.md) for unresolved owner/provider actions.
The working-tree checker does not certify rights, staged content, history, or server-side removal.

Before every push:

- [ ] `python -B tools/test_check_release.py` passes
- [ ] `python -B tools/check_release.py` passes
- [ ] local release rules remain ignored and are not force-added
- [ ] no credentials, tokens, private keys, internal hosts, or personal paths
- [ ] no files above GitHub's ordinary Git limit
- [ ] no virtual environments, logs, checkpoints, or datasets
- [ ] upstream `LICENSE` and attribution retained
- [ ] HuMoR and PyTorch3D notices and complete license texts retained
- [ ] A3 assets excluded unless redistribution permission is documented
- [ ] external-teacher permission is documented for each claimed use, or all
      affected claims/artifacts remain explicitly provisional and excluded
- [ ] README describes incomplete features honestly
- [ ] `108/108` is never reported without the four-second `106/108`, bounded
      three-second setup, unavailable public evidence, and sim-to-real limit
- [ ] installation commands tested in a clean environment
- [ ] current source files compile
- [ ] README/project progress match accepted checkpoints and current capability boundaries
- [ ] public and private current-result metrics agree; only publication detail differs
- [ ] no alignment dataset, BONES-SEED file, rollout NPZ, or local absolute path is staged
- [ ] no external walking-policy source, numerical contract, YAML, ONNX, CSV, or rollout is staged
- [ ] `git diff --cached --check` and the staged-file list have been reviewed

For the first clean public import, the unchanged upstream tree contains
pre-existing whitespace findings. Check the A3/project-authored paths
separately and record that baseline instead of mechanically rewriting upstream
license and documentation files. Require the full command for later commits.

Before changing the repository to public:

- [ ] review all Git history, not only the working tree
- [ ] remove restricted code and machine-specific rules from reachable history;
      separately check old-SHA access, hosting caches, and remaining redistribution rights
- [ ] confirm licenses for every robot asset, motion file, and checkpoint
- [ ] replace local paths with placeholders
- [ ] add a model/data card for every downloadable artifact
- [ ] verify the project does not imply vendor endorsement
- [ ] confirm the exact external walking contract remains absent from source,
      configs, docs, data, and Git history
- [ ] archive a private, reviewable record of the external teacher provider's
      permission scope before publishing teacher-derived artifacts or claims
