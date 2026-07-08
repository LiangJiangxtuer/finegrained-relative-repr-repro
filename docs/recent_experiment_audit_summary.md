# Recent Experiment Audit and GitHub Sync Summary

Updated: 2026-07-07

This audit summarizes the recent PAL reproduction experiment block before GitHub synchronization. All numeric claims below were read from local JSON artifacts under ignored `outputs/` paths; generated outputs/checkpoints remain excluded by `.gitignore`, while source code, tests, runners, and Markdown analyses are intended for commit.

## Scope audited

- ADE20K class-name / clean-alias cleanup and `--ignore-zero` handling.
- Frequent-class ADE20K error analysis for `wall/building/sky/floor/tree/person/road`.
- Dense-token/layer recovery for ADE20K (`last_hidden_state` vs explicit hidden-state indices).
- VOC20 / Pascal Context sanity check for the recovered ADE20K dense-token protocol.
- Targeted ADE20K class/group calibration probes and full confirmation.
- Downstream ablation summaries for retrieval, classification, and corrected segmentation.

## Code changes reviewed

- `src/pal_repro/segmentation.py`: conservative ADE20K clean aliases, alias-cleaning helper, and diagnostic dense-logit calibration helpers.
- `scripts/evaluate_segmentation.py`: explicit dense-token/layer controls, image-size probe flag, ADE20K class-prior diagnostics, manual `--class-bias`, logit-calibration metadata, per-class count outputs, and `--ignore-zero` support.
- `src/pal_repro/pipeline.py`: future ADE20K full segmentation commands default to `--alias-policy clean --ignore-zero`.
- New runners:
  - `scripts/run_classification_ablation_eval.py`
  - `scripts/run_classification_ablation_fast.py`
  - `scripts/run_segmentation_ablation_probes_fast.py`
  - `scripts/run_ade20k_dense_protocol_recovery.py`
  - `scripts/run_ade20k_group_calibration_probe.py`
  - `scripts/analyze_ade20k_segmentation_errors.py`
- Tests updated in `tests/test_segmentation_support.py` and `tests/test_pipeline.py` to cover parser flags, manual class bias, calibration helpers, clean aliases, and ADE20K pipeline defaults.

## Result artifacts audited

| Area | Machine artifact | Commit policy |
|---|---|---|
| Retrieval ablations | `outputs/ablations/retrieval/summary.json` | ignored output, cited in docs |
| Classification ablations | `outputs/ablations/classification_fast/summary.json` | ignored output, cited in docs |
| Corrected segmentation probes/full selected rows | `outputs/ablations/segmentation_probes_fast/summary.json`, `outputs/ablations/segmentation_full_selected/*.json` | ignored output, cited in docs |
| ADE20K clean aliases full rows | `outputs/ablations/segmentation_full_ade20k_clean_ignore0_all_variants/summary.json` | ignored output, cited in docs |
| ADE20K dense-token recovery | `outputs/diagnostics/ade20k_dense_protocol_recovery_full/summary.json` | ignored output, cited in docs |
| VOC20/Context sanity | `outputs/diagnostics/dense_protocol_sanity_voc_context_last_hidden/summary.json` | ignored output, cited in docs |
| ADE20K group calibration | `outputs/diagnostics/ade20k_group_calibration_limit64/summary.json`, `outputs/diagnostics/ade20k_group_calibration_full/summary.json` | ignored output, cited in docs |

`.gitignore` excludes `outputs/`, `data/tokens/`, `data/datasets/`, `data/splits/`, `*.pt`, and `*.pth`; no large experiment artifacts are intended for commit.

## Key metrics from audited artifacts

### Downstream ablations

| Task | Best / selected row | Metric | Paper target / reference | Relative |
|---|---|---:|---:|---:|
| Retrieval ablation | `pool_temperature:0_02` | Avg R@1 51.41 | 58.80 | 87.43% |
| Classification ablation | `tau_0_02` | Avg top-1 46.48 | 51.46 | 90.33% |
| Selected segmentation, recovered dense tokens | `tau_p=0.07` + ADE20K clean aliases / `--ignore-zero` / `last_hidden_state` | Avg mIoU 23.38 | 23.87 | 97.94% |
| Diagnostic selected segmentation, targeted ADE20K group bias | `wall,building,sky,floor,tree,person,road=0.04` | Avg mIoU 23.68 | 23.87 | 99.23% |

### ADE20K clean aliases and dense-token recovery

- Best full clean-alias ADE20K row: `tau/0_07` with mIoU `9.334` at `outputs/ablations/segmentation_full_ade20k_clean_ignore0_all_variants/tau/0_07/ade20k_full_segmentation.json`.
- Dense-token recovery full confirmation:

| Probe | ADE20K mIoU | Output |
|---|---:|---|
| `last_hidden_state_vlast_tlast` | 10.549 | `outputs/diagnostics/ade20k_dense_protocol_recovery_full/last_hidden_state_vlast_tlast.json` |
| `last_vision_hidden_text_t-2` | 10.229 | `outputs/diagnostics/ade20k_dense_protocol_recovery_full/last_vision_hidden_text_t-2.json` |

### VOC20 / Context sanity for ADE20K recovered `last_hidden_state`

Conclusion: do not globally replace VOC20/Context selected rows with `last_hidden_state`; it hurts VOC20 and slightly hurts Context.

| Dataset | Previous selected mIoU | `last_hidden_state` mIoU | Delta | Output |
|---|---:|---:|---:|---|
| VOC20 | 37.57 | 33.57 | -4.01 | `outputs/diagnostics/dense_protocol_sanity_voc_context_last_hidden/voc20_last_hidden_full.json` |
| Context | 22.00 | 21.90 | -0.10 | `outputs/diagnostics/dense_protocol_sanity_voc_context_last_hidden/context_last_hidden_full.json` |

### Targeted ADE20K group calibration

64-sample probe leaders:

| Probe | mIoU | Delta vs no-bias | Class bias |
|---|---:|---:|---|
| `boost_frequent7_p004` | 4.658 | 0.596 | `wall,building,sky,floor,tree,person,road=0.04` |
| `boost_underpred4_p004` | 4.453 | 0.391 | `wall,sky,person,road=0.04` |
| `underpred_p002_spurious_m002` | 4.404 | 0.342 | `wall,sky,person,road=0.02, house,screen door,bookcase,skyscraper,swivel chair=-0.02` |
| `boost_frequent7_p002` | 4.390 | 0.329 | `wall,building,sky,floor,tree,person,road=0.02` |
| `boost_wall_sky_p004` | 4.340 | 0.278 | `wall,sky=0.04` |

Full confirmation leaders:

| Probe | ADE20K mIoU | Delta vs no-bias | Class bias | Output |
|---|---:|---:|---|---|
| `boost_frequent7_p004` | 11.470 | 0.920 | `wall,building,sky,floor,tree,person,road=0.04` | `outputs/diagnostics/ade20k_group_calibration_full/boost_frequent7_p004.json` |
| `boost_underpred4_p004` | 11.219 | 0.670 | `wall,sky,person,road=0.04` | `outputs/diagnostics/ade20k_group_calibration_full/boost_underpred4_p004.json` |
| `underpred_p002_spurious_m002` | 11.022 | 0.473 | `wall,sky,person,road=0.02, house,screen door,bookcase,skyscraper,swivel chair=-0.02` | `outputs/diagnostics/ade20k_group_calibration_full/underpred_p002_spurious_m002.json` |

Interpretation: targeted frequent-class bias improves ADE20K from `10.549` to `11.470`. This is useful diagnostic evidence, but the row is validation-informed because class group and magnitude were selected from ADE20K diagnostics. It should not be reported as an unqualified paper-protocol result unless a held-out calibration protocol is added.

## Documentation updated for GitHub

- `README.md`
- `REPRODUCTION_SUMMARY.md`
- `docs/results_snapshot.md`
- `docs/full_reproduction_status.md`
- `docs/pipeline_results_snapshot.md`
- `docs/continuation_handoff.md`
- `docs/segmentation_debug_notes.md`
- `docs/zh_experiment_design_results_analysis.md`
- `docs/ablation_downstream_retrieval_results.md`
- `docs/ablation_downstream_classification_segmentation_results.md`
- `docs/ade20k_dense_debug_results.md`
- `docs/ade20k_dense_protocol_recovery.md`
- `docs/ade20k_frequent_class_error_analysis.md`
- `docs/ade20k_group_calibration_results.md`

## Verification status

Final pre-commit code verification:

```text
PYTHONPATH=src /home/hnxxzy/miniconda3/envs/ovvs/bin/python -m unittest discover -s tests -v
Ran 63 tests in 0.206s
OK
```

Also verified with `py_compile` over changed scripts/modules, `git diff --check`, runner `--list` smoke checks, staged large-artifact guard, and staged secret/static scan. The generated `outputs/` JSONs remain ignored and are cited rather than committed.

## Remaining caveats

1. The diagnostic group-calibrated ADE20K row (`11.47`) is validation-informed; add held-out calibration if it is to become a main paper-protocol row.
2. Remaining full VOC20/Context checkpoint rows are optional unless full ablation-table parity is required.
3. Strict CKA layer-selection parity and external baseline rows remain separate reproduction extensions.
