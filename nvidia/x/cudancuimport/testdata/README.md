# Nsight Compute details fixture

`transpose-details.csv` is an unmodified offline export of NVIDIA's bundled
`sharedBankConflicts/transposeNoBankConflicts.ncu-rep` sample. The report was
not profiled or recollected for this lane.

- Exporter: NVIDIA Nsight Compute 2026.3.0.0, build 38525999, public-release.
- Original report SHA256:
  `544b42110730bd2adf2325eeea44a69be6455767443fe1d7a1572bbd7183bb29`.
- CSV SHA256:
  `57ba1eddd8918134381f4db76672ffd16bc4b1e3520a432f42853f16d0e05411`.
- CSV: 17,427 bytes, 20 header columns, 65 data rows, 58 metric rows and
  seven rule rows. Metric rows omit the final four empty rule fields.

The producer exported the sample on 2026-10-02 using:

```sh
/opt/nvidia/nsight-compute/2026.3.0/ncu \
  --import /opt/nvidia/nsight-compute/2026.3.0/extras/samples/sharedBankConflicts/transposeNoBankConflicts.ncu-rep \
  --page details --csv --print-units base --print-metric-name name \
  --print-kernel-base mangled --print-summary none \
  --log-file "$HOME/tmp/lib-wt-next/20261002-ncu-export/transpose-details.csv"
```

This identifies the exporter, not the version that originally collected the
bundled report. NVIDIA documents these export options in its
[Nsight Compute CLI manual](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html).

The CSV has no artifact digest, compiler cache key, workload key or schema
version marker. The importer therefore wraps it in an explicitly project-owned
versioned envelope and leaves its kernels unjoined. Tests with synthetic launch
bindings are separately labeled; they are not metadata from this sample.
No `.ncu-rep` binary is shipped or parsed by this package.
