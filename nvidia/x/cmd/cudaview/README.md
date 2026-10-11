# cudaview

Open retained CUDA source and assembly files without a profile capture:

```
cudaview kernel.cu kernel.ptx kernel.sass
cudaview -html=report.html kernel.ptx kernel.sass
```

The viewer accepts one file per compilation tier: `.go`, `.cu` or `.cpp`,
`.tileir` or `.mlir`, `.ptx`, and `.sass`. It shares its renderer with
`cudaprof view`. Reading files does not invoke a compiler or disassembler.

Listings support search, next match, line/address navigation, instruction
selection, and copying line ranges. HTML exports embed their source data and
work offline. When clipboard access is unavailable, the selected range appears
in the instruction inspector for manual copying.

CUBIN loading, explicit enrichment, and the planned analysis summaries are not
implemented in this command yet.
