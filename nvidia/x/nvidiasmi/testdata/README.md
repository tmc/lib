# nvidia-smi -q -x fixtures

Each XML file is a byte-for-byte copy of a report published under the
MIT license. The license and copyright notice of each source is kept
alongside as LICENSE.<source>. Serials and UUIDs are as published; some
sources anonymized them before publishing (quadro-p4000-440.xml,
rtx-3080-595.xml).

| File | GPU | Driver | DTD | Source | License | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| a100-sxm4-mig-535.xml | A100-SXM4-80GB, MIG enabled | 535.54.03 | v12 | [telegraf a100-sxm4-v12.xml](https://github.com/influxdata/telegraf/blob/6f6d5b5be846be5f5c26804c9412ba72a8dc7a27/plugins/inputs/nvidia_smi/testdata/a100-sxm4-v12.xml) | MIT, LICENSE.telegraf | 6eeba22e13ac4203ae5ab71a6e72fc1ee1e585ed1ef8fbe6ab493a222e3b2b74 |
| gb200-570.xml | 4x GB200, NVLink fabric | 570.148.08 | v12 | [gpuid etc/gpus/gb200.xml](https://github.com/mchmarny/gpuid/blob/ab8b998509a8daac81a94df3e7b343b345b489ec/etc/gpus/gb200.xml) | MIT, LICENSE.gpuid | fd9ac804016260a9a705d2fe058c9c0a029679b42e05a0bd167aa0bc588bd1d8 |
| gtx-1660-ti-418.xml | GeForce GTX 1660 Ti, no DOCTYPE | 418.43 | none | [telegraf gtx-1660-ti.xml](https://github.com/influxdata/telegraf/blob/2a8735d1c6234f76c27564799c3e40235cc22d9d/plugins/inputs/nvidia_smi/testdata/gtx-1660-ti.xml) | MIT, LICENSE.telegraf | 8197a028fc54b06269079657a32ae62fcfb5f40ad2260dab7419b5d083b967cb |
| h100-570.xml | 8x H100 80GB HBM3 | 570.86.15 | v12 | [gpuid etc/gpus/h100.xml](https://github.com/mchmarny/gpuid/blob/ab8b998509a8daac81a94df3e7b343b345b489ec/etc/gpus/h100.xml) | MIT, LICENSE.gpuid | 03e75d3e4c478391bcbea94e9e91f7e32767db8f472b6d691cb845c08923b129 |
| quadro-p4000-440.xml | Quadro P4000 | 440.44 | v10 | [gpustat-unraid sample/nvidia-smi-stdout.txt](https://github.com/b3rs3rk/gpustat-unraid/blob/2792d58506c5b54201d9a3364d9093fd7f10ca27/sample/nvidia-smi-stdout.txt) | MIT, LICENSE.gpustat-unraid | f38e7704d0a4c6a3eae31aea7eb6e1b16c45e19547b1b5c367b2987c8e35a22f |
| rtx-3080-595.xml | GeForce RTX 3080 | 595.58.03 | v13 | [telegraf rtx-3080-v13.xml](https://github.com/influxdata/telegraf/blob/4fe83a4d1613810ccc464330c5cb1f3f1b5bacc9/plugins/inputs/nvidia_smi/testdata/rtx-3080-v13.xml) | MIT, LICENSE.telegraf | 9c920fedbae81e989262f11524ce3cf03985113cd90c5482b1976bc44ed914af |
| tesla-t4-515.xml | Tesla T4, two processes | 515.105.01 | v11 | [telegraf tesla-t4.xml](https://github.com/influxdata/telegraf/blob/cd535165d06debedae7b5caed1816f3a6275a6da/plugins/inputs/nvidia_smi/testdata/tesla-t4.xml) | MIT, LICENSE.telegraf | 777cc9289286d9c9dc4f14efb217c8d7dcbc59da78c0dde7ff9f6e1428ce0074 |

a100-sxm4-mig-535.xml reports attached_gpus 4 but holds one gpu element;
the source trimmed it.

To add a capture, save the output of `nvidia-smi -q -x` here as
GPU-DRIVER.xml, then add a row to `fixtures` in ../nvidiasmi_test.go with
values copied from the XML by hand. TestFixturesListed fails until the row
exists.

# nvidia-smi topo -m fixtures

The files in topo/ are `nvidia-smi topo -m` outputs published in
repositories under the MIT or Apache-2.0 license; each license is kept
here as LICENSE.<source>. None was captured for this package and none
is derived from NVIDIA documentation. Three are byte-for-byte copies of
a published file. Two are excerpts of a larger published file, cut at
the line numbers given and otherwise unchanged; nv6x4-numa.txt was
published with spaces where nvidia-smi writes tabs. The GPU models in
the names are as the sources state them. dgx1.txt's source names only
the system; the sources of nv2x4-nic.txt and nv6x4-numa.txt name no
GPU.

| File | System | Source | Copy | License | SHA-256 |
| --- | --- | --- | --- | --- | --- |
| dgx1.txt | DGX-1, 8 GPUs, NV1/NV2, 4 NICs named mlx5_N, CPU Affinity only | [gossip scripts/dgx1_topology.txt](https://github.com/Funatiq/gossip/blob/dca28648a77eb58507d3b5c7ccbbb14005113e55/scripts/dgx1_topology.txt) | whole file | MIT, LICENSE.gossip | f318241b820552f5318770835c2183cfc29abe84f24f211bcb5a6001f2d3c917 |
| h200x1.txt | 1x H200, driver 590.48.01, terminal escape codes | [dot-loom research/ministral-conductor/receipts/machine.txt](https://github.com/usedotai/dot-loom/blob/ad2d54efab464591c0cf800c0abb8dbc2631d921/research/ministral-conductor/receipts/machine.txt) | lines 7-18 | MIT, LICENSE.dot-loom | 9c1dc862e8c0c48801a0ae6bbdb91e9f9edabf355587bb275688ce5654188254 |
| h200x8-nic.txt | 8x H200, NV18, 9 NICs with NIC legend, 4 NUMA nodes | [warpcourier results/20260820-h200/topo.txt](https://github.com/VetleWammer2/warpcourier/blob/812502829c911eef49d5d673a556b67ca8ea808c/results/20260820-h200/topo.txt) | whole file | MIT, LICENSE.warpcourier | 9cfe4fd92753db45e8192d2030bb5ef280b3508a929d27abd4838fd2e1a29dd9 |
| nv2x4-nic.txt | 4 GPUs, NV2, 2 NICs behind one PCIe switch | [Magi micro_benchmarks/topo.out](https://github.com/ShAlireza/Magi/blob/babaef92891e18d945208c3e5861991e3dbbff3d/micro_benchmarks/topo.out) | whole file | MIT, LICENSE.magi | 2debe9a653be286e700981894eb0ec5ddf0021ccccc5029905ed63218deedd53 |
| nv6x4-numa.txt | 4 GPUs, NV6, 288 CPUs, GPU NUMA IDs | [hpc-ai-envs examples/torch-distributed/README.md](https://github.com/crickett-hpe/hpc-ai-envs/blob/540653ba27c74c2ad8947c5f39ccd871b8009c84/examples/torch-distributed/README.md) | lines 45-59 | Apache-2.0, LICENSE.hpc-ai-envs | fcaf22230577e8057861c2ec516f6c9eb6d4521227ffeb8d74e40c005676e5e5 |

dgx1.txt was committed in 2018, before gossip added its LICENSE file in
2019; it is pinned to a later commit that has both, at which its bytes
are unchanged.

To add a capture, save the output of `nvidia-smi topo -m` as
topo/NAME.txt and add a row to `topoFixtures` in ../topo_test.go.
TestTopologyFixturesListed fails until the row exists.
