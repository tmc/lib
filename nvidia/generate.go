package nvidia

// The packages beneath this directory are generated. Regenerating all of them
// is one command:
//
//	go generate
//
// That runs nvgen once per package, each invocation reading the CUDA headers
// and writing one directory. nvgen holds
// the recipe -- which generator runs with which arguments -- so this file
// lists only which packages exist and which of them the CUDA profile selects.
//
// nvgen lives in the generator repository, which this module does not depend
// on. Install it from a sibling checkout before generating:
//
//	(cd ../../libgen && go install ./cmd/nvgen)
//
// Choosing headers. By default nvgen reads the cuda-headers checkout beside
// this one, the tree upstream.json records. A materialized profile under
// cuda-headers-profiles selects another tree by name:
//
//	CUDA_PROFILE=cuda-12.6.3 go generate
//
// The packages in the first block below read that tree, so a toolkit bump
// moves them and only them. The second block passes no header tree at all:
// those inputs ship with their own NVIDIA SDK, which no CUDA version selects.
//
// Checking without writing. To report what a regeneration would change
// without touching the tree:
//
//	nvgen -diff -output .
//
// A clean -diff means every file nvgen owns matches its inputs. This is the
// check to run in CI; it is not run by go generate. nvgen -list prints all 35
// recipes, marking one whose gate is unset as skipped rather than omitting it.
//
// What nvgen does not cover. One package is generated entirely elsewhere:
// x/cudaptx carries its own go:generate directive, because its input is not
// the CUDA profile. See its package doc.
//
// Three more packages are generated from here but hold one file that is not.
// nvjitlink, nvrtc and nvvm each take flags.gen.go from genoptions, whose
// input is optiondata rather than the CUDA profile. nvgen neither writes nor
// diffs those three files, so -diff reports a clean package while one of them
// has drifted; regenerate with go generate in the package itself, or run
// TestFlagsGenUpToDate, which regenerates all three and compares.
//
// Today x is part of this module, so that regeneration needs nothing special.
// When x becomes its own module it will need the workspace: the command is then
// in another module, go.work is what lets this one reach it, and it must stay
// that way, because a require on x would put x in every consumer's graph.
//
// What the committed packages were generated from. The vendor archives named
// in upstream.json are the linux-x86_64 redistributables, and their versions
// differ per package rather than coming from one toolkit release. That is the
// platform of the INPUT headers. It is not the platform of the generation: the
// recorded toolchains are darwin/arm64 throughout, and three entries -- ncclep,
// ncclll and ncclm2n -- record no toolchain at all. So the committed API is
// derived from Linux headers by a generator that has only ever been run on
// Darwin, and upstream.json is the place that says so per package.
//
// Generated files are not edited. A file whose name ends in .gen.go or
// .gen_test.go is written by nvgen, and nvgen overwrites it without reading it,
// so an edit survives until the next regeneration and leaves no trace when it
// goes. A defect in a generated file is a defect in the generator: fix it in
// libgen and regenerate. That the fix is one character is not a reason to make
// an exception. The cost is not the edit; it is that the file and its inputs
// stop agreeing and nothing says so. The converse does not hold: x/tileir
// carries a generated file spelled _gen rather than .gen, and it is not
// nvgen's.
//
// What a regeneration has to clear. Regenerating moves an exported API, and the
// test suite is not the check for that: a constant whose value changes and a
// parameter that is renamed both compile and both pass. Compare the
// result against the last nvidia/v* tag with apidiff -m -incompatible and read
// what moved before committing.

// Eject the pinned local ARM64 registrar before generating its NCCL callers.
// FFI_SRC may name a clean checkout of the revision in scripts/ffi.sh.
//go:generate sh scripts/ffi.sh

//go:generate -command nvgen nvgen -output .

// Generated from the CUDA profile: a toolkit bump moves these.
//go:generate nvgen driver
//go:generate nvgen nvml
//go:generate nvgen nvrtc
//go:generate nvgen cublas
//go:generate nvgen cublaslt
//go:generate nvgen cufft
//go:generate nvgen curand
//go:generate nvgen cusolver
//go:generate nvgen cusparse
//go:generate nvgen cufile
//go:generate nvgen cudnn
//go:generate nvgen cutensor
//go:generate nvgen fabricmanager
//go:generate nvgen nccl
//go:generate nvgen nvfatbin
//go:generate nvgen nvjitlink
//go:generate nvgen nvvm
//go:generate nvgen sanitizer
//go:generate nvgen cublasmp
//go:generate nvgen cudensitymat
//go:generate nvgen cudss
//go:generate nvgen cusolvermp
//go:generate nvgen custatevec
//go:generate nvgen cutensornet
//go:generate nvgen nvshmem
//go:generate nvgen cupti

// Independent NVIDIA SDK inputs; the CUDA profile does not select these.
//go:generate nvgen dcgm
//go:generate nvgen gdrcopy
//go:generate nvgen ncclep
//go:generate nvgen ncclll
//go:generate nvgen ncclm2n
//go:generate nvgen nvcomp
//go:generate nvgen nvdec
//go:generate nvgen nvenc
//go:generate nvgen nvjpeg2k
