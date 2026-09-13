// Package cudatool runs CUDA binary utilities without a shell, and answers what
// options the installed toolkit's utilities accept.
//
// # Resolving a tool
//
// [LookTool] prefers a CUDA toolkit install over PATH, because a distribution
// package can leave an older build of a tool earlier on PATH than the toolkit's
// own, and such a build may not merely be out of date: it can fail to start
// against the libraries it finds, or refuse the machine's devices. An explicit
// path is taken as given, and CUDA_HOME, CUDA_PATH or CUDA_ROOT name a root
// ahead of the default prefixes.
//
// # Option differences between toolkits
//
// Options move between releases, and a command line that works under one
// toolkit can fail under another. Measured across the toolkits installed on the
// GB10 host, 13.0 to 13.4 renamed nvcc's --sanitize to --fdevice-sanitize, added
// five ptxas options and two nvlink options, and left nvdisasm's and cuobjdump's
// option names alone. Going further back, 12.6 to 13.0 gave nvdisasm
// --emit-json, which is the one option the packages in this tree emit that an
// installed toolkit may not have. Within compute-sanitizer, --kill and
// --track-unused-memory take no argument on any version, and passing one is
// rejected outright rather than ignored.
//
// So that callers need not guess, the optiondata directory holds one distilled
// option set per tool per toolkit version: a line per option naming its arity,
// its short form, its documented default and its enumerated values. That form is
// a fifth the size of the help output it came from, it is what a command line
// actually depends on, and a diff of two versions is the flag difference itself.
//
// [Capabilities] answers from that record whenever it covers the resolved
// toolkit, which costs one --version run. A toolkit newer than anything recorded
// is asked directly with --help, and the answer reports Probed so a caller can
// say it is working beyond what was measured.
//
// # Recording a new toolkit
//
// TestCorpusMatchesInstalledToolkits compares the record against every toolkit
// installed on the host, and is the gate on an upgrade: a changed option, arity,
// default or value list fails there rather than in a caller. To record a toolkit
// this machine has:
//
//	go test ./x/internal/cudatool -update
//
// To record one only another machine has, capture its help output there as
// <tool>-<version>.txt and distill it here. Note that custom flags must follow
// the package argument:
//
//	for v in 12.6 13.0 13.4; do
//	    for t in nvcc ptxas nvdisasm cuobjdump nvlink nvprune compute-sanitizer; do
//	        ssh gpu-host /usr/local/cuda-$v/bin/$t --help > raw/$t-$v.txt
//	    done
//	done
//	go test ./x/internal/cudatool -update -helpdir=raw
//
// Review the diff: it is the flag difference between the toolkits, and it is
// worth reading rather than committing unseen.
package cudatool
