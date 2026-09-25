// Nvtop shows NVIDIA GPU utilization, memory, clocks, temperature, power,
// and the processes using each GPU, in the manner of htop.
//
// It is a Go reimplementation of [Syllo/nvtop] for NVIDIA devices, following
// nvtop's development branch rather than its 3.3.2 release. It reads
// NVML through [github.com/tmc/lib/nvidia/nvml], so it needs libnvidia-ml.so
// but no cgo.
//
// Usage:
//
//	nvtop [flags]
//
// The flags are:
//
//	-d tenths
//		Refresh every tenths/10 seconds (default 10, or 1 with -s
//		and -l).
//	-s
//		Print one JSON snapshot of every device and exit.
//	-l
//		Print a JSON snapshot every refresh interval.
//	-p
//		Do not draw the utilization plot.
//	-P
//		Do not list processes.
//	-r
//		Plot recent data on the left.
//	-C
//		Do not use color.
//	-f
//		Show temperatures in Fahrenheit.
//	-i
//		Add a line to each device header with more about the
//		device: see Interface.
//	-E seconds
//		Hide the encoder and decoder meters after this many idle
//		seconds (default 30); zero or negative keeps them shown.
//	-x
//		Turn on the extensions: see Extensions.
//	-v
//		Print the version and exit.
//	-h
//		Print the help and exit.
//
// Each flag also has the long spelling nvtop uses, such as --delay and
// --snapshot.
//
// # Interface
//
// The interactive interface draws a header per device, a plot of GPU and
// memory utilization over time, and the process list, laid out as nvtop
// lays them out for the terminal's size. It needs Linux; elsewhere use -s.
// The keys are:
//
//	q, Esc, F10       quit
//	Up/k, Down/j      select a process
//	Left/h, Right/l   scroll the process list sideways
//	F6                choose the sort column
//	+, -              sort ascending, descending
//	F9                send a signal to the selected process
//	0-9               in the signal window, choose a signal by number
//	F5, Ctrl-L        redraw
//
// Nvtop's setup window (F2) and configuration file (F12, -c) are not
// implemented, and with them the PCIe throughput plots that only the
// setup window turns on. A terminal too narrow for a window clips it
// rather than failing.
//
// Devices with ECC memory show the volatile corrected and uncorrected
// error counts, and devices with active NVLink links show the link
// version, the number of links, and their combined throughput.
//
// With -i, each header gains a line showing the number of shader cores
// (NSHC), the L2 cache (L2CF), and the execution engines (NEXC), and, on
// devices with active NVLink links, the flit CRC (FL), ECC (EE), and data
// CRC (CR) errors on those links since nvtop started. NVML reports
// neither the L2 cache nor the execution engines, so they show N/A.
//
// # Extensions
//
// With -x, nvtop shows what nvtop proper does not. The help lists the
// extension flags after nvtop's own.
//
// Each header gains a line with the energy the device has used since
// nvtop started (NRG) and the reasons its clocks are below their maximum
// (THR), as NVML's clock event reasons name them: idle, app-clocks,
// sw-power, hw-slow, sync-boost, sw-therm, hw-therm, hw-brake, display,
// board-limit, and reliability. Where NVML counts the time a reason has
// held the clocks, the reason shows the share of the last refresh
// interval it held them, and shows even when it is no longer active.
// Hardware slowdowns are red; software power and thermal limits yellow.
//
// The same line counts the NVML events on the device since nvtop started
// (EVT), in red once one was a fault such as an Xid error. The e key
// shows the event log in place of the process list, newest first, with
// the commonest Xids described; e or Esc goes back, and Up and Down
// scroll. Clock changes are not logged: the header shows the clocks, and
// they change many times a second.
//
// # Snapshots
//
// The snapshot format matches nvtop -s: an array of devices whose values
// are strings with units ("2502MHz", "55C", "49W", "7%"), memory in bytes
// as decimal strings, and null for values the device does not report.
// NVML has no memory type, so memory_type is always null.
//
// # Differences from nvtop
//
// On devices with unified memory, such as GB10, NVML does not report
// memory use, so both programs take it from /proc/meminfo: the total is
// MemTotal and the free memory is MemAvailable.
//
// Nvtop does not read the shader core count from NVML, so with -i it
// shows N/A for every NVIDIA device; this one shows the count.
//
// A process that NVML lists as both a graphics and a compute process
// appears once, as type "Both G+C", holding the larger of the two memory
// figures.
//
// The process list sorts stably: values a process does not report sort
// last in either direction, and ties keep device and process ID order.
//
// While the signal window is open the process list stays as it was, as
// in nvtop, but the device figures derived from it are still recomputed
// from that list. Resizing the terminal or redrawing it keeps the
// selection, scrolling, and any open window. With -P, F6 and F9 do
// nothing. SIGTERM, like SIGINT and SIGQUIT, restores the terminal before
// exiting.
//
// [Syllo/nvtop]: https://github.com/Syllo/nvtop
package main
