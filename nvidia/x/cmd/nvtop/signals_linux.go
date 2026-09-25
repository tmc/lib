package main

import "syscall"

// signals are the kill overlay's entries; entry 0 is Cancel.
var signals = []struct {
	name string
	sig  syscall.Signal
}{
	{"Cancel", 0},
	{"SIGHUP", syscall.SIGHUP},
	{"SIGINT", syscall.SIGINT},
	{"SIGQUIT", syscall.SIGQUIT},
	{"SIGILL", syscall.SIGILL},
	{"SIGTRAP", syscall.SIGTRAP},
	{"SIGABRT", syscall.SIGABRT},
	{"SIGBUS", syscall.SIGBUS},
	{"SIGFPE", syscall.SIGFPE},
	{"SIGKILL", syscall.SIGKILL},
	{"SIGUSR1", syscall.SIGUSR1},
	{"SIGSEGV", syscall.SIGSEGV},
	{"SIGUSR2", syscall.SIGUSR2},
	{"SIGPIPE", syscall.SIGPIPE},
	{"SIGALRM", syscall.SIGALRM},
	{"SIGTERM", syscall.SIGTERM},
	{"SIGCHLD", syscall.SIGCHLD},
	{"SIGCONT", syscall.SIGCONT},
	{"SIGSTOP", syscall.SIGSTOP},
	{"SIGTSTP", syscall.SIGTSTP},
	{"SIGTTIN", syscall.SIGTTIN},
	{"SIGTTOU", syscall.SIGTTOU},
	{"SIGURG", syscall.SIGURG},
	{"SIGXCPU", syscall.SIGXCPU},
	{"SIGXFSZ", syscall.SIGXFSZ},
	{"SIGVTALRM", syscall.SIGVTALRM},
	{"SIGPROF", syscall.SIGPROF},
	{"SIGWINCH", syscall.SIGWINCH},
	{"SIGIO", syscall.SIGIO},
	{"SIGPWR", syscall.SIGPWR},
	{"SIGSYS", syscall.SIGSYS},
}

func sendSignal(pid int, sig syscall.Signal) error { return syscall.Kill(pid, sig) }
