#define _GNU_SOURCE
/*
 * Deterministic controls for the CUPTI producer. Each scenario drives the
 * real producer path -- buffer_requested, buffer_completed, the bounded ring,
 * the batched drain and the journal -- through the stub CUPTI in testcupti,
 * so no GPU and no CUDA toolkit are required. The harness asserts nothing; it
 * offers a known number of records and reports what it offered. The Go test
 * decodes the journal and owns every assertion.
 *
 * Usage: cupti-agent-control <scenario> [args...]
 */
#include "testcupti/cupti.h"

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/wait.h>
#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <unistd.h>

/* The producer's external-correlation wrappers, which a target calls directly.
   Declared here rather than included so the control links against the real
   symbols the shipping library exports. */
CUptiResult cudaprof_push_external(uint32_t external_kind, uint64_t external_id);
CUptiResult cudaprof_pop_external(uint32_t external_kind, uint64_t *external_id);

/* Counts this process's open descriptors whose /proc/self/fd link matches a
   prefix or suffix, which is how the control observes that a forked child
   really closed the parent's journal and termination pipe rather than merely
   forgetting their numbers. Returns -1 where /proc is unavailable, and the
   control then reports the descriptor question as unanswered rather than
   passed. */
static int count_fds(const char *suffix, const char *prefix) {
	DIR *d = opendir("/proc/self/fd");
	if (d == NULL) return -1;
	int n = 0;
	struct dirent *e;
	while ((e = readdir(d)) != NULL) {
		char path[64], target[PATH_MAX];
		if (e->d_name[0] == '.') continue;
		if (snprintf(path, sizeof(path), "/proc/self/fd/%s", e->d_name) >= (int)sizeof(path)) continue;
		ssize_t len = readlink(path, target, sizeof(target) - 1);
		if (len < 0) continue;
		target[len] = '\0';
		if (prefix != NULL && strncmp(target, prefix, strlen(prefix)) == 0) { n++; continue; }
		if (suffix != NULL) {
			size_t tl = strlen(target), sl = strlen(suffix);
			if (tl >= sl && strcmp(target + tl - sl, suffix) == 0) n++;
		}
	}
	closedir(d);
	return n;
}

/* What the forked child observed about the agent it inherited. The child
   cannot assert; it measures and hands the numbers back through a pipe, and
   the Go control owns every assertion. */
struct child_probe {
	int buffer_granted;   /* buffer_requested handed out a buffer */
	int push_result;      /* cudaprof_push_external return */
	int pop_result;       /* cudaprof_pop_external return */
	int journal_fds;      /* open descriptors on a capture.journal */
	int pipe_fds;         /* open pipe descriptors */
};

struct offer_plan {
	unsigned buffers;       /* buffers this thread offers */
	unsigned per_buffer;    /* records per buffer */
	unsigned pause_micros;  /* pause between buffers, 0 for a burst */
	long offered;           /* out: records accepted into buffers */
	long exhausted;         /* out: offers refused because the pool was busy */
};

static void *offer_thread(void *arg) {
	struct offer_plan *plan = arg;
	unsigned kernels = plan->per_buffer / 2;
	unsigned memcpys = plan->per_buffer / 4;
	unsigned apis = plan->per_buffer - kernels - memcpys;
	for (unsigned i = 0; i < plan->buffers; i++) {
		int placed = cudaprof_test_offer(kernels, memcpys, apis, 0);
		if (placed < 0) break;
		if (placed == 0) { plan->exhausted++; continue; }
		plan->offered += placed;
		if (plan->pause_micros) {
			struct timespec ts = {0, (long)plan->pause_micros * 1000};
			nanosleep(&ts, NULL);
		}
	}
	return NULL;
}

/* Runs threads offer_plans concurrently and returns the totals. */
static void run_offers(struct offer_plan *plans, unsigned threads, long *offered, long *exhausted) {
	pthread_t ids[64];
	if (threads > 64) threads = 64;
	for (unsigned i = 1; i < threads; i++) {
		if (pthread_create(&ids[i], NULL, offer_thread, &plans[i]) != 0) { plans[i].buffers = 0; ids[i] = 0; }
	}
	offer_thread(&plans[0]);
	for (unsigned i = 1; i < threads; i++) if (ids[i]) pthread_join(ids[i], NULL);
	*offered = 0;
	*exhausted = 0;
	for (unsigned i = 0; i < threads; i++) { *offered += plans[i].offered; *exhausted += plans[i].exhausted; }
}

struct trigger {
	unsigned delay_micros;
	int signo;
};

static void *trigger_thread(void *arg) {
	struct trigger *t = arg;
	struct timespec ts = {t->delay_micros / 1000000u, (long)(t->delay_micros % 1000000u) * 1000};
	nanosleep(&ts, NULL);
	kill(getpid(), t->signo);
	return NULL;
}

/* Stands in for an interpreter that installs its own SIGTERM handler after
   the producer's ELF constructor has already run, replacing it. CPython does
   exactly this, so the control exists to show what the producer still
   guarantees once its handler is gone. */
static volatile sig_atomic_t app_terminated;

static void app_sigterm(int signo) { (void)signo; app_terminated = 1; }

static unsigned arg_uint(int argc, char **argv, int index, unsigned fallback) {
	if (index >= argc) return fallback;
	unsigned long v = strtoul(argv[index], NULL, 10);
	return (unsigned)v;
}

int main(int argc, char **argv) {
	if (!cudaprof_test_callbacks_registered()) {
		fprintf(stderr, "producer did not register CUPTI callbacks; CUDAPROF_TRANSPORT/CUDAPROF_SESSION unset?\n");
		return 2;
	}
	const char *scenario = argc > 1 ? argv[1] : "sustained";
	unsigned threads = arg_uint(argc, argv, 2, 1);
	unsigned buffers = arg_uint(argc, argv, 3, 8);
	unsigned per_buffer = arg_uint(argc, argv, 4, 512);
	unsigned pause_micros = arg_uint(argc, argv, 5, 0);
	if (threads == 0) threads = 1;
	if (threads > 64) threads = 64;

	struct offer_plan plans[64];
	memset(plans, 0, sizeof(plans));
	for (unsigned i = 0; i < threads; i++) {
		plans[i].buffers = buffers;
		plans[i].per_buffer = per_buffer;
		plans[i].pause_micros = pause_micros;
	}

	pthread_t trigger_id = 0;
	struct trigger t = {0, 0};
	if (strcmp(scenario, "clobber") == 0 || strcmp(scenario, "fork-inert-clobber") == 0) {
		struct sigaction sa;
		memset(&sa, 0, sizeof(sa));
		sa.sa_handler = app_sigterm;
		sigemptyset(&sa.sa_mask);
		if (sigaction(SIGTERM, &sa, NULL) != 0) { fprintf(stderr, "install app handler: %s\n", strerror(errno)); return 2; }
		if (strcmp(scenario, "clobber") == 0) {
			t.signo = SIGTERM;
			t.delay_micros = arg_uint(argc, argv, 6, 20000);
		}
	}
	else if (strcmp(scenario, "race") == 0) { t.signo = SIGTERM; t.delay_micros = arg_uint(argc, argv, 6, 20000); }
	else if (strcmp(scenario, "forced-kill") == 0) { t.signo = SIGKILL; t.delay_micros = arg_uint(argc, argv, 6, 20000); }
	if (t.signo != 0 && pthread_create(&trigger_id, NULL, trigger_thread, &t) != 0) {
		fprintf(stderr, "trigger thread: %s\n", strerror(errno));
		return 2;
	}

	long offered = 0, exhausted = 0;
	run_offers(plans, threads, &offered, &exhausted);
	if (trigger_id) pthread_join(trigger_id, NULL);

	/* A forked child inherits the agent but not its drain thread. It must
	   exit on its own rather than hang finalizing, and must leave the
	   parent's journal alone. The child exits through exit(3) so the ELF
	   destructor and atexit handler both run; _exit would skip the path
	   under test. */
	int child_exited = 0, child_status = -1;
	if (strcmp(scenario, "fork") == 0) {
		pid_t pid = fork();
		if (pid == 0) exit(0);
		if (pid < 0) { fprintf(stderr, "fork: %s\n", strerror(errno)); return 2; }
		for (int i = 0; i < 500; i++) {
			pid_t got = waitpid(pid, &child_status, WNOHANG);
			if (got == pid) { child_exited = 1; break; }
			if (got < 0) break;
			struct timespec ts = {0, 10000000L};
			nanosleep(&ts, NULL);
		}
		if (!child_exited) kill(pid, SIGKILL);
	}


	/* Fork inertness, the narrow question 6DE59D5A asked. The child probes
	   every entry point it inherited and hands the numbers back, then raises
	   SIGTERM at itself. Two things are under test at once: the child must be
	   refused a buffer, refused both external wrappers, and hold neither the
	   parent's journal nor either end of its termination pipe; and the child's
	   SIGTERM must kill the child by the disposition the target had before
	   load without waking the parent's terminator. The parent proves the
	   second half by signalling itself afterwards and still finalizing, which
	   it cannot do if the child's signal already consumed its pipe byte. */
	struct child_probe probe = {-1, -1, -1, -1, -1};
	int parent_journal_fds = count_fds("/capture.journal", NULL);
	int parent_pipe_fds = count_fds(NULL, "pipe:");
	int probe_read = 0;
	int child_signaled = 0, child_termsig = -1;
	int parent_push_result = -1, parent_pop_result = -1;
	if (strcmp(scenario, "fork-inert") == 0) {
		/* Probe the wrappers in the parent first. Without this the child's
		   refusal would prove nothing: these wrappers also refuse a run that
		   never had external correlation enabled at all. */
		uint64_t parent_popped = 0;
		parent_push_result = (int)cudaprof_push_external(1, 98);
		parent_pop_result = (int)cudaprof_pop_external(1, &parent_popped);
		int report[2];
		if (pipe(report) != 0) { fprintf(stderr, "pipe: %s\n", strerror(errno)); return 2; }
		pid_t pid = fork();
		if (pid == 0) {
			close(report[0]);
			struct child_probe p;
			p.buffer_granted = cudaprof_test_request_buffer();
			uint64_t popped = 0;
			p.push_result = (int)cudaprof_push_external(1, 99);
			p.pop_result = (int)cudaprof_pop_external(1, &popped);
			p.journal_fds = count_fds("/capture.journal", NULL);
			p.pipe_fds = count_fds(NULL, "pipe:");
			/* The report pipe the child just inherited and still holds is one
			   of the pipe descriptors it counts; discount it so the number
			   means the producer's pipe alone. */
			if (p.pipe_fds > 0) p.pipe_fds--;
			ssize_t w = write(report[1], &p, sizeof(p));
			(void)w;
			close(report[1]);
			raise(SIGTERM);
			/* Only reached if SIGTERM was not restored to a disposition that
			   ends the child; the nonzero exit distinguishes that from a
			   child killed by the signal. */
			_exit(9);
		}
		if (pid < 0) { fprintf(stderr, "fork: %s\n", strerror(errno)); return 2; }
		close(report[1]);
		if (read(report[0], &probe, sizeof(probe)) == (ssize_t)sizeof(probe)) probe_read = 1;
		close(report[0]);
		if (waitpid(pid, &child_status, 0) == pid) {
			child_exited = WIFEXITED(child_status);
			child_signaled = WIFSIGNALED(child_status);
			if (child_signaled) child_termsig = WTERMSIG(child_status);
		}
	}
	/* An application handler installed after the producer loaded belongs to
	   the application. The child must inherit it untouched: restoring the
	   producer's saved disposition there would replace the target's handler
	   with SIG_DFL and change how the child terminates, which is the one thing
	   the producer promises never to do. The child raises SIGTERM at itself
	   and exits 5 only if the application's handler ran. */
	if (strcmp(scenario, "fork-inert-clobber") == 0) {
		uint64_t parent_popped = 0;
		parent_push_result = (int)cudaprof_push_external(1, 96);
		parent_pop_result = (int)cudaprof_pop_external(1, &parent_popped);
		int report[2];
		if (pipe(report) != 0) { fprintf(stderr, "pipe: %s\n", strerror(errno)); return 2; }
		pid_t pid = fork();
		if (pid == 0) {
			close(report[0]);
			struct child_probe p;
			p.buffer_granted = cudaprof_test_request_buffer();
			uint64_t popped = 0;
			p.push_result = (int)cudaprof_push_external(1, 97);
			p.pop_result = (int)cudaprof_pop_external(1, &popped);
			p.journal_fds = count_fds("/capture.journal", NULL);
			p.pipe_fds = count_fds(NULL, "pipe:");
			if (p.pipe_fds > 0) p.pipe_fds--;
			ssize_t w = write(report[1], &p, sizeof(p));
			(void)w;
			close(report[1]);
			raise(SIGTERM);
			_exit(app_terminated ? 5 : 6);
		}
		if (pid < 0) { fprintf(stderr, "fork: %s\n", strerror(errno)); return 2; }
		close(report[1]);
		if (read(report[0], &probe, sizeof(probe)) == (ssize_t)sizeof(probe)) probe_read = 1;
		close(report[0]);
		if (waitpid(pid, &child_status, 0) == pid) {
			child_exited = WIFEXITED(child_status);
			child_signaled = WIFSIGNALED(child_status);
			if (child_signaled) child_termsig = WTERMSIG(child_status);
			if (child_exited) child_status = WEXITSTATUS(child_status);
		}
	}

	printf("{\"scenario\":\"%s\",\"threads\":%u,\"buffers\":%u,\"per_buffer\":%u,\"offered\":%ld,\"pool_exhausted\":%ld,\"app_terminated\":%d,\"child_exited\":%d,\"child_status\":%d,\"child_signaled\":%d,\"child_termsig\":%d,\"probe_read\":%d,\"parent_push_result\":%d,\"parent_pop_result\":%d,\"parent_journal_fds\":%d,\"parent_pipe_fds\":%d,\"child_buffer_granted\":%d,\"child_push_result\":%d,\"child_pop_result\":%d,\"child_journal_fds\":%d,\"child_pipe_fds\":%d}\n",
		scenario, threads, buffers, per_buffer, offered, exhausted, app_terminated ? 1 : 0, child_exited, child_status,
		child_signaled, child_termsig, probe_read, parent_push_result, parent_pop_result, parent_journal_fds, parent_pipe_fds,
		probe.buffer_granted, probe.push_result, probe.pop_result, probe.journal_fds, probe.pipe_fds);
	fflush(stdout);
	if (strcmp(scenario, "fork-inert") == 0) {
		/* The parent's own termination service must have survived the child's
		   signal: this SIGTERM has to reach the producer's handler, wake its
		   terminator, and finalize the journal. */
		raise(SIGTERM);
		for (int i = 0; i < 100; i++) { struct timespec ts = {0, 100000000L}; nanosleep(&ts, NULL); }
	}
	if (strcmp(scenario, "clobber") == 0) {
		/* The target owns SIGTERM now, so it winds down the way it always
		   would and the producer finalizes from atexit. */
		for (int i = 0; i < 100 && !app_terminated; i++) { struct timespec ts = {0, 10000000L}; nanosleep(&ts, NULL); }
		return app_terminated ? 7 : 8;
	}
	if (strcmp(scenario, "race") == 0 || strcmp(scenario, "forced-kill") == 0) {
		/* Stay alive so the signal decides how the process ends. */
		for (int i = 0; i < 100; i++) { struct timespec ts = {0, 100000000L}; nanosleep(&ts, NULL); }
	}
	return 0;
}
