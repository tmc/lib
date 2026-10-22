# Include after the pinned oracle Makefile; do not copy source into that tree.
sample-observer.o: $(SAMPLE_OBSERVER_SOURCE)
	$(CC) -O2 -g -fno-fast-math -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=200809L -I$(ORACLE_DIR) -c -o $@ $<
sample-observer: sample-observer.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)
