ifeq ($(shell test -e ../../../make.conf),yes)
  include ../../../make.conf
else
  include /usr/include/dft/make.conf
endif

all: trace

trace: trace.o
	$(CC) $(CFLAGS) -o trace trace.o $(LDFLAGS)

trace.o: trace.c
	$(CC) $(CFLAGS) -c trace.c

clean:
	-rm *.o trace trace.o trace-* *~
