
include ../../../make.conf

all: spectrum-nozp

spectrum-nozp: spectrum-nozp.o
	$(CC) $(CFLAGS) -o spectrum-nozp spectrum-nozp.o $(LDFLAGS)

spectrum-nozp.o: spectrum-nozp.c
	$(CC) $(CFLAGS) -c spectrum-nozp.c

clean:
	-rm *.o spectrum-nozp spectrum-nozp.o output* *~
