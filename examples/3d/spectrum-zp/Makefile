
include ../../../make.conf

all: spectrum-zp

spectrum-zp: spectrum-zp.o
	$(CC) $(CFLAGS) -o spectrum-zp spectrum-zp.o $(LDFLAGS)

spectrum-zp.o: spectrum-zp.c
	$(CC) $(CFLAGS) -c spectrum-zp.c

clean:
	-rm *.o spectrum-zp spectrum-zp.o output* *~
