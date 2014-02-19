include ../../../make.conf

all: imag_time

imag_time: imag_time.o
	$(CC) $(CFLAGS) -o imag_time imag_time.o $(LDFLAGS)

imag_time.o: imag_time.c
	$(CC) $(CFLAGS) -c imag_time.c

clean:
	-rm *.o imag_time imag_time.o output-* *~ flux_*
