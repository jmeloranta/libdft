include ../../../make.conf

all: imag_time imag_time_pes imag_time_pes_zp he3+

he3+: he3+.o
	$(CC) $(CFLAGS) -o he3+ he3+.o $(LDFLAGS)

he3+.o: he3+.c
	$(CC) $(CFLAGS) -c he3+.c

imag_time: imag_time.o
	$(CC) $(CFLAGS) -o imag_time imag_time.o $(LDFLAGS)

imag_time.o: imag_time.c
	$(CC) $(CFLAGS) -c imag_time.c

imag_time_pes: imag_time_pes.o
	$(CC) $(CFLAGS) -o imag_time_pes imag_time_pes.o $(LDFLAGS)

imag_time_pes.o: imag_time_pes.c
	$(CC) $(CFLAGS) -c imag_time_pes.c

imag_time_pes_zp: imag_time_pes_zp.o
	$(CC) $(CFLAGS) -o imag_time_pes_zp imag_time_pes_zp.o $(LDFLAGS)

imag_time_pes_zp.o: imag_time_pes_zp.c
	$(CC) $(CFLAGS) -c imag_time_pes_zp.c


clean:
	-rm *.o imag_time imag_time.o imag_time_pes imag_time_pes.o imag_time_pes_zp imag_time_pes_zp.o output* flux* *~
