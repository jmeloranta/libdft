include ../../../make.conf

all: dispersion dispersion2

dispersion: dispersion.o
	$(CC) $(CFLAGS) -o dispersion dispersion.o $(LDFLAGS)

dispersion.o: dispersion.c
	$(CC) $(CFLAGS) -c dispersion.c

dispersion2: dispersion2.o
	$(CC) $(CFLAGS) -o dispersion2 dispersion2.o $(LDFLAGS)

dispersion2.o: dispersion2.c
	$(CC) $(CFLAGS) -c dispersion2.c

clean:
	-rm *.o dispersion dispersion.o dispersion2.o dispersion2 *~
