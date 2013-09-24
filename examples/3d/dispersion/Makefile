include ../../../make.conf

all: dispersion

dispersion: dispersion.o
	$(CC) $(CFLAGS) -o dispersion dispersion.o $(LDFLAGS)

dispersion.o: dispersion.c
	$(CC) $(CFLAGS) -c dispersion.c

clean:
	-rm *.o dispersion dispersion.o *~
