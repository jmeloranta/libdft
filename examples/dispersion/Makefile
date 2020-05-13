ifeq ($(shell test -e ../../../make.conf),yes)
  include ../../../make.conf
else
  include /usr/include/dft/make.conf
endif

all: dispersion dispersion2 dispersion3 static-resp dispersion-exp

dispersion: dispersion.o
	$(CC) $(CFLAGS) -o dispersion dispersion.o $(LDFLAGS)

dispersion.o: dispersion.c
	$(CC) $(CFLAGS) -c dispersion.c

dispersion2: dispersion2.o
	$(CC) $(CFLAGS) -o dispersion2 dispersion2.o $(LDFLAGS)

dispersion2.o: dispersion2.c
	$(CC) $(CFLAGS) -c dispersion2.c

dispersion3: dispersion3.o
	$(CC) $(CFLAGS) -o dispersion3 dispersion3.o $(LDFLAGS)

dispersion3.o: dispersion3.c
	$(CC) $(CFLAGS) -c dispersion3.c

static-resp.o: static-resp.c
	$(CC) $(CFLAGS) -c static-resp.c

static-resp: static-resp.o
	$(CC) $(CFLAGS) -o static-resp static-resp.o $(LDFLAGS)

dispersion-exp: dispersion-exp.o
	$(CC) $(CFLAGS) -o dispersion-exp dispersion-exp.o $(LDFLAGS)

dispersion-exp.o: dispersion-exp.c
	$(CC) $(CFLAGS) -c dispersion-exp.c

clean:
	-rm *.o dispersion dispersion.o dispersion2 dispersion2.o static-resp static-resp.o dispersion-exp.o dispersion-exp *~
