ifeq ($(shell test -e ../../../make.conf),yes)
  include ../../../make.conf
else
  include /usr/include/dft/make.conf
endif

all: film

film: film.o
	$(CC) $(CFLAGS) -o film film.o $(LDFLAGS)

film.o: film.c
	$(CC) $(CFLAGS) -c film.c

clean:
	-rm *.o film film.o film-* *~
