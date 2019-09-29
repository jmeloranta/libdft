ifeq ($(shell test -e ../../../make.conf),yes)
  include ../../../make.conf
else
  include /usr/include/dft/make.conf
endif

all: film film2

film: film.o
	$(CC) $(CFLAGS) -o film film.o $(LDFLAGS)

film.o: film.c
	$(CC) $(CFLAGS) -c film.c

film2: film2.o
	$(CC) $(CFLAGS) -o film2 film2.o $(LDFLAGS)

film2.o: film2.c
	$(CC) $(CFLAGS) -c film2.c


clean:
	-rm *.o film film.o film-* *.pair *~
