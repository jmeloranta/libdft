ifeq ($(shell test -e ../../../make.conf),yes)
  include ../../../make.conf
else
  include /usr/include/dft/make.conf
endif

all: fdyn

fdyn: fdyn.o
	$(CC) -g $(CFLAGS) -o fdyn fdyn.o $(LDFLAGS)

fdyn.o: fdyn.c
	$(CC) -g $(CFLAGS) -c fdyn.c

clean:
	-rm *.o fdyn *~ output-*
