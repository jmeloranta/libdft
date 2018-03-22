ifeq ($(shell test -e ../../../make.conf),yes)
  include ../../../make.conf
else
  include /usr/include/dft/make.conf
endif

all: piston

piston: piston.o
	$(CC) $(CFLAGS) -o piston piston.o $(LDFLAGS)

piston.o: piston.c
	$(CC) $(CFLAGS) -c piston.c

clean:
	-rm *.o piston piston.o piston-* *~
