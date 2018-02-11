ifeq ($(shell test -e ../../../make.conf),yes)
  include ../../../make.conf
else
  include /usr/include/dft/make.conf
endif

all: map

map: map.o
	$(CC) $(CFLAGS) -o map map.o $(LDFLAGS)

map.o: map.c
	$(CC) $(CFLAGS) -c map.c

clean:
	-rm *.o map map.o *~
