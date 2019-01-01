ifeq ($(shell test -e ../../../make.conf),yes)
  include ../../../make.conf
else
  include /usr/include/dft/make.conf
endif

all: bubble-1d

bubble-1d: bubble-1d.o
	$(CC) $(CFLAGS) -o bubble-1d bubble-1d.o $(LDFLAGS)

bubble-1d.o: bubble-1d.c
	$(CC) $(CFLAGS) -c bubble-1d.c

clean:
	-rm *.o bubble-1d bubble-1d.o *~ *.grd *.x *.y *.z
