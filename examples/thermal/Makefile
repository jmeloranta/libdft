ifeq ($(shell test -e ../../../make.conf),yes)
  include ../../../make.conf
else
  include /usr/include/dft/make.conf
endif

all: thermal

thermal: thermal.o
	$(CC) $(CFLAGS) -o thermal thermal.o $(LDFLAGS)

thermal.o: thermal.c
	$(CC) $(CFLAGS) -c thermal.c

clean:
	-rm -f *.o thermal thermal.o output* *~ thermal-*.grd *.x *.y *.z *.dat
