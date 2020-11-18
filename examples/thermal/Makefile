ifeq ($(shell test -e ../../../make.conf),yes)
  include ../../../make.conf
else
  include /usr/include/dft/make.conf
endif

all: thermal quench

thermal: thermal.o
	$(CC) $(CFLAGS) -o thermal thermal.o $(LDFLAGS)

thermal.o: thermal.c
	$(CC) $(CFLAGS) -c thermal.c

quench: quench.o
	$(CC) $(CFLAGS) -o quench quench.o $(LDFLAGS)

quench.o: quench.c
	$(CC) $(CFLAGS) -c quench.c

clean:
	-rm -f *.o thermal thermal.o quench quench.o output* *~ thermal-*.grd *.x *.y *.z *.dat
