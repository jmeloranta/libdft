ifeq ($(shell test -e ../../../make.conf),yes)
  include ../../../make.conf
else
  include /usr/include/dft/make.conf
endif

all: bubble

bubble: bubble.o pot.o analyze.o
	        $(CC) $(CFLAGS) -o bubble bubble.o pot.o analyze.o $(LDFLAGS) 

bubble.o: bubble.c bubble.h config.h
	        $(CC) $(CFLAGS) -c bubble.c

pot.o: pot.c config.h
	        $(CC) $(CFLAGS) -c pot.c

analyze.o: analyze.c config.h
	        $(CC) $(CFLAGS) -c analyze.c

clean:
	-rm bubble *.[oxyz] *.grd *.x *.y *.z *~ *.dat


