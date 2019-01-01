ifeq ($(shell test -e ../../../make.conf),yes)
  include ../../../make.conf
else
  include /usr/include/dft/make.conf
endif

all: soliton-1d

soliton-1d: soliton-1d.o
	$(CC) $(CFLAGS) -o soliton-1d soliton-1d.o $(LDFLAGS)

soliton-1d.o: soliton-1d.c
	$(CC) $(CFLAGS) -c soliton-1d.c

clean:
	-rm *.o soliton-1d soliton-1d.o *~ *.grd *.x *.y *.z
