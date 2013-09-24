
include ../../../make.conf 

all: dynamics

dynamics: dynamics.o
	$(CC) $(CFLAGS) -o dynamics dynamics.o $(LDFLAGS)

dynamics.o: dynamics.c
	$(CC) $(CFLAGS) -c dynamics.c

clean:
	-rm *.o dynamics dynamics.o output* *~
