
include ../../../make.conf 

all: piston

piston: piston.o
	$(CC) $(CFLAGS) -o piston piston.o $(LDFLAGS)

piston.o: piston.c
	$(CC) $(CFLAGS) -c piston.c

clean:
	-rm *.o piston piston.o piston-* *~
