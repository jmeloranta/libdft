
include ../../../make.conf

all: classical

classical: classical.o
	$(CC) $(CFLAGS) -o classical classical.o $(LDFLAGS)

classical.o: classical.c verlet.c
	$(CC) $(CFLAGS) -c classical.c

clean:
	-rm *.o classical classical.o *~
