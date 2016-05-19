
include ../../../make.conf 

all: shock

shock: shock.o
	$(CC) $(CFLAGS) -o shock shock.o $(LDFLAGS)

shock.o: shock.c
	$(CC) $(CFLAGS) -c shock.c

clean:
	-rm *.o shock shock.o output* *~
