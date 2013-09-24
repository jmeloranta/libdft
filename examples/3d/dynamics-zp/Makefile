
include ../../../make.conf

all: dynamics-zp

dynamics-zp: dynamics-zp.o
	$(CC) $(CFLAGS) -o dynamics-zp dynamics-zp.o $(LDFLAGS)

dynamics-zp.o: dynamics-zp.c
	$(CC) $(CFLAGS) -c dynamics-zp.c

clean:
	-rm *.o dynamics-zp dynamics-zp.o output* *~
