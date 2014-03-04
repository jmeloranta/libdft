include ../../../make.conf

all: map

map: map.o
	$(CC) $(CFLAGS) -o map map.o $(LDFLAGS)

map.o: map.c
	$(CC) $(CFLAGS) -c map.c

clean:
	-rm *.o map map.o *~
