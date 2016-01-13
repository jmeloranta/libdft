include ../../../make.conf

all: bubble

bubble: bubble.o
	        $(CC) $(CFLAGS) -o bubble bubble.o $(LDFLAGS) 

bubble.o: bubble.c
	        $(CC) $(CFLAGS) -c bubble.c
clean:
	-rm bubble *.[oxyz] *.grd *.xy *.zx *.yz

