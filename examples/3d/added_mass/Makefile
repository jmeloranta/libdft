include /home/dmateo/code/libdft-code/make.conf

all: extpot

added_mass: added_mass.o
	        $(CC) $(CFLAGS) -o added_mass added_mass.o $(LDFLAGS) 

added_mass.o: added_mass.c
	        $(CC) $(CFLAGS) -c added_mass.c

extpot: extpot.o
	        $(CC) $(CFLAGS) -o extpot extpot.o $(LDFLAGS) 

extpot.o: extpot.c
	        $(CC) $(CFLAGS) -c extpot.c

clean:
	rm added_mass extpot *.[oxyz] *.grd *.xy *.zx *.yz

