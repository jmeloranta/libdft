include ../../../make.conf

all: added_mass

added_mass: added_mass.o
	        $(CC) $(CFLAGS) -o added_mass added_mass.o $(LDFLAGS) 

added_mass.o: added_mass.c
	        $(CC) $(CFLAGS) -c added_mass.c
clean:
	rm added_mass extpot *.[oxyz] *.grd *.xy *.zx *.yz

