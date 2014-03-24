include /home/dmateo/code/libdft-code/make.conf

all: dynamic_dampingmic_damping

dynamic_damping: dynamic_damping.o
	        $(CC) $(CFLAGS) -o dynamic_damping dynamic_damping.o $(LDFLAGS) 

dynamic_damping.o: dynamic_damping.c
	        $(CC) $(CFLAGS) -c dynamic_damping.c

clean:
	rm dynamic_damping extpot *.[oxyz] *.grd *.xy *.zx *.yz 
