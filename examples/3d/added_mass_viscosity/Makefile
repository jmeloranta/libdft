include ../../../make.conf

all: added_mass2

added_mass2: added_mass2.o
	        $(CC) $(CFLAGS) -o added_mass2 added_mass2.o $(LDFLAGS) 

added_mass2.o: added_mass2.c
	        $(CC) $(CFLAGS) -c added_mass2.c
clean:
	-rm added_mass2 added_mass2.o

