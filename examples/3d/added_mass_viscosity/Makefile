include ../../../make.conf

all: added_mass2 added_mass3

added_mass2: added_mass2.o
	        $(CC) $(CFLAGS) -o added_mass2 added_mass2.o $(LDFLAGS) 

added_mass2.o: added_mass2.c
	        $(CC) $(CFLAGS) -c added_mass2.c

added_mass3: added_mass3.o
	        $(CC) $(CFLAGS) -o added_mass3 added_mass3.o $(LDFLAGS) 

added_mass3.o: added_mass3.c
	        $(CC) $(CFLAGS) -c added_mass3.c

clean:
	-rm added_mass2 added_mass2.o added_mass3 added_mass3.o

