include ../../../make.conf

all: added_mass3 analyze3 added_mass2

added_mass3: added_mass3.o
	        $(CC) $(CFLAGS) -o added_mass3 added_mass3.o $(LDFLAGS) 

added_mass3.o: added_mass3.c added_mass3.h
	        $(CC) $(CFLAGS) -c added_mass3.c

analyze3: analyze3.o added_mass3.h
	        $(CC) $(CFLAGS) -o analyze3 analyze3.o $(LDFLAGS) 

analyze3.o: analyze3.c
	        $(CC) $(CFLAGS) -c analyze3.c

added_mass2: added_mass2.o
	        $(CC) $(CFLAGS) -o added_mass2 added_mass2.o $(LDFLAGS) 

added_mass2.o: added_mass2.c
	        $(CC) $(CFLAGS) -c added_mass2.c

clean:
	-rm added_mass2 added_mass2.o added_mass3 added_mass3.o analyze3 analyze3.o

