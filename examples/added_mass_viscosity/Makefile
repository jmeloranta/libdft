ifeq ($(shell test -e ../../../make.conf),yes)
  include ../../../make.conf
else
  include /usr/include/dft/make.conf
endif

all: added_mass2 added_mass4 analyze4

added_mass2: added_mass2.o
	        $(CC) $(CFLAGS) -o added_mass2 added_mass2.o $(LDFLAGS) 

added_mass2.o: added_mass2.c
	        $(CC) $(CFLAGS) -c added_mass2.c

added_mass4: added_mass4.o
	        $(CC) $(CFLAGS) -o added_mass4 added_mass4.o $(LDFLAGS) 

added_mass4.o: added_mass4.c added_mass4.h
	        $(CC) $(CFLAGS) -c added_mass4.c

analyze4: analyze4.o added_mass4.h
	        $(CC) $(CFLAGS) -o analyze4 analyze4.o $(LDFLAGS) 

analyze4.o: analyze4.c
	        $(CC) $(CFLAGS) -c analyze4.c
clean:
	-rm added_mass2 added_mass2.o added_mass4 added_mass4.o analyze4 analyze4.o


