ifeq ($(shell test -e ../../../make.conf),yes)
  include ../../../make.conf
else
  include /usr/include/dft/make.conf
endif

all: shock

shock: shock.o
	$(CC) $(CFLAGS) -o shock shock.o $(LDFLAGS)

shock.o: shock.c
	$(CC) $(CFLAGS) -c shock.c

clean:
	-rm *.o shock shock.o output* *~
