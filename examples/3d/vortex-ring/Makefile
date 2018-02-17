ifeq ($(shell test -e ../../../make.conf),yes)
  include ../../../make.conf
else
  include /usr/include/dft/make.conf
endif

all: vring

vring: vring.o
	$(CC) $(CFLAGS) -o vring vring.o $(LDFLAGS)

vring.o: vring.c
	$(CC) $(CFLAGS) -c vring.c

clean:
	-rm *.o vring vring.o vring-* *~
