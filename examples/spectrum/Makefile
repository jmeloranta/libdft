ifeq ($(shell test -e ../../../make.conf),yes)
  include ../../../make.conf
else
  include /usr/include/dft/make.conf
endif

all: spectrum-pol spectrum-anderson spectrum-anderson-zp spectrum-bin

spectrum-pol: spectrum-pol.o
	$(CC) $(CFLAGS) -o spectrum-pol spectrum-pol.o $(LDFLAGS)

spectrum-pol.o: spectrum-pol.c
	$(CC) $(CFLAGS) -c spectrum-pol.c

spectrum-anderson: spectrum-anderson.o
	$(CC) $(CFLAGS) -o spectrum-anderson spectrum-anderson.o $(LDFLAGS)

spectrum-anderson.o: spectrum-anderson.c
	$(CC) $(CFLAGS) -c spectrum-anderson.c

spectrum-anderson-zp: spectrum-anderson-zp.o
	$(CC) $(CFLAGS) -o spectrum-anderson-zp spectrum-anderson-zp.o $(LDFLAGS)

spectrum-anderson-zp.o: spectrum-anderson-zp.c
	$(CC) $(CFLAGS) -c spectrum-anderson-zp.c

spectrum-bin: spectrum-bin.o
	$(CC) $(CFLAGS) -o spectrum-bin spectrum-bin.o $(LDFLAGS)

spectrum-bin.o: spectrum-bin.c
	$(CC) $(CFLAGS) -c spectrum-bin.c

clean:
	-rm *.o spectrum-pol spectrum-pol.o spectrum-anderson spectrum-anderson.o 
	-rm spectrum-anderson-zp spectrum-anderson-zp.o spectrum-bin spectrum-bin.o output* *~
