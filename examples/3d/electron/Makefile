include ../../../make.conf

all: electron

electron: electron.o e-he-ps.o
	  $(CC) $(CFLAGS) -o electron electron.o e-he-ps.o $(LDFLAGS)

electron.o: electron.c 
	    $(CC) $(CFLAGS) -c electron.c

e-he-ps.o: e-he-ps.c 
	    $(CC) $(CFLAGS) -c e-he-ps.c

clean:
	-rm *.o *~ electron


