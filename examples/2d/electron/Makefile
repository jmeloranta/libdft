include ../../../make.conf

all: electron-2d

electron-2d: electron-2d.o e-he-ps.o
	  $(CC) $(CFLAGS) -o electron-2d electron-2d.o e-he-ps.o $(LDFLAGS)

electron-2d.o: electron-2d.c 
	    $(CC) $(CFLAGS) -c electron-2d.c

e-he-ps.o: e-he-ps.c 
	    $(CC) $(CFLAGS) -c e-he-ps.c

clean:
	-rm *.o *~ electron-2d


