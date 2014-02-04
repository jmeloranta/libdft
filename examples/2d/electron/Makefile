include ../../../make.conf

all: electron-2d

electron-2d: electron-2d.o
	  $(CC) $(CFLAGS) -o electron-2d electron-2d.o $(LDFLAGS)

electron-2d.o: electron-2d.c 
	    $(CC) $(CFLAGS) -c electron-2d.c

clean:
	-rm *.o *~ electron-2d


