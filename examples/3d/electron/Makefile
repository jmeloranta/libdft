include ../../../make.conf

all: electron

electron: electron.o
	  $(CC) $(CFLAGS) -o electron electron.o $(LDFLAGS)

electron.o: electron.c 
	    $(CC) $(CFLAGS) -c electron.c
clean:
	-rm *.o *~ electron


