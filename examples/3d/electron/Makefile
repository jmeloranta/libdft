ifeq ($(shell test -e ../../../make.conf),yes)
  include ../../../make.conf
else
  include /usr/include/dft/make.conf
endif

all: electron

electron: electron.o
	  $(CC) $(CFLAGS) -o electron electron.o $(LDFLAGS)

electron.o: electron.c 
	    $(CC) $(CFLAGS) -c electron.c
clean:
	-rm *.o *~ electron el-* helium-*

