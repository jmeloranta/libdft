include ../../../make.conf

all: bubble bubble2 analyze2

bubble: bubble.o
	        $(CC) $(CFLAGS) -o bubble bubble.o $(LDFLAGS) 

bubble.o: bubble.c
	        $(CC) $(CFLAGS) -c bubble.c

bubble2: bubble2.o
	        $(CC) $(CFLAGS) -o bubble2 bubble2.o $(LDFLAGS) 

bubble2.o: bubble2.c
	        $(CC) $(CFLAGS) -c bubble2.c

analyze2: analyze2.o
	        $(CC) $(CFLAGS) -o analyze2 analyze2.o $(LDFLAGS) 

analyze2.o: analyze2.c
	        $(CC) $(CFLAGS) -c analyze2.c

clean:
	-rm bubble bubble2 *.[oxyz] *.grd *.x *.y *.z liquid.dat

