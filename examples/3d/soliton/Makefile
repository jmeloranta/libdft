
include ../../../make.conf 

all: soliton

soliton: soliton.o
	$(CC) $(CFLAGS) -o soliton soliton.o $(LDFLAGS)

soliton.o: soliton.c
	$(CC) $(CFLAGS) -c soliton.c

clean:
	-rm *.o soliton soliton.o soliton-* *~
