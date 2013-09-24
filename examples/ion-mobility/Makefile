
include /home/git/libdft/make.conf

all: classical classical-3d

classical: classical.o
	$(CC) $(CFLAGS) -o classical classical.o $(LDFLAGS) 

classical.o: classical.c verlet.c
	$(CC) $(CFLAGS) -c classical.c

classical-3d: classical-3d.o
	$(CC) $(CFLAGS) -o classical-3d classical-3d.o $(LDFLAGS) 

classical-3d.o: classical-3d.c verlet-3d.c
	$(CC) $(CFLAGS) -c classical-3d.c

clean:
	-rm *.o classical classical.o classical-3d classical-3d.o *~
