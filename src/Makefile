include ../make.conf

OBJS = ot2d.o ot3d.o common.o driver2d.o driver3d.o driver3d_cyl.o classical.o bulk3d.o bulk2d.o

libdft.a: prototypes $(OBJS)
	$(AR) cr libdft.a $(OBJS)

prototypes:
	egrep -h \^EXPORT *.c | tr \{ \; | sed -e "s/EXPORT //g" > proto.h

install: 
	cp libdft.a $(ROOT)/lib
	-$(RANLIB) $(ROOT)/lib/libdft.a
	-mkdir $(ROOT)/include/dft
	cp dft.h ot.h proto.h $(ROOT)/include/dft
	cp ../make.conf $(ROOT)/include/dft

driver2d.o: driver2d.c ot.h
	$(CC) -I. $(CFLAGS) -c driver2d.c

driver3d.o: driver3d.c ot.h
	$(CC) -I. $(CFLAGS) -c driver3d.c

driver3d_cyl.o: driver3d_cyl.c ot.h
	$(CC) -I. $(CFLAGS) -c driver3d_cyl.c

common.o: common.c dft.h
	$(CC) -I. $(CFLAGS) -c common.c

ot2d.o: ot2d.c ot.h dft.h
	$(CC) -I. $(CFLAGS) -c ot2d.c

ot3d.o: ot3d.c ot.h dft.h
	$(CC) -I. $(CFLAGS) -c ot3d.c

classical.o: classical.c dft.h
	$(CC) -I. $(CFLAGS) -c classical.c

bulk3d.o: bulk3d.c ot.h
	$(CC) -I. $(CFLAGS) -c bulk3d.c

bulk2d.o: bulk2d.c ot.h
	$(CC) -I. $(CFLAGS) -c bulk2d.c

clean:
	-rm *.o libdft.a proto.h *~
