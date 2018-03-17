include ../make.conf

OBJS = ot3d.o ot3d-cuda.o common.o driver3d.o classical.o bulk3d.o

libdft.a: prototypes | $(OBJS)
	$(AR) cr libdft.a $(OBJS)

prototypes:
	egrep -h \^EXPORT *.c | tr \{ \; | sed -e "s/EXPORT //g" > proto.h

install: 
	cp libdft.a $(ROOT)/lib
	-$(RANLIB) $(ROOT)/lib/libdft.a
	-mkdir $(ROOT)/include/dft
	cp dft.h ot.h proto.h $(ROOT)/include/dft
	cp ../make.conf $(ROOT)/include/dft

driver3d.o: driver3d.c ot.h
	$(CC) -I. $(CFLAGS) -c driver3d.c

common.o: common.c dft.h
	$(CC) -I. $(CFLAGS) -c common.c

ot3d.o: ot3d.c ot.h dft.h
	$(CC) -I. $(CFLAGS) -c ot3d.c

ot3d-cuda.o: ot3d-cuda.c ot.h dft.h
	$(CC) -I. $(CFLAGS) -c ot3d-cuda.c

classical.o: classical.c dft.h
	$(CC) -I. $(CFLAGS) -c classical.c

bulk3d.o: bulk3d.c ot.h
	$(CC) -I. $(CFLAGS) -c bulk3d.c

clean:
	-rm *.o libdft.a proto.h *~
