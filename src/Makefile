include ../make.conf

all:
	make prototypes
	make libdft.a

OBJS = ot.o ot-energy.o common.o bulk.o spectroscopy.o initial.o

libdft.a: $(OBJS)
	ar cr libdft.a $(OBJS)

prototypes:
	egrep -h \^EXPORT *.c | tr \{ \; | sed -e "s/EXPORT //g" > proto.h
	echo "#define VERSION \""`git rev-parse master`"\"" > git-version.h

install: 
	cp libdft.a $(ROOT)/lib
	-ranlib $(ROOT)/lib/libdft.a
	-mkdir $(ROOT)/include/dft
	cp dft.h ot.h proto.h $(ROOT)/include/dft
	cp ../make.conf $(ROOT)/include/dft

common.o: common.c ot.h dft.h
	$(CC) -I. $(CFLAGS) -c common.c

ot.o: ot.c ot.h ot-private.h dft.h
	$(CC) -I. $(CFLAGS) -c ot.c

ot-energy.o: ot-energy.c ot.h dft.h
	$(CC) -I. $(CFLAGS) -c ot-energy.c

bulk.o: bulk.c dft.h ot.h
	$(CC) -I. $(CFLAGS) -c bulk.c

spectroscopy.o: spectroscopy.c dft.h ot.h
	$(CC) -I. $(CFLAGS) -c spectroscopy.c

intial.o: initial.c dft.h ot.h
	$(CC) -I. $(CFLAGS) -c initial.c

clean:
	-rm *.o libdft.a proto.h git-version.h *~
