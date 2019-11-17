include ../make.conf

all:
	make prototypes
	make libdft.a

OBJS = ot.o ot-energy.o common.o bulk.o spectroscopy1a.o spectroscopy1b.o spectroscopy2.o spectroscopy3.o initial.o classical.o classical-common.o

libdft.a: $(OBJS)
	ar cr libdft.a $(OBJS)

prototypes:
	egrep -h \^EXPORT *.c | tr \{ \; | sed -e "s/EXPORT //g" > proto.h
	echo "#define VERSION \""`git rev-parse master`"\"" > git-version.h

install: 
	-mkdir -p $(ROOT)/lib
	cp libdft.a $(ROOT)/lib
	-ranlib $(ROOT)/lib/libdft.a
	-mkdir -p $(ROOT)/include/dft
	cp dft.h ot.h classical.h proto.h $(ROOT)/include/dft
	cp ../make.conf $(ROOT)/include/dft

common.o: common.c ot.h dft.h
	$(CC) -I. $(CFLAGS) -c common.c

ot.o: ot.c ot.h ot-private.h dft.h
	$(CC) -I. $(CFLAGS) -c ot.c

ot-energy.o: ot-energy.c ot.h dft.h
	$(CC) -I. $(CFLAGS) -c ot-energy.c

bulk.o: bulk.c dft.h ot.h
	$(CC) -I. $(CFLAGS) -c bulk.c

spectroscopy1a.o: spectroscopy1a.c dft.h ot.h
	$(CC) -I. $(CFLAGS) -c spectroscopy1a.c

spectroscopy1b.o: spectroscopy1b.c dft.h ot.h
	$(CC) -I. $(CFLAGS) -c spectroscopy1b.c

spectroscopy2.o: spectroscopy2.c dft.h ot.h
	$(CC) -I. $(CFLAGS) -c spectroscopy2.c

spectroscopy3.o: spectroscopy3.c dft.h ot.h
	$(CC) -I. $(CFLAGS) -c spectroscopy3.c

intial.o: initial.c dft.h ot.h
	$(CC) -I. $(CFLAGS) -c initial.c

classical.o: classical.c dft.h ot.h classical.h
	$(CC) -I. $(CFLAGS) -c classical.c

classical-common.o: classical-common.c dft.h ot.h classical.h
	$(CC) -I. $(CFLAGS) -c classical-common.c

clean:
	-rm *.o libdft.a proto.h git-version.h *~
