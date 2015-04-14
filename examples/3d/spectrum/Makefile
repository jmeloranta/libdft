
include ../../../make.conf

all: spectrum-nozp spectrum-eval

spectrum-nozp: spectrum-nozp.o
	$(CC) $(CFLAGS) -o spectrum-nozp spectrum-nozp.o $(LDFLAGS)

spectrum-nozp.o: spectrum-nozp.c
	$(CC) $(CFLAGS) -c spectrum-nozp.c

spectrum-eval: spectrum-eval.o
	$(CC) $(CFLAGS) -o spectrum-eval spectrum-eval.o $(LDFLAGS)

spectrum-eval.o: spectrum-eval.c
	$(CC) $(CFLAGS) -c spectrum-eval.c

clean:
	-rm *.o spectrum-nozp spectrum-nozp.o spectrum-eval spectrum-eval.o output* *~
