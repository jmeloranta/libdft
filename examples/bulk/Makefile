include ../../make.conf

all: bulk_properties

bulk_properties: bulk_properties.o
	$(CC) $(CFLAGS) -o bulk_properties bulk_properties.o $(LDFLAGS)

bulk_properties.o: bulk_properties.c
	$(CC) $(CFLAGS) -c bulk_properties.c

clean:
	-rm *.o bulk_properties bulk_properties.o output* *~
