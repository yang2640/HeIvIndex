CC=g++
CFLAGS=-std=c++0x -Wall -DNDEBUG -O3
LDFLAGS=-L/usr/local/lib
ODIR=obj

DEPS=he_ivindex.h
LIBS=-lopencv_core -lopencv_highgui -lopencv_ml -lopencv_flann

OBJ1_o=he_ivindex.o he_quantize.o
OBJ1=$(patsubst %, $(ODIR)/%, $(OBJ1_o))

OBJ2_o=he_ivindex.o he_query.o
OBJ2=$(patsubst %, $(ODIR)/%, $(OBJ2_o))

all: ../he_quantize ../kmeans ../he_median ../he_query

$(ODIR)/%.o: %.cpp $(DEPS)
	$(CC) -c $(CFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

../he_quantize: $(OBJ1)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

../he_query: $(OBJ2)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

../kmeans: kmeans.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

../he_median: he_median.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o  ../he_quantize ../kmeans ../he_median ../he_query
