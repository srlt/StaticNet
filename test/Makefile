BIN = ../bin
SRC = ../src

CC     = g++
CFLAGS = -Wall -O2 -std=c++14 -I$(SRC)
LFLAGS =

.PHONY: clean

%: %.cpp ../src/staticnet.hpp
	$(CC) $(CFLAGS) -c -o $@.o $<
	$(CC) $(LFLAGS) -o $@ $@.o

clean:
	$(RM) $(wildcard *.o) $(patsubst %.o,%,$(wildcard *.o))
