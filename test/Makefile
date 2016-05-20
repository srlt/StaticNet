BIN = ../bin
SRC = ../src

CC     = g++
CFLAGS = -Wall -Wfatal-errors -O2 -std=c++14 -I$(SRC) -fno-exceptions -fnothrow-opt -fno-extern-tls-init -fno-rtti -fstrict-enums -nostdlib
LFLAGS =

.PHONY: clean

%: %.cpp ../src/staticnet.hpp
	$(CC) $(CFLAGS) -c -o $@.o $<
	$(CC) $(LFLAGS) -o $@ $@.o

clean:
	$(RM) $(wildcard *.o) $(patsubst %.o,%,$(wildcard *.o))
