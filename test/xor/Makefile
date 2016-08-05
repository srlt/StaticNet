NAME = xor
BIN  = bin/$(NAME)
SRC  = src
HDR  = ../../src

HEADERS = $(wildcard $(SRC)/*.h) $(wildcard $(SRC)/*.hpp) $(wildcard $(HDR)/*.h) $(wildcard $(HDR)/*.hpp)
SOURCES = $(wildcard $(SRC)/*.S) $(wildcard $(SRC)/*.c) $(wildcard $(SRC)/*.cpp)
OBJECTS = $(SOURCES:%=%.o)

AS       := $(AS)
ASFLAGS  :=
CC       := cc
CCFLAGS  := -Wall -O2 -std=c11 -I$(HDR)
CXX      := c++
CXXFLAGS := -Wall -O2 -std=c++14 -I$(HDR)
LD       := c++
LDFLAGS  :=

.PHONY: build run clean

build: $(BIN)
run: $(BIN)
	@$(BIN) $(ARGS)
clean:
	$(RM) $(OBJECTS) $(BIN)

%.S.o: %.S $(HEADERS)
	$(AS) $(ASFLAGS) -o $@ $<
%.c.o: %.c $(HEADERS)
	$(CC) $(CCFLAGS) -c -o $@ $<
%.cpp.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BIN): $(OBJECTS)
	$(LD) $(LDFLAGS) -o $@ $^
