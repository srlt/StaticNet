NAME = mnist
BIN  = bin/$(NAME)
SRC  = src
HDR  = ../../src

HEADERS = $(wildcard $(SRC)/*.h) $(wildcard $(SRC)/*.hpp) $(wildcard $(HDR)/*.h) $(wildcard $(HDR)/*.hpp)
SOURCES = $(wildcard $(SRC)/*.S) $(wildcard $(SRC)/*.c) $(wildcard $(SRC)/*.cpp)
OBJECTS = $(SOURCES:%=%.o)

AS       := $(AS)
ASFLAGS  :=
CC       := cc
CCFLAGS  := -Wall -Ofast -std=c11 -I$(HDR)
CXX      := c++
CXXFLAGS := -Wall -Ofast -std=c++14 -I$(HDR)
LD       := c++
LDFLAGS  :=

PLOT_DIR = plot
PLOT_GP  = $(PLOT_DIR)/plot.gp
PLOT_DAT = $(PLOT_DIR)/transfert.dat

.PHONY: build run plot clean

build: $(BIN)
run: $(BIN)
	@$(BIN) $(ARGS)
plot: $(PLOT_GP) $(PLOT_DAT)
	@gnuplot -e "filename='$(PLOT_DAT)'" $(PLOT_GP)
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

$(PLOT_DAT): $(BIN) $(PLOT_GP)
	@$(BIN) plot $(ARGS) > $(PLOT_DAT)
