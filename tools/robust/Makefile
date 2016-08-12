NAME = robust
BIN  = bin/$(NAME)
SRC  = src
HDR  = ../../src

HEADERS = $(wildcard $(SRC)/*.h) $(wildcard $(SRC)/*.hpp) $(wildcard $(HDR)/*.h) $(wildcard $(HDR)/*.hpp)
SOURCES = $(wildcard $(SRC)/*.S) $(wildcard $(SRC)/*.c) $(wildcard $(SRC)/*.cpp)
OBJECTS = $(SOURCES:%=%.o)

PLOT_DIR  := plot
PLOT_GP   := $(PLOT_DIR)/plot.gp
PLOT_DATA := $(PLOT_DIR)/data.dat
PLOT_NET  := ../../test/mnist/net/784-98-10_epoch-1078_0.net
PLOT_CMDL := 784-98-10 1 0.25
PLOT_OUT  := $(PLOT_DATA).svg

AS       := $(AS)
ASFLAGS  :=
CC       := cc
CCFLAGS  := -Wall -Ofast -std=c11 -I$(HDR)
CXX      := c++
CXXFLAGS := -Wall -Ofast -std=c++14 -I$(HDR)
LD       := c++
LDFLAGS  :=
GP       := gnuplot
GPFLAGS  := -e "filename='$(PLOT_DATA)'"

.PHONY: plot build run clean

plot: $(PLOT_DATA)
	$(GP) $(GPFLAGS) $(PLOT_GP)
build: $(BIN)
run: $(PLOT_DATA)
clean:
	$(RM) $(OBJECTS) $(BIN) $(wildcard $(PLOT_DIR)/*.dat) $(wildcard $(PLOT_DIR)/*.svg)

%.S.o: %.S $(HEADERS)
	$(AS) $(ASFLAGS) -o $@ $<
%.c.o: %.c $(HEADERS)
	$(CC) $(CCFLAGS) -c -o $@ $<
%.cpp.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BIN): $(OBJECTS)
	$(LD) $(LDFLAGS) -o $@ $^

$(PLOT_DATA): $(BIN) $(PLOT_NET)
	$(BIN) $(PLOT_CMDL) < $(PLOT_NET) > $(PLOT_DATA)
