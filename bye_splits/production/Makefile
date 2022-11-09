CC = g++
EXEC = skimming.exe #basename must match the name of the *.cc file containing the main
RM = rm -r

BASEDIR := $(shell pwd)
SRCDIR  := src
INCDIR  := include
DEPDIR  := .deps

DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d
DEBUG_LEVEL := -g -fdiagnostics-color=never
EXTRA_CCFLAGS := -Wall -std=c++17 -O -pedantic -pedantic-errors \
	-Wformat-security \
	-Wformat-y2k \
	-Wimport  -Winit-self \
	-Winvalid-pch \
	-Wunsafe-loop-optimizations -Wmissing-braces \
	-Wmissing-field-initializers -Wmissing-format-attribute \
	-Wmissing-include-dirs -Wmissing-noreturn

CXXFLAGS        = $(DEBUG_LEVEL) $(EXTRA_CCFLAGS)
CCFLAGS         = $(CXXFLAGS)

ROOTFLAGS = `root-config --cflags --ldflags` -L $(ROOTSYS)/lib -lMinuit -lrt -lCore -lROOTDataFrame
EXTRAFLAGS = $(ROOTFLAGS)

SRCS := $(basename $(EXEC)).cc \
	$(wildcard $(SRCDIR)/*.cc)

OBJS := $(patsubst %.cc, %.o, $(SRCS))

DEPFILES := $(patsubst %.cc, $(DEPDIR)/%.d, $(notdir $(SRCS)))

.PHONY: all clean
.DEFAULT_GOAL = all

all: $(DEPDIR) $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CCFLAGS) $^ $(EXTRAFLAGS) -o $@
	@echo Executable $(EXEC) created.

%.o: %.cc #rewrite implicit rules
%.o: %.cc Makefile
	$(CC) $(DEPFLAGS) $(CCFLAGS) -c $< $(EXTRAFLAGS) -I$(BASEDIR) -o $@

$(SRCDIR)/%.o: $(SRCDIR)/%.cc $(DEPDIR)/%.d | $(DEPDIR)
	$(CC) $(DEPFLAGS) $(CCFLAGS) -c $< $(EXTRAFLAGS) -I$(BASEDIR) -o $@

$(DEPDIR):
	@mkdir -p $@

$(DEPFILES):

clean:
	$(RM) $(OBJS) $(EXEC) $(DEPDIR)

-include $(wildcard $(DEPFILES))
