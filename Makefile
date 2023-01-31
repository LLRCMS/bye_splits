CC = g++
EXEC = produce.exe #basename must match the name of the *.cc file containing the main
RM = rm -r

BASEDIR := $(shell pwd)
BUILDIR := bye_splits/production
SRCDIR  := $(BUILDIR)/src
INCDIR  := $(BUILDIR)/include
DEPDIR  := $(BUILDIR)/.deps

DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d
DEBUG_LEVEL := -g -fdiagnostics-color=always
EXTRA_CCFLAGS := -Wall -std=c++11 -O -pedantic -pedantic-errors \
	-Wformat-security \
	-Wformat-y2k \
	-Wimport  -Winit-self \
	-Winvalid-pch \
	-Wunsafe-loop-optimizations -Wmissing-braces \
	-Wmissing-field-initializers -Wmissing-format-attribute \
	-Wmissing-include-dirs -Wmissing-noreturn

CXXFLAGS := $(DEBUG_LEVEL) $(EXTRA_CCFLAGS)
CCFLAGS  := $(CXXFLAGS)

ROOT_LIBS := $(shell root-config --cflags --ldflags --libs --glibs --auxlibs --auxcflags)
ROOT_LIBS_EXTRA := -lMinuit -lrt -lCore -lROOTDataFrame
BOOSTFLAGS := -lboost_program_options
ROOTFLAGS := $(ROOT_LIBS) -L $(ROOTSYS)/lib $(ROOT_LIBS_EXTRA)
YAMLFLAGS := -L $(BASEDIR)/yaml-cpp-yaml-cpp-0.7.0/build/ -lyaml-cpp
EXTRAFLAGS := $(ROOTFLAGS) $(YAMLFLAGS) $(BOOSTFLAGS)

SRCS := $(BUILDIR)/$(basename /$(EXEC)).cc \
	$(wildcard $(SRCDIR)/*.cc)

OBJS := $(patsubst %.cc, %.o, $(SRCS))

DEPFILES := $(patsubst %.cc, $(DEPDIR)/%.d, $(notdir $(SRCS)))

.PHONY: all clean
.DEFAULT_GOAL = all

all: $(DEPDIR) $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CCFLAGS) $^ $(EXTRAFLAGS) -o $@
	@echo Executable $(EXEC) created.

$(BUILDIR)%.o: $(BUILDIR)%.cc #rewrite implicit rules
$(BUILDIR)%.o: $(BUILDIR)%.cc Makefile
	$(CC) $(DEPFLAGS) $(CCFLAGS) -c $< $(EXTRAFLAGS) -I$(BASEDIR)/$(BUILDIR) -I $(BASEDIR)/yaml-cpp-yaml-cpp-0.7.0/include -o $@

$(SRCDIR)/%.o: $(SRCDIR)/%.cc $(DEPDIR)/%.d | $(DEPDIR)
	$(CC) $(DEPFLAGS) $(CCFLAGS) -c $< $(EXTRAFLAGS) -I$(BASEDIR)/$(BUILDIR) -I $(BASEDIR)/yaml-cpp-yaml-cpp-0.7.0/include -o $@

$(DEPDIR):
	@mkdir -p $@

$(DEPFILES):

clean:
	$(RM) $(OBJS) $(EXEC) $(DEPDIR)

-include $(wildcard $(DEPFILES))
