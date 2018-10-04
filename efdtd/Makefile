# Authors: mcrowe@gcdataconcepts.com
#
# DESCRIPTION: 
#
# Changes:
#

PROJECT = efdtd
#DEBUG = --debug
CC = gcc
#NVCC = /usr/lib/nvidia-cuda-toolkit/bin/nvcc
#NVCC = /usr/local/cuda-9.1/bin/nvcc
NVCC = nvcc
NVCCFLAGS = -D_FORCE_INLINES -O2 -use_fast_math -DUCUDA -rdc=false
NVCC_OBJ = finite-difference.o

OBJECTS =  #gcdcInterface.o ihex.o version.o firmwareUpdater.o
SRCS = $(PROJECT).c xpu.c frect.c material.c layer.c med_vect.c file_edit.c
AUX = Makefile
PREMADES = 
#GWY_CFLAGS=-I/usr/include/gwyddion -I/usr/lib//gwyddion/include
#GWY_LIBS=-L$/usr/lib -lgwyapp2 -lgwymodule2 -lgwydgets2 -lgwydraw2 -lgwyprocess2 -lgwyddion2
LIBS=-lm -lglib-2.0 -L/usr/lib/x86_64-linux-gnu -lcudart -lgcc_s -lrt -lcudart -lstdc++
# $(GWY_LIBS) -lgomp
prefix=/usr/local
bindir=$(prefix)/bin
INSTALL = /usr/bin/install  

UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
# do something Linux-y
LIBUSB_INCLUDE_PATH=/usr/include
CFLAGS =  -fopenmp -I/usr/include/libxml2 -I$(LIBUSB_INCLUDE_PATH) -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include $(GWY_CFLAGS) -Wall
ALIBS = -L$(prefix)/lib $(LIBS) -Wl,-rpath=$(prefix)/lib
ALIBS_XML = -lxml2
else
# windows
OS_EXTENSION=.exe
LIBUSB_INCLUDE_PATH=/usr/local/include
CFLAGS = -I/usr/include -I. -I$(LIBUSB_INCLUDE_PATH)  -Wall
OBJECTS += windows.o
ALIBS_XML = /usr/local/lib/libxml2.dll
endif

ALINK = $(CC)  $(ALFLAGS)



all: $(PROJECT)

test: $(PROJECT)
	./$(PROJECT) ../test1/boardInfo.xml

#docs:
#	doxygen Doxyfile

clean:
	rm -rf $(PROJECT)$(OS_EXTENSION) $(PROJECT2)$(OS_EXTENSION) *~ $(PROJECT).o $(OBJECTS) $(NVCC_OBJ) doc  testbed trackball.o

$(PROJECT)$(OS_EXTENSION): $(PROJECT).o $(PREMADES) $(OBJECTS) $(NVCC_OBJ)
	@echo linking
	@$(ALINK) -o $@ $(PROJECT).o $(OBJECTS) $(NVCC_OBJ) $(ALIBS) $(ALIBS_XML)
	
%.o: %.c
	@echo compiling $<
	@$(CC) -c $(CFLAGS) $(DEBUG) $<

%.o: %.cu
	@echo compiling $< with NVCC
	@$(NVCC) -o $@ -c $< $(NVCCFLAGS)

testbed: testbed.c trackball.o
	@$(CC) $< -o $@ -lGLU -lGL -lXmu -lXi -lXext -lX11 -lm -lglut trackball.o

install: all
	$(INSTALL) $(PROJECT2) $(bindir)/$(PROJECT2)
	#cp x6-1analysis.xml /usr/etc

dist: $(SRCS) $(AUX)
	echo gcdc_tool-`sed \
		-e '/version_string/!d' \
                -e 's/[^0-9.]*\([0-9.]*\).*/\1/' \
                -e q version.c` > .fname
	-rm -rf `cat .fname`
	mkdir `cat .fname`
	ln $(SRCS) $(AUX) `cat .fname`
	tar chjf `cat .fname`.tar.bz2 `cat .fname`
	-rm -rf `cat .fname` .fname


