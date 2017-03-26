# Authors: mcrowe@gcdataconcepts.com
#
# DESCRIPTION: 
#
# Changes:
#

PROJECT = pcb2gsvit
#DEBUG = --debug
CC = gcc
OBJECTS = #gcdcInterface.o ihex.o version.o firmwareUpdater.o
SRCS = $(PROJECT).c
AUX = Makefile
PREMADES = 
LIBS=-lm
prefix=/usr/local
bindir=$(prefix)/bin
INSTALL = /usr/bin/install  

UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
# do something Linux-y
LIBUSB_INCLUDE_PATH=/usr/include
CFLAGS =  -I/usr/include/libxml2 -I$(LIBUSB_INCLUDE_PATH) -Wall
ALIBS = -L$(prefix)/lib $(LIBS) -Wl,-rpath=$(prefix)/lib
ALIBS_XML = -lxml2
else
# windows
OS_EXTENSION=.exe
LIBUSB_INCLUDE_PATH=/usr/local/include
CFLAGS = -I/usr/include -I. -I$(LIBUSB_INCLUDE_PATH) -Wall
OBJECTS += windows.o
ALIBS_XML = /usr/local/lib/libxml2.dll
endif

ALINK = $(CC)  $(ALFLAGS)



all: $(PROJECT)


#docs:
#	doxygen Doxyfile

clean:
	rm -rf $(PROJECT)$(OS_EXTENSION) $(PROJECT2)$(OS_EXTENSION) *~ $(PROJECT).o $(OBJECTS) doc

$(PROJECT)$(OS_EXTENSION): $(PROJECT).o $(PREMADES) $(OBJECTS)
	@$(ALINK) -o $@ $(PROJECT).o $(OBJECTS) $(ALIBS) $(ALIBS_XML)
	
%.o: %.c
	@$(CC) -c $(CFLAGS) $(DEBUG) $<

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


