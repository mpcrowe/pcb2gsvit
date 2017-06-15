# Authors: mcrowe@gcdataconcepts.com
# Copyright (C) 2007-2008,  Gulf Coast Data Concepts, LLC
#
# DESCRIPTION: 
#
# Changes:
#
# SVN configuration information below. Do not modify
# $Id:$
#
PROJECT = test2
BUILD_NAME=min
BOM_REV=Rev_A

SRCS =  $(PROJECT).sch $(PROJECT).pcb filter.xsl
SCH_SRCS =  $(PROJECT).sch 
AUX = Makefile attribs rules.txt

BIN_PATH=/usr/bin
LOCAL_BIN_PATH=~/temp/pcb/pcb/src
TRAGESYM = $(BIN_PATH)/tragesym
PCB = $(LOCAL_BIN_PATH)/pcb
GSCHEM = $(BIN_PATH)/gschem
ATTRIB = $(BIN_PATH)/gattrib
NETLIST = $(BIN_PATH)/gnetlist
REFDES_RENUM = $(BIN_PATH)/refdes_renum
DATE =  $(shell date +%Y%m%d)

GEDA_SHARED_PATH=/opt/geda_shared
SCRIPT_PATH = $(GEDA_SHARED_PATH)/bin
PCB_FOOTPRINT_PATH = $(GEDA_SHARED_PATH)

TEMP_PATH=/tmp

all:	$(PROJECT).sym

distclean: clean
	svn st | grep ^\? | awk '{print $2}' | xargs -r rm -rf
	
clean:
	rm -rf *~ *- *.save *.backup *.log *.bak* *.gbr *.xy *.xxy *.xos *.scad workTemp $(PROJECT).new.pcb $(PROJECT).cmd $(PROJECT).net $(PROJECT)*.png $(PROJECT).nelma.em $(PROJECT).nelma.xem

embedd:
	gschlas -e $(SCH_SRCS)

unembedd:
	gschlas -u $(SCH_SRCS)
	
#$(PROJECT).sym: $(PROJECT).src
#	$(TRAGESYM)  $< $(PROJECT).sym
	

install: $(PROJECT).sym


renumber:
	$(REFDES_RENUM) --gentle $(PROJECT).sch

%.forcerenum:
	$(REFDES_RENUM) --force $*.sch
	
number:
	$(REFDES_RENUM)  --gentle $(PROJECT).sch 

attrib:
	$(ATTRIB) $(PROJECT).sch


newpcb:
	rm -f $(PROJECT).pcb $(PROJECT).net
	$(BIN_PATH)/gsch2pcb --skip-m4 --use-files --elements-dir $(PCB_FOOTPRINT_PATH)/footprints $(PROJECT).sch 

%.updatepcb:
	$(BIN_PATH)/gsch2pcb -s --skip-m4 --use-files --elements-dir $(PCB_FOOTPRINT_PATH)/footprints $*.sch 

updatepcb:
	$(BIN_PATH)/gsch2pcb -s --skip-m4 --use-files --elements-dir $(PCB_FOOTPRINT_PATH)/footprints $(PROJECT).sch 

pcb:
	$(PCB) $(PROJECT).pcb

nelma: $(PROJECT).nelma.em $(PROJECT).xy


%.nelma.em: $(PROJECT).pcb
	/home/crowe/temp/pcb/pcb/src/pcb -x nelma $(PROJECT).pcb
	rm *.save

sch:
	$(GSCHEM) $(PROJECT).sch

	
dist: $(SRCS) $(AUX)
#       echo gcdc_tool-`sed -e '/version_string/!d' -e 's/[^0-9.]*\([0-9.]*\).*/\1/' -e q version.c` > .fname
	echo swem-r`svnversion .` > .fname
	-rm -rf `cat .fname`
	mkdir `cat .fname`
	ln $(SRCS) $(AUX) `cat .fname`
	tar chjf `cat .fname`.tar.bz2 `cat .fname`
	-rm -rf `cat .fname` .fname
