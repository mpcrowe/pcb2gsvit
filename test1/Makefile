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
PROJECT = test1
BUILD_NAME=min
BOM_REV=Rev_A

SRCS =  $(PROJECT).sch $(PROJECT).pcb filter.xsl
SCH_SRCS =  $(PROJECT).sch 
AUX = Makefile attribs rules.txt

BIN_PATH=/usr/bin
#LOCAL_BIN_PATH=/home/crowe/temp/pcb/git-pcb/bert/pcb/src
LOCAL_BIN_PATH=../../pcb/src
PCB = $(LOCAL_BIN_PATH)/pcb
GSCHEM = $(BIN_PATH)/gschem
DATE =  $(shell date +%Y%m%d)

GEDA_SHARED_PATH=/opt/geda_shared
SCRIPT_PATH = $(GEDA_SHARED_PATH)/bin
PCB_FOOTPRINT_PATH = $(GEDA_SHARED_PATH)
PARAVIEW = ../../paraview/ParaView-5.5.0-Qt5-MPI-Linux-64bit/bin/paraview

TEMP_PATH=/tmp

all:	gsvit

distclean: clean
	svn st | grep ^\? | awk '{print $2}' | xargs -r rm -rf
	
clean:
	rm -rf *~ *- *.save *.backup *.log *.bak* *.gbr *.xy *.xxy *.xos *.scad workTemp $(PROJECT).new.pcb $(PROJECT).cmd $(PROJECT).net \
	$(PROJECT)*.png $(PROJECT).gsvit.em $(PROJECT).gsvit.xem ./vout xspectrum.txt medLin.bin out.gwy

pcb:
	$(PCB) $(PROJECT).pcb

centroid: $(PROJECT).xy
	
gsvit: $(PROJECT).gsvit.em $(PROJECT).xy

%.xy: %.pcb
	$(PCB)  -x bom --bomfile /dev/null --xyfile $@ $<
	

%.gsvit.em: $(PROJECT).pcb
	$(PCB) -x gsvit --dpi 200 $(PROJECT).pcb
	rm *.save

workTemp:
	mkdir workTemp
	
sch:
	$(GSCHEM) $(PROJECT).sch
	
test: gsvit
	../src/pcb2gsvit boardInfo.xml
	
gsvit_test: test
	mkdir -p vout
	gsvit first.par

test_display:
	$(PARAVIEW) temp.xdmf

display: medLin.bin

	$(PARAVIEW) test1.xdmf

dist: $(SRCS) $(AUX)
#       echo gcdc_tool-`sed -e '/version_string/!d' -e 's/[^0-9.]*\([0-9.]*\).*/\1/' -e q version.c` > .fname
	echo swem-r`svnversion .` > .fname
	-rm -rf `cat .fname`
	mkdir `cat .fname`
	ln $(SRCS) $(AUX) `cat .fname`
	tar chjf `cat .fname`.tar.bz2 `cat .fname`
	-rm -rf `cat .fname` .fname
