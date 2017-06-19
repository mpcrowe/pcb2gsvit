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


%.xnet: $(PROJECT).sch
	$(NETLIST) -g xmlnet $(PROJECT).sch -o $*.xnet

xmlnetlist: $(PROJECT).xnet

schematicBom.xml: $(PROJECT).sch Makefile
	@$(NETLIST) -g xmlbom $< -o schematicBom.xml
#	@xmlstarlet ed --subnode "/bom" --type elem -n board_info -v "" schematicBomt1.xml > schematicBomt2.xml
#	@xmlstarlet ed --subnode "/bom/board_info" --type elem -n part_number -v "`$(SCRIPT_PATH)/pcbPartGet.sh $(PROJECT).pcb`" schematicBomt2.xml > schematicBomt3.xml
#	@xmlstarlet ed --subnode "/bom/board_info" --type elem -n build_type -v "x6-1" schematicBomt3.xml > schematicBomt4.xml
#	@xmlstarlet ed --subnode "/bom/board_info" --type elem -n build_date -v "20140307" schematicBomt4.xml > schematicBom.xml
#	@rm schematicBomt1.xml schematicBomt2.xml schematicBomt3.xml schematicBomt4.xml

remake_bom:
	$(eval REALNAME := $(shell $(SCRIPT_PATH)/bomName.sh `$(SCRIPT_PATH)/pcbPartGet.sh $(PROJECT).pcb`_$(BUILD_NAME)))
	echo $(REALNAME)
	$(SCRIPT_PATH)/bom2build.sh $(REALNAME) tempbom.xml
	$(SCRIPT_PATH)/xbom2pdf2.sh tempbom.xml $(REALNAME)
	rm tempbom.xml

# use this when getting a copy of the Bom from the sql server
bom: remake_bom

# use this when updating the bom on the sql server	
enter_bom: schematicBom.xml filter.xsl
	$(SCRIPT_PATH)/bomFilter.sh filter.xsl schematicBom.xml `$(SCRIPT_PATH)/pcbPartGet.sh $(PROJECT).pcb` $(BOM_REV) > $(TEMP_PATH)/xbom__temp2.xml
	$(SCRIPT_PATH)/enterBom2.sh $(TEMP_PATH)/xbom__temp2.xml 
	rm $(TEMP_PATH)/xbom__temp2.xml



updateschematic: $(PROJECT).tsch
	rm *.tsch

%.tsch: %.xbom
	$(SCRIPT_PATH)/mergeXbomWithSch.sh $*.sch $@ $*.xbom 
	cp $@ $*.sch
	
%.pdf: %.sch
	$(GSCHEM) -p -o $*.ps -s /usr/share/gEDA/scheme/print.scm $<
	ps2pdf $*.ps
	rm $*.ps
	
schpkg.pdf: $(PROJECT).pdf

%.xbom: %.sch
	# enter temp bom into database
	$(NETLIST) -g xmlbom $< -o $(TEMP_PATH)/xbom_$*_temp.xml
	$(SCRIPT_PATH)/enterBom.sh $(TEMP_PATH)/xbom_$*_temp.xml $*-$(DATE)-temp
	rm $(TEMP_PATH)/xbom_$*_temp.xml
	# generate xbom from database
	$(SCRIPT_PATH)/mysql2xbom.sh $*-$(DATE)-temp $@
	#remove temparary bom from database
	$(SCRIPT_PATH)/deleteBom.sh $*-$(DATE)-temp


# make spice circuit of design	
%.cir: %.tsch
	$(NETLIST) -g spice-sdb $< -o $@

newpcb:
	rm -f $(PROJECT).pcb $(PROJECT).net
	$(BIN_PATH)/gsch2pcb --skip-m4 --use-files --elements-dir $(PCB_FOOTPRINT_PATH)/footprints $(PROJECT).sch 

%.updatepcb:
	$(BIN_PATH)/gsch2pcb -s --skip-m4 --use-files --elements-dir $(PCB_FOOTPRINT_PATH)/footprints $*.sch 

updatepcb:
	$(BIN_PATH)/gsch2pcb -s --skip-m4 --use-files --elements-dir $(PCB_FOOTPRINT_PATH)/footprints $(PROJECT).sch 

pcb:
	$(PCB) $(PROJECT).pcb

centroid: $(PROJECT).xy
	
nelma: $(PROJECT).nelma.em $(PROJECT).xy

%.xy: %.pcb
	$(PCB)  -x bom --bomfile /dev/null --xyfile $@ $<
	
workTemp/$(PROJECT).xy: $(PROJECT).pcb workTemp
	$(PCB)  -x bom --bomfile /dev/null --xyfile $@ $<

%.nelma.em: $(PROJECT).pcb
	/home/crowe/temp/pcb/pcb/src/pcb -x nelma --dpi 200 $(PROJECT).pcb
	rm *.save


%-assembly.pdf : $(PROJECT).pcb
	mkdir -p pstemp
	pcb -x ps --psfile pstemp/$(PROJECT) --media Letter --fill-page --psfade 0.2 --multi-file --outline --auto-mirror --align-marks $(PROJECT).pcb
	cat ./pstemp/$(PROJECT).topassembly.ps ./pstemp/$(PROJECT).bottomassembly.ps > ./pstemp/$(PROJECT)-assembly.ps
	ps2pdf ./pstemp/$(PROJECT)-assembly.ps $(PROJECT)-assembly.pdf
	rm -rf pstemp


assembly: $(PROJECT)-assembly.pdf

one2one-assembly:
	mkdir -p pstemp
	pcb -x ps --psfile pstemp/$(PROJECT) --media Letter --psfade 0.2 --multi-file --outline --auto-mirror --align-marks $(PROJECT).pcb
	cat ./pstemp/$(PROJECT).topassembly.ps ./pstemp/$(PROJECT).bottomassembly.ps > ./pstemp/$(PROJECT)-assembly.ps
	ps2pdf ./pstemp/$(PROJECT)-assembly.ps $(PROJECT)-one2one-assembly.pdf
	rm -rf pstemp


build_pkg:  $(PROJECT).xy $(PROJECT)-assembly.pdf 
	$(eval ZIPARTWORK := `$(SCRIPT_PATH)/pcbPartGet.sh $(PROJECT).pcb`_`$(SCRIPT_PATH)/pcbDateGet.sh $(PROJECT).pcb`.zip)
	$(eval REALNAME := $(shell $(SCRIPT_PATH)/bomName.sh `$(SCRIPT_PATH)/pcbPartGet.sh $(PROJECT).pcb`_$(BUILD_NAME)))
	mkdir -p buildpkg
	cp $(PROJECT).xy buildpkg/
	cp $(PROJECT)-assembly.pdf buildpkg/
	cp artwork/$(ZIPARTWORK) buildpkg/
	cp bom_$(REALNAME).pdf buildpkg/

workTemp:
	mkdir workTemp

# most of these dependancies are used to create a g-code file to generate a board test fixture
workTemp/drill.xml: $(PROJECT).pcb workTemp
#	rm -rf workTemp
#	mkdir workTemp
	pcb -x gerber $(PROJECT).pcb
	rm *.gbr
	$(SCRIPT_PATH)/drillToXml.sh $(PROJECT).plated-drill.cnc $@

workTemp/outline.xml: $(PROJECT).pcb workTemp
	pcb -x gerber $(PROJECT).pcb
	mv $(PROJECT).group4.gbr $(PROJECT).outline.tmp
	rm *.gbr *.cnc
	mv $(PROJECT).outline.tmp $(PROJECT).outline.gbr	
	$(SCRIPT_PATH)/gerbToXml.sh $(PROJECT).outline.gbr $@
	
workTemp/outline.ngc: workTemp/outline.xml $(SCRIPT_PATH)/xslt/gerbx2ngc.xsl
	xsltproc $(SCRIPT_PATH)/xslt/gerbx2ngc.xsl workTemp/outline.xml > $@

workTemp/tpcentroids.xml: workTemp/$(PROJECT).xy
	$(SCRIPT_PATH)/extractTestPointCentroids.sh $< $@

rootcnc.xml: drill.xml tpcentroids.xml
	cp $(SCRIPT_PATH)/xslt/rootcnc.xml .
cnc.xml: rootcnc.xml
	xsltproc  $(SCRIPT_PATH)/xslt/rootxsl.xsl rootcnc.xml > cnc.xml
	
$(PROJECT).ngc: cnc.xml
	xsltproc $(SCRIPT_PATH)/xslt/makeCnc.xsl  $< >  $@
$(PROJECT)-plastic.ngc: cnc.xml
	xsltproc $(SCRIPT_PATH)/xslt/makeCncPlasticCase.xsl  $< >  $@

#end gcode generation depends

# used to create opeonscad code which compiles to 3-d model
workTemp/$(PROJECT).xxy: workTemp/$(PROJECT).xy
	$(SCRIPT_PATH)/centroids2xcent.sh $< $@

workTemp/out_path.xml:workTemp/outline.xml
	xsltproc $(SCRIPT_PATH)/xslt/gerbx2outline.xsl workTemp/outline.xml | xsltproc $(SCRIPT_PATH)/xslt/gerbx2outline2.xsl - > $@

workTemp/bomFlat.xml: schematicBom.xml
	xsltproc $(SCRIPT_PATH)/xslt/bomFilters/split.xsl schematicBom.xml | xsltproc $(SCRIPT_PATH)/xslt/bomFilters/split2.xsl - > $@

workTemp/$(PROJECT).xos: workTemp/$(PROJECT).xxy workTemp/drill.xml workTemp/out_path.xml workTemp/bomFlat.xml three_d_info.xslt
	echo '<xml/>' | xsltproc three_d_info.xslt - > $@
	$(SCRIPT_PATH)/lines2path --out $@ $@

$(PROJECT).scad: workTemp/$(PROJECT).xos $(SCRIPT_PATH)/xslt/openscad.xsl workTemp
	xsltproc $(SCRIPT_PATH)/xslt/openscad.xsl $< >$@
#end 3d-model depends
	
sch:
	$(GSCHEM) $(PROJECT).sch

%.drc: %.sch
	gnetlist -g drc2 -o $@ $<

check: $(PROJECT).drc
	cat $(PROJECT).drc
	
dist: $(SRCS) $(AUX)
#       echo gcdc_tool-`sed -e '/version_string/!d' -e 's/[^0-9.]*\([0-9.]*\).*/\1/' -e q version.c` > .fname
	echo swem-r`svnversion .` > .fname
	-rm -rf `cat .fname`
	mkdir `cat .fname`
	ln $(SRCS) $(AUX) `cat .fname`
	tar chjf `cat .fname`.tar.bz2 `cat .fname`
	-rm -rf `cat .fname` .fname
