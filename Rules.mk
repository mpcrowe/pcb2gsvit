
CROSS_COMPILE = /opt/arm-tools/gcc-arm-none-eabi-4_6-2012q2/bin/arm-none-eabi-

# Compilation tools
CC = $(CROSS_COMPILE)gcc
LD = $(CROSS_COMPILE)ld 
SIZE = $(CROSS_COMPILE)size
STRIP = $(CROSS_COMPILE)strip
OBJCOPY = $(CROSS_COMPILE)objcopy
GDB = $(CROSS_COMPILE)gdb
NM = $(CROSS_COMPILE)nm  
AR = $(CROSS_COMPILE)ar

# Libraries
LIBRARIES = /home/crowe/mc/armSource/libraries

# Chip library directory
CHIP_LIB = $(LIBRARIES)/libchip_sam3s

# Board library directory
#BOARD_LIB = $(LIBRARIES)/libboard_sam3s-ek
#BOARD_LIB = $(LIBRARIES)/libboard_arm_demo
GENERIC_BOARD_LIB = $(LIBRARIES)/libboard_generic

# GCDC library directory
GCDC_LIB = $(LIBRARIES)/libgcdc

# USB library directory
USB_LIB = $(LIBRARIES)/libusb

# GCDC library directory
GCDC_LIB = $(LIBRARIES)/libgcdc

MEMORY_LIB = $(LIBRARIES)/libmemories

FATFS_LIB = $(LIBRARIES)/libfat
