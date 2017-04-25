#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#define PNG_DEBUG 3
#include <png.h>
#include "layer.h"

void abort_(const char * s, ...);
void abort_(const char * s, ...)
{
	va_list args;
	va_start(args, s);
	vfprintf(stderr, s, args);
	fprintf(stderr, "\n");
	va_end(args);
//	abort();
}

int x, y;

int width, height;
png_byte color_type;
png_byte bit_depth;

png_structp png_ptr;
png_infop info_ptr;
int number_of_passes;
png_bytep * row_pointers;


int LAYER_ReadPng(char* file_name)
{
	unsigned char header[8];    // 8 is the maximum size that can be checked

	// open file and test for it being a png
	FILE *fp = fopen(file_name, "rb");
	if (!fp)
	{
//		fprintf(stderr,"[read_png_file] File %s could not be opened for reading", file_name);
		return(-1);
	}
			
	fread(header, 1, 8, fp);
	if( png_sig_cmp(header, 0, 8) )
	{
//		fprintf(stderr, "[read_png_file] File %s is not recognized as a PNG file", file_name);
		return(-2);
	}

	// initialize stuff
	png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	if(!png_ptr)
	{
		abort_("[read_png_file] png_create_read_struct failed");
		return(-3);
	}

	info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr)
	{
		abort_("[read_png_file] png_create_info_struct failed");
		return(-4);
	}
	
	if (setjmp(png_jmpbuf(png_ptr)))
	{
		abort_("[read_png_file] Error during init_io");
		return(-5);
	}

	png_init_io(png_ptr, fp);
	png_set_sig_bytes(png_ptr, 8);

	png_read_info(png_ptr, info_ptr);

	width = png_get_image_width(png_ptr, info_ptr);
	height = png_get_image_height(png_ptr, info_ptr);
	color_type = png_get_color_type(png_ptr, info_ptr);
	bit_depth = png_get_bit_depth(png_ptr, info_ptr);

	number_of_passes = png_set_interlace_handling(png_ptr);
	png_read_update_info(png_ptr, info_ptr);


	// read file 
	if (setjmp(png_jmpbuf(png_ptr)))
	{
		abort_("[read_png_file] Error during read_image");
		return(-6);
	}	

	row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
	for (y=0; y<height; y++)
		row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png_ptr,info_ptr));

	png_read_image(png_ptr, row_pointers);

	fclose(fp);
	return(0);
}
