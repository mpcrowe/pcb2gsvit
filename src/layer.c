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


int color_type;
//int bit_depth;
int interlace_type;
int compression_type;
int filter_method;

int number_of_passes;

struct image {
	int width;
	int height;
	int bit_depth;
	png_bytep * row_pointers;
};

struct image img;
void LAYER_Dump(void );
png_structp png_ptr;
png_infop info_ptr;

int LAYER_ReadPng(char* file_name)
{
	unsigned char header[8];    // 8 is the maximum size that can be checked

	// open file and test for it being a png
	FILE *fp = fopen(file_name, "rb");
	if (!fp)
	{
		abort_("[read_png_file] File %s could not be opened for reading", file_name);
		return(-1);
	}

	fread(header, 1, 8, fp);
	if( png_sig_cmp(header, 0, 8) )
	{
		abort_("[read_png_file] File %s is not recognized as a PNG file", file_name);
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
		png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
		abort_("[read_png_file] png_create_info_struct failed");
		return(-4);
	}

	if(setjmp(png_jmpbuf(png_ptr)))
	{	// Free all of the memory associated with the png_ptr and info_ptr */
		png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
		fclose(fp);

		abort_("[read_png_file] Error during init_io");
		return(-5);
	}
	png_init_io(png_ptr, fp);
	png_set_sig_bytes(png_ptr, 8);
	
	png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

	img.width = png_get_image_width(png_ptr,info_ptr);
	img.height = png_get_image_height(png_ptr, info_ptr);
	img.bit_depth = png_get_bit_depth(png_ptr, info_ptr);

	color_type = png_get_color_type(png_ptr, info_ptr);
	interlace_type = png_get_interlace_type(png_ptr, info_ptr);
	compression_type = png_get_compression_type(png_ptr, info_ptr);
	filter_method = png_get_filter_type(png_ptr, info_ptr);

	fprintf(stdout, "w: %d  h: %d bit_depth %d\n", img.width, img.height, img.bit_depth);
	img.row_pointers = png_get_rows(png_ptr, info_ptr);
	LAYER_Dump();

	png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
	fclose(fp);
	return(0);
}

void LAYER_Dump(void)
{
	int x;
	int y;
	for(x=0;x<100;x++)
	{
		for(y=0; y<26; y++)
		{
			fprintf(stdout, "%x ", img.row_pointers[x][y]);
		}
		fprintf(stdout,"\n");
	}
}

void LAYER_Done()
{
	png_free_data(png_ptr, info_ptr,PNG_FREE_ALL, -1);
}
