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
	png_colorp palette;
	int num_palette;
};

struct image img;
void LAYER_Dump(void );
void LAYER_PaletteDump(void);

png_structp png_ptr;
png_infop info_ptr;
#undef USE_LOCAL_MALLOC
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

#ifdef USE_LOCAL_MALLOC
	png_read_info(png_ptr, info_ptr);
	img.width = png_get_image_width(png_ptr,info_ptr);
	img.height = png_get_image_height(png_ptr, info_ptr);
	img.bit_depth = png_get_bit_depth(png_ptr, info_ptr);
	int y;
	img.row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * img.height);
	for(y = 0; y < img.height; y++)
	{
		img.row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png_ptr,info_ptr));
	}
	png_read_image(png_ptr, img.row_pointers);
#else
	png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

	img.width = png_get_image_width(png_ptr,info_ptr);
	img.height = png_get_image_height(png_ptr, info_ptr);
	img.bit_depth = png_get_bit_depth(png_ptr, info_ptr);
#endif
	color_type = png_get_color_type(png_ptr, info_ptr);
	interlace_type = png_get_interlace_type(png_ptr, info_ptr);
	compression_type = png_get_compression_type(png_ptr, info_ptr);
	filter_method = png_get_filter_type(png_ptr, info_ptr);

	fprintf(stdout, "w: %d  h: %d bit_depth %d\n", img.width, img.height, img.bit_depth);
#ifndef USE_LOCAL_MALLOC
	img.row_pointers = png_get_rows(png_ptr, info_ptr);
#endif
	
	png_get_PLTE(png_ptr, info_ptr, &img.palette, &img.num_palette);
//	LAYER_PaletteDump();	
//	LAYER_Dump();

//	png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
	fclose(fp);
	return(0);
}

void LAYER_Dump(void)
{
	int x;
	int y;
	fprintf(stdout,"%s %d %d\n", __FUNCTION__, img.width, img.height);
//	for(y=0; y<26; y++)
	for(y=0; y<img.height; y++)
	{
//		fprintf(stdout, "rowp %p \n", img.row_pointers);
		png_bytep row = img.row_pointers[y];
//		fprintf(stdout, "row %p \n", row);
//		for(x=0;x<100;x++)
		for(x=0;x<img.width/2;x++)
		{
			fprintf(stdout, "%x ", row[x]);
		}
		fprintf(stdout,"\n");
	}
}
void LAYER_PaletteDump(void)
{
	int i;
	fprintf(stdout, "\nnum palette %d\n",img.num_palette);
	for(i=0;i<img.num_palette;i++)
	{
			fprintf(stdout, "%x, %x %x\n", img.palette[i].red, img.palette[i].green, img.palette[i].blue);
	}
	fprintf(stdout, "\n");
}

void LAYER_Done()
{
	png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
	png_free_data(png_ptr, info_ptr,PNG_FREE_ALL, -1);
}


void LAYER_ProcessOutline(fRect* dest, indexSize_t matrlIndex)
{
	int borderIndex = 1; 	// fixme, need better way 
	int backgroundIndex = 5; 	// fixme, need better way
	int x;
	int y;
	fprintf(stdout, "\nProcessing Outline with border:%d and background: %d\n", borderIndex, backgroundIndex);
	fprintf(stdout, "processing Outline with x:%d y:%d mat index:%d\n", dest->xres, dest->yres, matrlIndex);
	
	if(dest->xres != img.width)
	{
		fprintf(stderr, "ERROR, xres != width %d != %d\n", dest->xres, img.width);
		return;
	}	
	if(dest->yres != img.height)
	{
		fprintf(stderr, "ERROR, yres != height %d != %d\n", dest->yres, img.height);
		return;
	}
	FRECT_Fill(dest, matrlIndex);

	// top down search for border
	fprintf(stdout, "\ttop down search\n");
	int dest_x = 0;
	for(x=0, dest_x=0; x< img.width/2; x++)
	{
		for(y=0;y<img.height;y++)
		{ 
			uint8_t pngDatum = img.row_pointers[y][x];
			if( ((pngDatum >>4) != backgroundIndex) )
				break;
			dest->data[dest_x][y] = 0;
			if(dest_x < dest->xres)
			{
				if( ((pngDatum & 0x0f) != backgroundIndex) )
					break;
				dest->data[dest_x+1][y] = 0;
			}
		}
		dest_x += 2;
	}

	// bottom up search for boarder
	fprintf(stdout, "\tbottom up search\n");
	dest_x = 0;
	for(x=0, dest_x=0; x< img.width/2; x++)
	{
		for(y=img.height-1;y>=0;y--)
		{ 
			if( dest->data[dest_x][y] == 0)
			{
//				fprintf(stdout,"preexisting found at x:%d y:%d\n", x,y);
				break;
			}

			uint8_t pngDatum = img.row_pointers[y][x];
			if( ((pngDatum >>4) != backgroundIndex) )
			{
//				fprintf(stdout,"even found at x:%d y:%d %x\n", x,y, pngDatum);
				break;
			}
			dest->data[dest_x][y] = 0;
			if(dest_x < dest->xres)
			{
				if( ((pngDatum & 0x0f) != backgroundIndex) )
				{
//					fprintf(stdout,"odd found at x:%d y:%d %x\n", x,y, pngDatum);
					break;
				}
				dest->data[dest_x+1][y] = 0;
			}
		}
		dest_x +=2;
	}

	// left to right search for boarder
	fprintf(stdout, "\tleft to right search\n");
	dest_x = 0;
	for(y=0;y<img.height; y++)
	{
		for(x=0, dest_x=0; x< img.width/2; x++)
		{ 
			uint8_t pngDatum = img.row_pointers[y][x];
			if( ((pngDatum >>4) != backgroundIndex) )
			{
//				fprintf(stdout,"even found at x:%d y:%d %x\n", x,y, pngDatum);
				break;
			}
			dest->data[dest_x++][y] = 0;
			if(dest_x < dest->xres)
			{ // odd
				if( ((pngDatum & 0x0f) != backgroundIndex) )
				{
//					fprintf(stdout,"odd found at x:%d y:%d %x\n", x,y, pngDatum);
					break;
				}
				dest->data[dest_x++][y] = 0;
			}
		}
	}

	// right to left search for boarder
	fprintf(stdout, "\tright to left search\n");
	dest_x = 0;
	for(y=0;y<img.height; y++)
	{
		for(x=img.width/2, dest_x=dest->xres-1; x>=0; x--)
		{ 
			uint8_t pngDatum = img.row_pointers[y][x];
			if( ((pngDatum >>4) != backgroundIndex) )
			{
//				fprintf(stdout,"even found at x:%d y:%d %x\n", x,y, pngDatum);
				break;
			}
			dest->data[dest_x--][y] = 0;
			if(dest_x < dest->xres)
			{ // odd
				if( ((pngDatum & 0x0f) != backgroundIndex) )
				{
//					fprintf(stdout,"odd found at x:%d y:%d %x\n", x,y, pngDatum);
					break;
				}
				dest->data[dest_x--][y] = 0;
			}
		}
	}
	
	fprintf(stdout, "processing Outline completed\n");

}


void LAYER_ProcessLayer(fRect* dest, indexSize_t matrlIndex)
{
	int borderIndex = 1; 	// fixme, need better way 
	int backgroundIndex = 5; 	// fixme, need better way
	int x;
	int y;
	fprintf(stdout, "\n%s with border:%d and background: %d\n", __FUNCTION__, borderIndex, backgroundIndex);
	fprintf(stdout, "%s with x:%d y:%d mat index:%d\n", __FUNCTION__, dest->xres, dest->yres, matrlIndex);
	
	if(dest->xres != img.width)
	{
		fprintf(stderr, "ERROR, xres != width %d != %d\n", dest->xres, img.width);
		return;
	}	
	if(dest->yres != img.height)
	{
		fprintf(stderr, "ERROR, yres != height %d != %d\n", dest->yres, img.height);
		return;
	}

	fprintf(stdout, "\tscaning image of layer\n");
	int dest_x = 0;
	for(x=0, dest_x=0; x< img.width/2; x++)
	{
		for(y=0;y<img.height;y++)
		{ 
			uint8_t pngDatum = img.row_pointers[y][x];
			if( ((pngDatum >>4) != backgroundIndex) )
			{
				dest->data[dest_x][y] = matrlIndex;
			}
			if(dest_x < dest->xres)
			{
				if( ((pngDatum & 0x0f) != backgroundIndex) )
				{
					dest->data[dest_x+1][y] = matrlIndex;
				}
			}
		}
		dest_x += 2;
	}
	fprintf(stdout, "%s completed\n", __FUNCTION__);
}
