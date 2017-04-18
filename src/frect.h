#ifndef SV_FCUBE
#define SV_FCUBE

/*  frect.h : 
 *  floating point 2D data representation
 */

#include <glib.h>
#include <stdint.h>
#include "material.h"

typedef struct {
   gint xres;
   gint yres;
   gdouble xreal;
   gdouble yreal;
   indexSize_t **data;
} fRect;

fRect* FRECT_New(gint xres, gint yres, gdouble xreal, gdouble yreal, gboolean nullme);
fRect* FRECT_NewAlike(fRect *frect, gboolean nullme);
fRect* FRECT_Copy(fRect* dest, fRect* src);
fRect* FRECT_Clone(fRect* src);
void FRECT_Free(fRect *frect);
void FRECT_Fill(fRect *frect, indexSize_t material_index);

#endif
