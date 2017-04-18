#ifndef SV_FCUBE
#define SV_FCUBE

#include <glib.h>
#include <stdint.h>
#include "material.h"

typedef struct {
   gint xres;
   gint yres;
   indexSize_t **data;
} fRect;

fRect* FRECT_New(gint xres, gint yres, gboolean nullme);
fRect* FRECT_NewAlike(fRect *frect, gboolean nullme);
fRect* FRECT_Copy(fRect* dest, fRect* src);
fRect* FRECT_Clone(fRect* src);
void FRECT_Free(fRect *frect);
void FRECT_Fill(fRect *frect, indexSize_t material_index);

#endif
