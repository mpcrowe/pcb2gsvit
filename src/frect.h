#ifndef SV_FCUBE
#define SV_FCUBE

/*  frect.h : 
 *  floating point 2D data representation
 */

#include <glib.h>

typedef struct {
   gint xres;
   gint yres;
   gdouble xreal;
   gdouble yreal;
   gfloat **data;
} fRect;

fRect* FRECT_New(gint xres, gint yres, gdouble xreal, gdouble yreal, gboolean nullme);
fRect* FRECT_FewAlike(fRect *frect, gboolean nullme);
fRect* FRECT_FewCopy(fRect* dest, fRect src);
fRect* FRECT_FewClone(fRect src);
void FRECT_Free            (fRect *frect);
void FRECT_Fill            (fRect *frect, gfloat value);
void FRECT_Add             (fRect *frect, gfloat value);
void FRECT_Multiply        (fRect *frect, gfloat value);

#endif
