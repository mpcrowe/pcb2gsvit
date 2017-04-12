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
} PgFRect;

PgFRect *PGFrectNew(gint xres, gint yres, gdouble xreal, gdouble yreal, gboolean nullme);
PgFRect *PGFrectFewAlike   (PgFRect *frect, gboolean nullme);
void PGFrectFree            (PgFRect *frect);
void PGFrectFill            (PgFRect *frect, gfloat value);
void PGFrectAdd             (PgFRect *frect, gfloat value);
void PGFrectMultiply        (PgFRect *frect, gfloat value);

#endif
