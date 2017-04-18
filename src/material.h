#ifndef _MATERIAL
#define MATERIAL 1

#include <stdint.h>

typedef struct
{
  char* name;
  char* defaultThickness;
  gfloat er;
  gfloat conductivity;
} material;
            
extern struct material* materials;

typedef uint16_t indexSize_t;

void MATRL_Init(indexSize_t tableSize);

#endif