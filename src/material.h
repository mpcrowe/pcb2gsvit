#ifndef _MATERIAL
#define MATERIAL 1

#include <stdint.h>

typedef struct
{
  char* name;
  char* defaultThickness;
  gfloat er;
  gfloat conductivity;
} material_t;
            
extern material_t* materialTable;

typedef uint16_t indexSize_t;

void MATRL_Init(indexSize_t tableSize);
void MATRL_CreateTableFromNodeSet(xmlNodeSetPtr xnsMaterials);
        
#endif