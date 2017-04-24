#ifndef _MATERIAL
#define MATERIAL 1

#include <stdint.h>

typedef struct
{
  xmlChar* name;
  xmlChar* defaultThickness;
  gfloat er;
  gfloat conductivity;
} material_t;
            
extern material_t* materialTable;

typedef uint16_t indexSize_t;

void MATRL_Init(indexSize_t tableSize);
int MATRL_CreateTableFromNodeSet(xmlNodeSetPtr xnsMaterials);
void MATRL_Dump(material_t* mat);
void MATRL_DumpAll(void);
        
#endif