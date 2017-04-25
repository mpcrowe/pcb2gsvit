#ifndef _MATERIAL
#define MATERIAL 1
#include <libxml/xmlstring.h>
#include <libxml/xpath.h>

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

// these need to be fast lookup macros as they are used over and over again
// when constructing the z-axix line of data
#define MATRL_Er(index) (materialTable[index].er)
#define MATRL_Cond(index) (materialTable[index].conductivity)
        
#endif