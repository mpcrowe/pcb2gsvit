#ifndef _MATERIAL
#define _MATERIAL 1
#include <libxml/xmlstring.h>
#include <libxml/xpath.h>

#include <stdint.h>

typedef struct
{
  xmlChar* name;
  xmlChar* defaultThickness;
  gfloat er;
  gfloat conductivity;
  gfloat ur;
  gfloat sus;
} material_t;
            
extern material_t* materialTable;

typedef uint16_t indexSize_t;

void MATRL_Init(indexSize_t tableSize);
int MATRL_CreateTableFromNodeSet(xmlNodeSetPtr xnsMaterials);
void MATRL_Dump(material_t* mat);
void MATRL_DumpAll(void);
int MATRL_GetIndex(char* name);
int MATRL_StringToCounts(char* pString, double metersPerPixel);
double MATRL_ScaleToMeters(double val, char* units);


extern float MATRL_Er(int index);
extern float MATRL_Ur(int index);
extern float MATRL_Cond(int index);
extern float MATRL_Sus(int index);

#define MATRL_DefaultThickness(index) (materialTable[index].defaultThickness)
        
#endif