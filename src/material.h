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


// these need to be fast lookup macros as they are used over and over again
// when constructing the z-axix line of data
//#define MATRL_Er(index) (materialTable[index].er)
extern float MATRL_Er(int index);
#define MATRL_Ur(index) (materialTable[index].ur)
//#define MATRL_Cond(index) (materialTable[index].conductivity)
extern float MATRL_Cond(int index);
#define MATRL_Sus(index) (materialTable[index].sus)
#define MATRL_DefaultThickness(index) (materialTable[index].defaultThickness)
        
#endif