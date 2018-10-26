#include "gpu_translate.h"

#include <stdio.h>
//#include <assert.h>
//#include <cuda.h>

#include <stdlib.h>
//#include <argp.h>
//#include <libxml/xmlreader.h>
//#include <libxml/tree.h>
//#include <libxml/parser.h>
//#include <libxml/xpath.h>
//#include <libxml/xpathInternals.h>
#include <glib.h>  // for  using GList

//#include "../src/xpu.h"
//#include "../src/xpathConsts.h"
#include "../src/material.h"

#include "finite-difference.h"

// updates the material properties table
extern int GT_UpdateMaterialTable(float dt  )
{
    float tmp[MAX_SIZE_MATERIAL_TABLE];
    int i;
    for(i=0;i<MAX_SIZE_MATERIAL_TABLE;i++)
    {
        tmp[i] = 1.0f;
    }
    for(i=0;i<MAX_SIZE_MATERIAL_TABLE;i++)
    {
        float val = MATRL_Er(i);
        if( val < 0)
            break;
        tmp[i] = val;
    }
    FD_UpdateGa(tmp, MAX_SIZE_MATERIAL_TABLE);
    
    
//extern float MATRL_Ur(int index);
//extern float MATRL_Cond(int index);
//extern float MATRL_Sus(int index);
    
    return(0);
        

}
