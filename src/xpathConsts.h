#ifndef _XPATH_CONSTS
#define _XPATH_CONSTS 1

#define XPATH_XEM_BOARD_DOC "/boardInformation"
#define XPATH_XEM_NAME "/boardInformation/gsvitExport/text()"
#define XPATH_XEM_OUTPUT_FILENAME "/boardInformation/gsvit/mediumLinearFilename/text()"
#define XPATH_XEM_MATERIALS "/boardInformation/materials/material"
#define XPATH_XEM_LAYERS "/boardInformation/boardStackup/layer"
#define XPATH_XEM_OUTLINE "/boardInformation/boardStackup/layer[name/text()='outline']/material/text()"

#define XPATH_NELMA_DRILLS "/gsvit/drills/drill"
#define XPATH_NELMA_WIDTH "/gsvit/space/width/text()"
#define XPATH_NELMA_HEIGHT "/gsvit/space/height/text()"
#define XPATH_NELMA_RES "/gsvit/space/resolution/text()"
#define XPATH_NELMA_RES_UNITS   "/gsvit/space/resolution/@units"

#endif
