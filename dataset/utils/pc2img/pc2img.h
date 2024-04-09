//
//  pc2img.h
//  PointCloud2Image
//
//  Created by Theodora on 26/03/2024.
//

#ifndef pc2img_h
#define pc2img_h

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct __attribute__((__packed__))
{
    uint16_t    red;
    uint16_t    green;
    uint16_t    blue;
} LASPointExtra2_t;

typedef struct __attribute__((__packed__))
{
    char        fileSignature[4];
    uint16_t    fileSourceId;
    uint16_t    globalEncoding;
    uint32_t    projectId_GUIDData_1;
    uint16_t    projectId_GUIDData_2;
    uint16_t    projectId_GUIDData_3;
    uint8_t     projectId_GUIDData_4[8];
    uint8_t     versionMajor;
    uint8_t     versionMinor;
    char        systemIdentifier[32];
    char        generatingSoftware[32];
    uint16_t    fileCreationDayOfYear;
    uint16_t    fileCreationYear;
    uint16_t    headerSize;
    uint32_t    offsetToPointData;
    uint32_t    numberOfVaribleLenghtRecords;
    uint8_t     pointDataRecordFormat;
    uint16_t    pointDataRecordLength;
    uint32_t    legacyNumberOfPointRecords;
    uint32_t    legacyNumberOfPointByReturn[5];
    double      xScaleFactor;
    double      yScaleFactor;
    double      zScaleFactor;
    double      xOffset;
    double      yOffset;
    double      zOffset;
    double      maxX;
    double      minX;
    double      maxY;
    double      minY;
    double      maxZ;
    double      minZ;
    uint64_t    startOfWaveformDataPacketRecord;
    uint64_t    startOfFirstExtendedVariableLengthRecord;
    uint32_t    numberOfExtendedVariableLengthRecords;
    uint64_t    numberOfPointRecords;
    uint64_t    numberOfPointsbyReturn[15];

} LASPublicHeader_t;

typedef struct __attribute__((__packed__))
{
    int32_t     x;
    int32_t     y;
    int32_t     z;
    uint16_t    intensity;
    uint8_t     returnNumber: 3;
    uint8_t     numberOfReturnsOnGivenPulse: 3;
    uint8_t     scanDirectionFlag: 1;
    uint8_t     edgeOfFlightLine: 1;
    uint8_t     classification;
    int8_t      scanAngle;
    uint8_t     userData;
    uint16_t    pointSourceId;

} LASPointCommon012345_t;

typedef struct __attribute__((__packed__))
{   
    double       x;
    double       y;
    double       z;
    uint16_t    red;
    uint16_t    green;
    uint16_t    blue;
    uint8_t     intensity;
    uint32_t    pointId;
    bool        isEmpty;

} Point;

typedef struct __attribute__((__packed__)) {
   
    Point*       points;
    uint32_t     numberOfPoints;
    
} PointsArr;

typedef struct __attribute__((__packed__)) {
    
    float        width;
    float        height;
    uint32_t     numberOfPoints;
    
} BinaryHeader;

typedef struct __attribute__((__packed__)) {
   
    int         i;
    int         j;
    PointsArr pointsArr;
    
} Cell;

typedef struct __attribute__((__packed__)) {
   
    Cell        *cells;
    Cell        *overlappingCells;
    int         rows;
    int         colCells;
    int         colOvCells;
    
} CellArr;

typedef struct __attribute__((__packed__)) {
   
    PointsArr pointsArr;
    int nRows;
    int nCols;
    
} Image;

#define PIXEL_LIMIT_PER_DIM 256

CellArr initializeCells(LASPublicHeader_t *lasHeader, int cellSize);
CellArr assignPointsToCells(const char *lasFileName);
void calculateCellIndex(CellArr *cellsArr, Point currentPoint, LASPublicHeader_t *lasHeader, int cellSize);
Point calculateMin3D(PointsArr *points);
Point calculateMax3D(PointsArr *points);
Point calculateCentroid(int i, int j, Point minPt, double resolution);
Image createImageFromCell(Cell *currentCell, Point minPoint, double resolution, int nRows, int nCols);
double calculateMaxDim(Point minPoint, Point maxPoint);
uint32_t writeToLasFile(const char *lasFileName, PointsArr pointsArr);
void writeToBinaryFile(const char *binFileName, PointsArr pointsToWrite, BinaryHeader headerToWrite);
PointsArr readBinaryFile(const char *binFileName);
void pointCloud2Images(const char *lasFileName);

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* pc2img_h */
