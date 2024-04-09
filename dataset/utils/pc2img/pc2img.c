//
//  pc2img.c
//  PointCloud2Image
//
//  Created by Theodora on 26/03/2024.
//

#include "pc2img.h"
#include <string.h>

// setu
CellArr initializeCells(LASPublicHeader_t *lasHeader, int cellSize)
{
    // calculate the number of rows and cells based on the cell size
    int rows = ceil(lasHeader->maxY - lasHeader->minY)/cellSize;
    int colCells = ceil(lasHeader->maxX - lasHeader->minX)/cellSize;
    int colOvCells = colCells - 1;
    
    CellArr cellsArray;
    cellsArray.cells = (Cell*)malloc(rows*colCells * sizeof(Cell));
    cellsArray.overlappingCells = (Cell*)malloc(rows*colOvCells * sizeof(Cell));
    
    cellsArray.rows = rows;
    cellsArray.colCells = colCells;
    cellsArray.colOvCells = colOvCells;
    
    for (int i = 0; i < rows; i++) {
        for(int j = 0; j < colCells; j++)
        {
            cellsArray.cells[i*colCells + j].i = i;
            cellsArray.cells[i*colCells + j].j = j;
            cellsArray.cells[i*colCells + j].pointsArr.numberOfPoints = 0;
            cellsArray.cells[i*colCells + j].pointsArr.points = (Point*) malloc(cellsArray.cells[i*colCells + j].pointsArr.numberOfPoints * sizeof(Point));
        }
        
        for(int k = 0; k < colOvCells; k++)
        {
            cellsArray.overlappingCells[i*colOvCells + k].i = i;
            cellsArray.overlappingCells[i*colOvCells + k].j = k;
            cellsArray.overlappingCells[i*colOvCells + k].pointsArr.numberOfPoints = 0;
            cellsArray.overlappingCells[i*colOvCells + k].pointsArr.points = (Point*) malloc(cellsArray.overlappingCells[i*colOvCells + k].pointsArr.numberOfPoints * sizeof(Point));
        }
    }
    
    return cellsArray;
}

CellArr assignPointsToCells(const char *lasFileName)
{
    int cellSize = 100;
    uint64_t pointsRead = 0;
    int pointsToRead;
    
    CellArr cellsArray;
    cellsArray.cells = NULL;
    cellsArray.overlappingCells = NULL;
        
    LASPublicHeader_t lasHeader;
    FILE *lasFile = fopen(lasFileName, "r");
    
    if (lasFile != NULL) {
        
        fread(&lasHeader, sizeof(LASPublicHeader_t), 1, lasFile);
    
        fseek(lasFile, lasHeader.offsetToPointData, SEEK_SET);
        cellsArray = initializeCells(&lasHeader, cellSize);
        pointsToRead = lasHeader.legacyNumberOfPointRecords;
        
        uint8_t *lasPointData = (uint8_t *)malloc(pointsToRead * lasHeader.pointDataRecordLength);
        Point point;
        pointsRead = fread(lasPointData, lasHeader.pointDataRecordLength, pointsToRead, lasFile);

        fclose(lasFile);

        for (int i = 0; i < pointsRead; i++) {
            
            switch (lasHeader.pointDataRecordFormat) {
                    
                case 0 ... 5: {
                    LASPointCommon012345_t *lasPointCommon = (LASPointCommon012345_t*)&lasPointData[i * lasHeader.pointDataRecordLength];
                    point.x                = (float)((double)lasPointCommon->x * lasHeader.xScaleFactor + lasHeader.xOffset);
                    point.y                = (float)((double)lasPointCommon->y * lasHeader.yScaleFactor + lasHeader.yOffset);
                    point.z                = (float)((double)lasPointCommon->z * lasHeader.zScaleFactor + lasHeader.zOffset);
                    point.intensity        = (uint8_t)(lasPointCommon->intensity >> 8);
                    
                    LASPointExtra2_t *lasPointExtra = (LASPointExtra2_t *)((uint8_t *)lasPointCommon + sizeof(LASPointCommon012345_t));
                    point.red         = (uint8_t)(lasPointExtra->red >> 8);
                    point.green       = (uint8_t)(lasPointExtra->green >> 8);
                    point.blue        = (uint8_t)(lasPointExtra->blue >> 8);
                    
                    point.isEmpty = false;
                    point.pointId = i;
                    calculateCellIndex(&cellsArray, point, &lasHeader, cellSize);
                }
            }
        }
    }
        
    return cellsArray;
}

void calculateCellIndex(CellArr *cellsArr, Point currentPoint, LASPublicHeader_t *lasHeader, int cellSize)
{
    float overlap = cellSize/2;
    
    //starting from minX,minY
    float startX = lasHeader->minX;
    float startY = lasHeader->minY;
    float dx = (currentPoint.x - startX) / cellSize;
    float dy = (currentPoint.y - startY) / cellSize;
    int i = (dx < 5 )? floor(dx) : (floor(dx) - 1) ;
    int j = (dy < 5 )? floor(dy) : (floor(dy) - 1) ;
    cellsArr->cells[i*cellsArr->colCells + j].pointsArr.numberOfPoints = cellsArr->cells[i*cellsArr->colCells + j].pointsArr.numberOfPoints + 1;
    cellsArr->cells[i*cellsArr->colCells + j].pointsArr.points = (Point*)realloc(cellsArr->cells[i*cellsArr->colCells + j].pointsArr.points, cellsArr->cells[i*cellsArr->colCells + j].pointsArr.numberOfPoints * sizeof(Point));
    cellsArr->cells[i*cellsArr->colCells + j].pointsArr.points[cellsArr->cells[i*cellsArr->colCells + j].pointsArr.numberOfPoints - 1] = currentPoint;
    
    //overlapping areas (for now only in i indices)
    startX = startX + overlap;
    dx = (currentPoint.x - startX) / cellSize;
    int iOver = (dx >= 0 && floor(dx)<= 3) ? floor(dx) : -1;
    int jOver = (dx >= 0 && floor(dx)<= 3) ? j : -1;
    
    if(iOver != -1 && jOver != -1)
    {
        cellsArr->overlappingCells[iOver*cellsArr->colOvCells + jOver].pointsArr.numberOfPoints = cellsArr->overlappingCells[iOver*cellsArr->colOvCells + jOver].pointsArr.numberOfPoints + 1;
        cellsArr->overlappingCells[iOver*cellsArr->colOvCells + jOver].pointsArr.points = (Point*)realloc(cellsArr->overlappingCells[iOver*cellsArr->colOvCells + jOver].pointsArr.points, cellsArr->overlappingCells[iOver*cellsArr->colOvCells + jOver].pointsArr.numberOfPoints * sizeof(Point));
        cellsArr->overlappingCells[iOver*cellsArr->colOvCells + jOver].pointsArr.points[cellsArr->overlappingCells[iOver*cellsArr->colOvCells + jOver].pointsArr.numberOfPoints - 1] = currentPoint;
    }
    
}

Point calculateMin3D(PointsArr *points)
{
    Point minPoint = points->points[0];
    
    for (int i = 0; i < points->numberOfPoints; i++) {
        minPoint.x = (minPoint.x > points->points[i].x) ? points->points[i].x : minPoint.x;
        minPoint.y = (minPoint.y > points->points[i].y) ? points->points[i].y : minPoint.y;
        minPoint.z = (minPoint.z > points->points[i].z) ? points->points[i].z : minPoint.z;
    }
    
    return minPoint;
}

Point calculateMax3D(PointsArr *points)
{
    Point maxPoint = points->points[0];
    
    for (int i = 0; i < points->numberOfPoints; i++) {
        maxPoint.x = (maxPoint.x < points->points[i].x) ? points->points[i].x : maxPoint.x;
        maxPoint.y = (maxPoint.y < points->points[i].y) ? points->points[i].y : maxPoint.y;
        maxPoint.z = (maxPoint.z < points->points[i].z) ? points->points[i].z : maxPoint.z;
    }
    
    return maxPoint;
}

Point calculateCentroid(int i, int j, Point minPt, double resolution)
{
    Point point;
    point.x = minPt.x + (i * resolution) + (resolution/2);
    point.y = minPt.y + (j * resolution) + (resolution/2);
    point.z = __FLT_MIN__;
    point.red = 0;
    point.green = 0;
    point.blue = 0;
    point.isEmpty = true;
    point.intensity = 0;
    point.pointId = -1;
    
    return point;
}

Image createImageFromCell(Cell *currentCell, Point minPoint, double resolution, int nRows, int nCols)
{
    Image currentImage;
    currentImage.nRows = nRows;
    currentImage.nCols = nCols;
    currentImage.pointsArr.numberOfPoints = currentImage.nRows*currentImage.nCols;
    currentImage.pointsArr.points = (Point*)malloc(currentImage.pointsArr.numberOfPoints * sizeof(Point));
    
    for(int col = 0; col < currentImage.nCols; col++)
    {
        for(int row = 0; row < currentImage.nRows; row++)
        {
            currentImage.pointsArr.points[col*currentImage.nRows + row] = calculateCentroid(col, row, minPoint, resolution);
        }
    }
    
    for (int n = 0; n < currentCell->pointsArr.numberOfPoints; n++) {
        Point currentPoint = currentCell->pointsArr.points[n];
        float dx = (currentPoint.x - minPoint.x) / resolution;
        float dy = (currentPoint.y - minPoint.y) / resolution;
        int i = (dx < PIXEL_LIMIT_PER_DIM)? floor(dx) : (floor(dx) - 1) ;
        int j = (dy < PIXEL_LIMIT_PER_DIM )? floor(dy) : (floor(dy) - 1) ;
        
        if (currentImage.pointsArr.points[i*currentImage.nRows + j].z < currentPoint.z) {
            
            currentImage.pointsArr.points[i*currentImage.nRows + j] = currentPoint;
            currentImage.pointsArr.points[i*currentImage.nRows + j].isEmpty = false;
        }
    }
    
    return currentImage;
}

double calculateMaxDim(Point minPoint, Point maxPoint)
{
    return ((maxPoint.x - minPoint.x) > (maxPoint.y - minPoint.y) ? (maxPoint.x - minPoint.x) : (maxPoint.y - minPoint.y));
}

void writeToBinaryFile(const char *binFileName, PointsArr pointsToWrite, BinaryHeader headerToWrite)
{
    FILE *file;
    file = fopen(binFileName, "wb");
    
    if(file != NULL)
    {
        fseek(file, sizeof(BinaryHeader), SEEK_SET);
        uint32_t pointsWritten = (uint32_t)fwrite(pointsToWrite.points, sizeof(Point), pointsToWrite.numberOfPoints, file);
        headerToWrite.numberOfPoints = pointsWritten;
        fseek(file, 0, SEEK_SET);
        fwrite(&headerToWrite, sizeof(BinaryHeader), 1, file);
        printf("%lu \n", sizeof(BinaryHeader));
        printf("%u \n", pointsWritten);
    }
    fclose(file);
}

PointsArr readBinaryFile(const char *binFileName)
{
    PointsArr pointsArr;
    pointsArr.numberOfPoints = 0;
    pointsArr.points = (Point*)malloc(pointsArr.numberOfPoints * sizeof(Point));

    BinaryHeader header;
    FILE *file = fopen(binFileName, "rb");
    
    if(file != NULL)
    {
        fread(&header, sizeof(BinaryHeader), 1, file);
        pointsArr.numberOfPoints = header.numberOfPoints;
        printf("%u \n", pointsArr.numberOfPoints);    

        pointsArr.points = (Point*)realloc(pointsArr.points, pointsArr.numberOfPoints * sizeof(Point));
        fread(pointsArr.points, sizeof(Point), pointsArr.numberOfPoints, file);
        
        // for (size_t i = 0; i < pointsArr.numberOfPoints; i++)
        // {   
        //     if(pointsArr.points[i].isEmpty == false)
        //     {
        //         printf("%f %f %f %u %u %u %d\n", pointsArr.points[i].x, pointsArr.points[i].y, pointsArr.points[i].z, pointsArr.points[i].red, pointsArr.points[i].green, pointsArr.points[i].blue, pointsArr.points[i].isEmpty);
        //     }
        // }
        
        // Point point;
        // for(int i = 0; i<header.numberOfPoints; i++)
        // {
        //     fread(&point, sizeof(Point), 1, file);
        //     printf("%f %f %f %d\n", point.x, point.y, point.z, point.isEmpty);
        //     printf("\n");
        // }
    }
    fclose(file);

    return pointsArr;
}

void pointCloud2Images(const char *lasFileName)
{
    int nRows = PIXEL_LIMIT_PER_DIM;
    int nCols = PIXEL_LIMIT_PER_DIM;
    
    Cell currentCell;
    CellArr cellsArr = assignPointsToCells(lasFileName);
    
    for(int i = 0; i < cellsArr.rows; i++)
    {
        //normal cells
        for(int j = 0; j < cellsArr.colCells; j++)
        {
            currentCell = cellsArr.cells[i* cellsArr.colCells + j];
            Point minPoint = calculateMin3D(&currentCell.pointsArr);
            Point maxPoint = calculateMax3D(&currentCell.pointsArr);
            double resolution = calculateMaxDim(minPoint, maxPoint)/PIXEL_LIMIT_PER_DIM;
            
            Image currentImage = createImageFromCell(&currentCell, minPoint, resolution, nRows, nCols);
            
            char currentBinFileName[512] = "", extStr[128];
            strncpy(currentBinFileName, lasFileName, (strlen(lasFileName) - 4));

            snprintf(extStr, 128, "_%d_%d", i, j);
            strcat(currentBinFileName, extStr);
            printf("%s \n", currentBinFileName);
            
            BinaryHeader currentHeader;
            currentHeader.numberOfPoints = 0;
            currentHeader.width = currentImage.nCols;
            currentHeader.height = currentImage.nRows;
            
            writeToBinaryFile(currentBinFileName, currentImage.pointsArr, currentHeader);
        }
        
        //overlapping cells
        for(int k = 0; k < cellsArr.colOvCells; k++)
        {
            currentCell = cellsArr.overlappingCells[i* cellsArr.colOvCells + k];
            Point minPoint = calculateMin3D(&currentCell.pointsArr);
            Point maxPoint = calculateMax3D(&currentCell.pointsArr);
            double resolution = calculateMaxDim(minPoint, maxPoint)/PIXEL_LIMIT_PER_DIM;
            
            Image currentImage = createImageFromCell(&currentCell, minPoint, resolution, nRows, nCols);
            
            char currentBinFileName[512] = "", extStr[128];
            strncpy(currentBinFileName, lasFileName, (strlen(lasFileName) - 4));

            snprintf(extStr, 128, "_%d_%d_ov", i, k);
            strcat(currentBinFileName, extStr);
            printf("%s \n", currentBinFileName);
            
            BinaryHeader currentHeader;
            currentHeader.numberOfPoints = 0;
            currentHeader.width = currentImage.nCols;
            currentHeader.height = currentImage.nRows;
            
            writeToBinaryFile(currentBinFileName, currentImage.pointsArr, currentHeader);
            printf("\n");
        }
    }
}

int main(int argc, const char * argv[]) {
    
    char fileName[512];
    snprintf(fileName, 512, "Tile_+003_+005.las");
    pointCloud2Images(fileName);

    // PointsArr arr = readBinaryFile("Tile_+003_+005_0_0");
    // printf("%d\n", arr.numberOfPoints);
    // printf("%f %f %f %u %u %u %d\n", arr.points[0].x, arr.points[0].y, arr.points[0].z, arr.points[0].red, arr.points[0].green, arr.points[0].blue, arr.points[0].isEmpty);

    return 0;
}