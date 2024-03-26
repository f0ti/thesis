#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct __attribute__((__packed__))
{
    float x;
    float y;
    float z;
    uint16_t red;
    uint16_t green;
    uint16_t blue;
    uint16_t intensity;
    bool isEmpty;
} Point;

typedef struct __attribute__((__packed__))
{
    uint64_t numberOfPointRecords;

} Header;

int main()
{

    char fileName[1024];
    sprintf(fileName, "5Points");
    Header header;
    FILE *file = fopen(fileName, "rb");
    
    // print sizeof(Header)
    printf("Size of Header: %lu\n", sizeof(Header));

    if (file != NULL)
    {
        fread(&header, sizeof(Header), 1, file);
        printf("%lu \n", header.numberOfPointRecords);

        Point point;

        for (int i = 0; i < header.numberOfPointRecords; i++)
        {
            fread(&point, sizeof(Point), 1, file);

            printf("%f %f %f %d %d %d %d %d\n", point.x, point.y, point.z, point.red, point.green, point.blue, point.intensity, point.isEmpty);
        }
    }

    return 0;
}