import click
import pdal

@click.command()
@click.argument('input_las', type=click.Path(exists=True))
@click.argument('output_pcd')
def las_to_pcd(input_las, output_pcd):
    pipeline = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": input_las
            },
            {
                "type": "writers.pcd",
                "filename": output_pcd
            }
        ]
    }

    pipeline_str = pdal.pipeline2json(pipeline)
    pipeline = pdal.Pipeline(pipeline_str)
    pipeline.execute()

if __name__ == '__main__':
    las_to_pcd()
