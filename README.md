# SAR_RARP50 toolkit

The repository provides evaluation and data manipulation code for the SAR-RARP50
dataset, introduced as a part of the EndoVis MICCAI 2022 challenge. The dataset
provides action recognition and surgical instrumentation segmentation labels
reference for 50 RARP procedure clips, focusing on the DVC suturing phase of the
procedure. For more information about the dataset and the challenge please visit
the [page of the challenge](https://www.synapse.org/#!Synapse:syn27618412/wiki/)

## Setup

The toolkit can be installed using pip and used as a command line tool, or by building a docker image and run as a container.  

### install with pip

``` bash 
pip install git+https://github.com/surgical-vision/SAR_RARP50-evaluation
```

### Building a docker image

We recommend using a docker container to run the SAR-RARP50 toolkit code.
Currently, we only support building this docker image from source:

Clone this repository and cd into it

```bash
git clone https://github.com/surgical-vision/SAR_RARP50-evaluation && cd ./SAR_RARP50-evaluation
```

Build the docker image

```bash
docker image build -t sarrarp_tk .
```

## How to use


We provide scripts to perform data preprocessing and method evaluation.
For the SAR-RARP50 Endovis challenge we also provide code to generate
mock predictions in the format we expect competing methods to export results.

To use any of the provided scripts, you must mount the SAR-RARP50 dataset directory
as a volume in the docker container. By doing so, data processing and predictions can be
stored under the SAR-RARP50 dataset directory.

For each command, you will need to run the docker container as follows:

package:
``` bash 
rarptk command args
```

docker:
``` bash
docker container run --rm \
                     -v /path_to_root_data_dir/:/data \
                     sarrarp_tk \
                     command args 
```

The `-v` flag mounts the directory containing a local copy of the SAR-RARP50 dataset
to the /data directory inside the docker container. Be sure to provide the absolute
path of your local data directory. The SAR-RARP50 dataset directory is assumed to have
the following file structure

```tree
path_to_root_data_dir
├── train1
│   ├── video_*
│   ├── ...
│   └── video_*
├── train2
│   ├── video_*
│   ├── ...
│   └── video_*
└── test
    ├── video_*
    ├── ...
    └── video_*

```

Replace the command and args according to the task you want to perform

Currently, the codebase supports the following command:

- evaluate: perform method evaluation
- unpack: extracts images from the downloaded videos at 10Hz
- generate: generates mock predictions that serve as an example to the SAR-RARP50 expected prediction format


We also provide the corresponding python scripts in case you use devcontainers.

### Unpack videos to rgb images(10Hz)

During the SAR-RARP50challenge we provide zip files of each video sequence and
ask participants to unpack each individual video_**.zip file, replicating the root
directory structure as described [here](https://www.synapse.org/#!Synapse:syn27618412/wiki/618427)
After unpacking the .zip files videos need to be sampled at 10Hz which can be done
by running the following

installed using pip:

``` bash 
rarptk unpack /path_to_root_data_dir/ -j4 -r
```


Docker container:
``` bash
docker container run --rm \
                     -v /path_to_root_data_dir/:/data/ \
                     sarrarp_tk \
                     unpack /data/ -j4 -r 
```

devcontainer:

``` bash
python -m scripts.sarrarp_tk unpack /path_to_root_data_dir/ -j4 -r 
```

The `unpack` script exposes the following command line interface

unpack video_dir [`--recursive`] [`--jobs`]

- `data_dir`: path pointing to the dataset or video directory
- [`-r`, `--recursive`] : search recursively for video directories that have video_left.avi as a child
- [`-j`, `--jobs`] : number of concurrent jobs to run while exporting .png files
- [`-f`, `--frequency`] : sampling rate in Hz, choices=[1, 10], default=10

### Generating mock predictions

To generate mock predictions run the following

installed using pip:
```bash
rarptk generate /path_to_root_data_dir/test/ /path_to_root_data_dir/mock_predictions/ 
```

docker:
``` bash
docker container run --rm \
                     -v /path_to_root_data_dir/:/data/ \
                     sarrarp_tk \
                     generate /data/test/ /data/mock_predictions/ 
```

devcontainer:

``` bash
python -m scripts.sarrarp_tk generate /path_to_root_data_dir/test/ /path_to_root_data_dir/mock_predictions/ 
```

The `generate` script exposes the following command line interface

generate test_dir prediction_dir [--overwrite]

- `test_dir` : Absolute path of the test directory inside the container
- `prediction_dir` : Absolute path of the directory to store the mock predictions under
- [`-o`, `--overwrite`] : Flag to overwrite the mock predictions if prediction_dir exists.

### Evaluation

SAR-RARP50 provides both action recognition and surgical instrumentation
segmentation labels. Our evaluation tool assumes that predictions follow the
same file structure as the train sets(segmentation folder containing .pngs and
an action_discrete.txt file right under each video_* directory). Please refer to
[Generating mock predictions](#generating-mock-predictions) to see how you can
generate predictions in the required format.

After a given algorithms stores predictions under /path_to_root_data_dir/predictions/
in the host filesystem, you can evaluate by running the following

installed using pip:
``` bash
rarptk evaluate /path_to_root_data_dir/ref_set/ /path_to_root_data_dir/predictions/ 
```

docker:
``` bash
docker container run --rm \
                     -v /path_to_root_data_dir/:/data/ \
                     sarrarp_tk \
                     evaluate /data/custom_ref_set/ /data/predictions/ 
```

devcontainer:

``` bash
python -m scripts.sarrarp_tk evaluate /path_to_root_data_dir/custom_ref_set/ /path_to_root_data_dir/predictions/ 
```

The `evaluate` script exposes the following command line interface

evaluate ref_dir prediction_dir [--ignore_actions] [--ignore_segmentation]

- `ref_dir` : absolute path to the directory of the reference set
- `prediction_dir` : absolute path to the directory predictions are stored
- [`--ignore_actions`] : do not perform action recognition evaluation
- [`--ignore_segmentation`] : do not perform segmentation evaluation
- [`--class_errors`] : save per-class scores for segmentation aggregated in video level

After a quick file structure evaluation, the script will compute the accuracy of the predictions
as described [here](https://www.synapse.org/#!Synapse:syn27618412/wiki/617968)

When the evaluation for the whole reference set finishes, the following files will
get generated under prediction_dir:

- per_video_results.csv : aggregates scores of each metric at video level
- final_results.csv : final score across all videos which is used to rank approaches
