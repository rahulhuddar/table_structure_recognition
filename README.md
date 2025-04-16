
# Companies-llm

This is an implementation of "Microsoft - Table Transformer", an object detection model for extracting tables from images and PDFs.

Basic Architecture:

TT Detection --> TT Structure recognition + PaddleOCR--> html

Tables detected by TT Detection model are cropped and sent to TT Structure recognition model to detect the rows and columns in the cropped table images. 

Using PaddleOCR the text line coordinates are obtained.

Based on the coordinates of rows and columns, table cell coordinates are calculated.

Based on overlap percentage between the ocr text coordinates and cell coordinates, text is assigned to the correct cell. These cells are then converted to html tables.

Important Note: Tables detected by Detection model having height less than 700 are sent to TT Structure recognition model finetuned on TATR-v1.1-Pub

Tables detected by Detection model having height more than 700 are sent to TT Structure recognition model finetuned on TATR-v1.1-Fin
## Installation



```bash
conda create -n tabledetr python=3.10

conda activate tabledetr

pip install -r requirements.txt 

#Note: This is the CPU version of paddleocr
python -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
```
    
## Finetuned model weights

#### Download all the model files and save it locally, remember to change the model path in custom_inference.py file

| Type | Model     | Link                |
| :-------- | :------- | :------------------------- |
| `Detection` | `DETR R18` | https://drive.google.com/file/d/1HS8AxnC2kCBy8aWmWlBimHeXI6qJkAlu/view?usp=drive_link |
| `Structure Recognition` | `TATR-v1.1-Pub` | https://drive.google.com/file/d/1SN3UTi07eeMoHs2AC1IZ4YqdQVBmO-3v/view?usp=drive_link |
| `Structure Recognition` | `TATR-v1.1-Fin` | https://drive.google.com/file/d/1Lh9MxN1urBDIDjjhJ-ddlLz3MwbiKQum/view?usp=drive_link |

## Quick Start

```bash
cd src/
python custom_inference.py -z 
```
The above command expects a input folder path containing .jpg/.png files. Default input folder name is 'input'.

The output will saved in a 'output' folder.

Using -z flag will help Visualize detected table structure.

You can change the input and output folder paths using below mentioned flags
```bash
python custom_inference.py --image_dir path/to/input --out_dir path/to/output/ -z 
```
## Usage
Optionally you can add other flags to change the model path, config path, crop padding.

```--structure_config_path``` Filepath to the structure model config file

```--structure_model_path``` File path to the structure model for tables having height less than 700(prefer pub model)

```--structure_config_path2``` File path to the structure model for tables having height more than 700(prefer fin model)

```--detection_config_path``` Filepath to the detection model config file

```--detection_model_path``` File path to the detection model

```--detection_device``` Device for running detection model. You can change to cpu or cuda depending on availability of gpu. By default it will automatically detect and use if gpu is available else cpu.

```--structure_device``` Device for running structure recognition model. You can change to cpu or cuda depending on availability of gpu. By default it will automatically detect and use if gpu is available else cpu.

```--verbose``` Filepath to the structure model config file

```--structure_config_path``` Verbose output

```--crop_padding``` The amount of padding to add around a detected table when cropping. Default set to 10

## Training/Finetuning

Refer the train_table_transformers.ipynb jupyter notebook file for detailed training documentaion and code. Upload the file on Google Colab to use GPU for training.

## Reference Links

 - [Microsoft - Table Transformer](https://github.com/microsoft/table-transformer)
 - [PaddlePaddle - PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

