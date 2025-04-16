#Installation
conda create -n tabledetr python=3.10
conda activate tabledetr
pip install -r requirements.txt 
python -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple #Note: This is the CPU version of paddleocr


#Quick Start
cd src/
python custom_inference.py -z 

#Use jpg for training(If using png, make changes in main.py- replace jpg with png)
