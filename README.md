# yolo-labels-python-visualiser

no-nonsense python script to visualise labels of YOLO format.


## Usage
1. Install requirements
```sh
pip install -r requirements.txt
```
2. set up yaml. See eg_dataset.yaml. Required fields are: {"nc", "names", and "class_clrs"}

2. Run the script
```sh
python3 vis_BB.py \
--images_dir <path to the directory with the rgb images> \
--labels_dir <path to the labels directory> \
--output_dir <path to output images directory> \
--classes_yaml <path to yaml prepared in previous step> \
--show_text 
```
