import os
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
import click
import yaml
from pathlib import Path
from matplotlib import font_manager


@click.command()
@click.option(
    "--images_dir",
    type=str,
    help="path to the rgb directory",
    required=True,
)
@click.option(
    "--labels_dir",
    type=str,
    help="path to the labels directory",
    required=True,
)
@click.option(
    "--output_dir",
    type=str,
    help="path to the output directory",
    required=True,
)
@click.option(
    "--classes_yaml",
    type=str,
    help="path to the cfg describing the classes",
    default="eg_dataset.yaml",
)
@click.option("--show_text", is_flag=True, help="show text on top of bounding box?")
def draw_bbs(images_dir, labels_dir, output_dir, classes_yaml, show_text):
    """
    Function to draw bbs (from YOLO format txt labels) onto images
    Some parts were written by Julian Bauer
    """

    os.makedirs(output_dir, exist_ok=True)

    filenames = os.listdir(images_dir)

    with open(classes_yaml, "r") as classes_yaml_f:
        classes_yaml_dict = yaml.safe_load(classes_yaml_f)

    nc = classes_yaml_dict["nc"]
    names = classes_yaml_dict["names"]
    class_clrs = classes_yaml_dict["class_clrs"]

    if nc != len(names):
        raise ValueError(
            f"number of classes (nc={nc}) does not match number of class names (names={names}) in cfg {classes_yaml}!"
        )

    if len(class_clrs) != len(names):
        raise ValueError(
            f"number of colors does not match number of class names in cfg {classes_yaml}!"
        )

    if show_text:
        font = font_manager.FontProperties(family="sans-serif", weight="bold")
        font_file = font_manager.findfont(font)
        print(f"using font from {font_file}")
    else:
        font_pil = None

    max_w = 0 # DBG
    max_h = 0 # DBG

    for filename in tqdm(filenames):
        try:
            img = Image.open(os.path.join(images_dir, filename))
        except:
            print(f"Skipping {filename} because PIL cannot read file as image.")

        # check if label exists. if not, just skip
        fp = os.path.join(labels_dir, Path(filename).stem + ".txt")
        if not os.path.isfile(fp):
            print(f"Skipping file {filename} since label is not found")
            continue
        f = open(fp, "r")
        img = ImageOps.exif_transpose(
            img
        )  # for smartphone images to be oriented correctly

        w, h = img.size
        coord = np.zeros((1, 5))
        for i, line in enumerate(f):
            l = line.split(" ")
            coord[0, 0] = l[1]
            coord[0, 1] = l[2]
            coord[0, 2] = l[3]
            coord[0, 3] = l[4][:-1]
            coord[0, 4] = l[0]

            img_draw = ImageDraw.Draw(img)
            # draw the bb
            m = 0
            x_min = w * coord[m, 0] - w * coord[m, 2] / 2
            x_max = w * coord[m, 0] + w * coord[m, 2] / 2
            y_min = h * coord[m, 1] - h * coord[m, 3] / 2
            y_max = h * coord[m, 1] + h * coord[m, 3] / 2
            rect = [x_min, y_min, x_max, y_max]

            clr_str = class_clrs[int(coord[m, 4])]
            img_draw.rectangle(rect, outline=clr_str, width=3)

            if show_text:
                # also write the class onto the image
                dw = x_max - x_min
                min_font_size = 12  # tuned based on my eyesight
                dt = 2
                # set the font size dynamically based on image size and bounding box
                name = names[int(coord[m, 4])]
                fill_dw_fs = int(dw / len(name))
                font_size = max(fill_dw_fs, min_font_size)
                font_pil = ImageFont.truetype(font_file, font_size)
                img_draw.text(
                    (x_min, y_min - font_size - dt), name, font=font_pil, fill=clr_str
                )

            max_w = max(max_w, x_max - x_min)
            max_h = max(max_h, y_max - y_min)

        img.save(os.path.join(output_dir, filename))
    print(f"max w: {max_w}, h: {max_h}")

if __name__ == "__main__":
    draw_bbs()
