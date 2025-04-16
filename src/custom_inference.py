from collections import OrderedDict, defaultdict
import json
import argparse
import sys
import xml.etree.ElementTree as ET
import os
import random

from paddleocr import PaddleOCR

import torch
from torchvision import transforms
from PIL import Image
from fitz import Rect
import numpy as np
import cv2
import copy

from main import get_model
sys.path.append("../detr")
from models import build_model

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', default= 'input_1',
                        help="Directory for input images")
    parser.add_argument('--out_dir', default= 'output_1',
                        help="Output directory")
    parser.add_argument('--structure_config_path', default= 'structure_config.json',
                        help="Filepath to the structure model config file")
    parser.add_argument('--structure_model_path', default= 'structure_fin_model/fin_model_20.pth',
                        help="The path to the structure model for tables having height less than 700(prefer pub model)")
    parser.add_argument('--structure_model_path2', default= 'structure_fin_model/fin_model_20.pth',
                        help="The path to the structure model for tables having height more than 700(prefer fin model)")
    parser.add_argument('--detection_config_path', default='detection_config.json',
                        help="Filepath to the detection model config file")
    parser.add_argument('--detection_model_path', default='detect_model/model_20.pth',
                        help="The path to the detection model")                       
    parser.add_argument('--detection_device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--structure_device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output') 
    parser.add_argument('--visualize', '-z', action='store_true',
                        help='Visualize detected table structure') 
    parser.add_argument('--crop_padding', type=int, default=0,
                        help="The amount of padding to add around a detected table when cropping.")

    return parser.parse_args()

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))
        
        return resized_image

detection_transform = transforms.Compose([
    MaxResize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

structure_transform = transforms.Compose([
    MaxResize(1000),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_class_map(data_type):
    if data_type == 'structure':
        class_map = {
            'table': 0,
            'table column': 1,
            'table row': 2,
            'table column header': 3,
            'table projected row header': 4,
            'table spanning cell': 5,
            'no object': 6
        }
    elif data_type == 'detection':
        class_map = {'table': 0, 'table rotated': 1, 'no object': 2}
    return class_map

detection_class_thresholds = {
    "table": 0.60,
    "table rotated": 0.60,
    "no object": 10
}

structure_class_thresholds = {
    "table": 0.55,
    "table column": 0.55,
    "table row": 0.70,
    "table column header": 0.55,
    "table projected row header": 0.55,
    "table spanning cell": 0.55,
    "no object": 10
}

def compute_overlap(bbox1, bbox2):
    # Calculate the overlap area between two bounding boxes
    rect1 = Rect(bbox1[0],bbox1[1],bbox1[2],bbox1[3])
    rect2 = Rect(bbox2[0],bbox2[1],bbox2[2],bbox2[3])
    intersection = rect1.intersect(rect2)

    bbox1_width = bbox1[2] - bbox1[0]
    bbox1_height = bbox1[3] - bbox1[1]
    bbox1_area = bbox1_width * bbox1_height
    if intersection != 0:
        overlap = intersection.get_area()/bbox1_area
        return overlap
    else:
        return 0

def calculate_overlap(box1, box2):
    # Calculate the overlap area between two bounding boxes
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    intersection = max(0, min(x2, x4) - max(x1, x3)) * max(0, min(y2, y4) - max(y1, y3))
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    return intersection / min(area1, area2)

def postprocess(values, threshold):
    """ Postprocess will be performed on the output(objects) of table transfmormer structure recognition model.
    In this extra rows/columns will be added in the gaps in table and cells will be created based on the 
    row number and column number"""

    #apply threshold to all class labels
    processed_values = [obj for obj in values if obj['score'] >= threshold[obj['label']]]
    values = processed_values    

    #Group objects according to labels
    tables = [obj for obj in values if obj['label']=='table'] 
    spanning_cells = [obj for obj in values if obj['label']=='table spanning cell']
    column_header = [obj for obj in values if obj['label']=='table column header']
    projected_row = [obj for obj in values if obj['label']=='table projected row header']
    columns = [obj for obj in values if obj['label']=='table column']
    rows = [obj for obj in values if obj['label']=='table row']

    #Only one table is allowed, tables with low scores are removed
    if tables: 
        high_score_table = (max(tables, key=lambda x: x['score']))
        tables = [high_score_table]
    
    if tables:
        #Reject tabels with no rows and columns
        if len(rows) == 0: 
            cells = [] 
            values = []
            return cells, values
        if len(columns) == 0: 
            cells = [] 
            values = [] 
            return cells, values

        #Arrange rows and columns based on coordinates
        rows = sorted(rows, key=lambda x: x['bbox'][1])
        columns = sorted(columns, key=lambda x: x['bbox'][0])

        #Resize the heights of columns to height of table
        for column in columns: 
            column['bbox'][1] = tables[0]['bbox'][1]
            column['bbox'][3] = tables[0]['bbox'][3]

        #Resize the width of rows to width of table
        for rowid, row in enumerate(rows): 
            row['bbox'][0] = tables[0]['bbox'][0]
            row['bbox'][2] = tables[0]['bbox'][2]

        #Eliminate rows that are outside table
        for rowid, row in enumerate(rows):
            row_check1 = int(row['bbox'][1]) - int(tables[0]['bbox'][1]) 
            row_check2 = int(row['bbox'][3]) - int(tables[0]['bbox'][3])
            if row_check1 < -7:
                rows.pop(rowid)
            if row_check2 > 7:
                rows.pop(rowid)
        
        # Fill in the gaps in table with rows
        rows_copy = copy.deepcopy(rows) 
        for rowid, row in enumerate(rows_copy):
            if rowid == 0:
                gap = row['bbox'][1] - tables[0]['bbox'][1]
                if gap > 10:
                    rows.insert(0, {'label': 'table row', 'score': 1, 'bbox': [tables[0]['bbox'][0], tables[0]['bbox'][1], tables[0]['bbox'][2], row['bbox'][1]]})
            else:
                gap = row['bbox'][1] - rows[rowid - 1]['bbox'][3]
                if gap > 10:
                    rows.append({'label': 'table row', 'score': 1, 'bbox': [tables[0]['bbox'][0], rows[rowid - 1]['bbox'][3], tables[0]['bbox'][2], row['bbox'][1]]})

        # Fill in the gaps in table with columns
        columns_copy = copy.deepcopy(columns) 
        for colid, column in enumerate(columns_copy):
            if colid == 0:
                gap = column['bbox'][0] - tables[0]['bbox'][0]
                if gap > 25:
                    columns.insert(0, {'label': 'table column', 'score': 1, 'bbox': [tables[0]['bbox'][0], tables[0]['bbox'][1], column['bbox'][0], column['bbox'][3]]})
            else:
                gap = column['bbox'][0] - columns[colid - 1]['bbox'][2]
                if gap > 25:
                    columns.append({'label': 'table column', 'score': 1, 'bbox': [columns[colid - 1]['bbox'][2], tables[0]['bbox'][1], column['bbox'][0], column['bbox'][3]]})

        #Again Arrange rows and columns based on coordinates
        rows = sorted(rows, key=lambda x: x['bbox'][1]) 
        columns = sorted(columns, key=lambda x: x['bbox'][0])

        # Remove columns with more than 50% overlap and lower score
        columns_to_remove = []
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                if calculate_overlap(columns[i]['bbox'], columns[j]['bbox']) > 0.5:
                    if columns[i]['score'] < columns[j]['score']:
                        columns_to_remove.append(i)
                    else:
                        columns_to_remove.append(j)
        columns = [columns[i] for i in range(len(columns)) if i not in columns_to_remove]

        # Remove rows with more than 50% overlap and lower score
        rows_to_remove = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if calculate_overlap(rows[i]['bbox'], rows[j]['bbox']) > 0.5:
                    if rows[i]['score'] < rows[j]['score']:
                        rows_to_remove.append(i)
                    else:
                        rows_to_remove.append(j)
        rows = [rows[i] for i in range(len(rows)) if i not in rows_to_remove]

        values = tables+spanning_cells+column_header+projected_row+columns+rows

        #Get Spanning cell's row and column numbers
        for span_cell in spanning_cells: 
            for row_num, row in enumerate(rows):
                overlap = compute_overlap(span_cell['bbox'], row['bbox'])
                if overlap>0.5:
                    if 'row_num' in span_cell:
                        span_cell['row_num'].append(row_num)
                    else:
                        span_cell['row_num'] = [row_num]
            for col_num, column in enumerate(columns):
                overlap = compute_overlap(span_cell['bbox'], column['bbox'])
                if overlap>0.5:
                    if 'col_num' in span_cell:
                        span_cell['col_num'].append(col_num)
                    else:
                        span_cell['col_num'] = [col_num]
        
        #Create cells using rows and columns
        cells = [] 
        for row_num, row in enumerate(rows):
            for col_num, column in enumerate(columns):
                rect1 = Rect(row['bbox'][0],row['bbox'][1],row['bbox'][2],row['bbox'][3])
                rect2 = Rect(column['bbox'][0],column['bbox'][1],column['bbox'][2],column['bbox'][3])
                cell_bbox = rect1.intersect(rect2)
                cells.append({'bbox': list(cell_bbox), 'row_num': [row_num], 'col_num': [col_num]})
        
        #Add the spanning cells to cells list
        # table_cells_dict = {(cell['row_num'][0], col): cell for cell in cells for col in cell['col_num']}
        # for span_cell in spanning_cells:
        #     row = span_cell['row_num'][0]
        #     for col in span_cell['col_num']:
        #         if (row, col) in table_cells_dict:
        #             del table_cells_dict[(row, col)]
        # cells = list(table_cells_dict.values())
        # for span_cell in spanning_cells:
        #     cells.append({'bbox': span_cell['bbox'], 'row_num': span_cell['row_num'], 'col_num': span_cell['col_num']})
            
    else:
        cells = []

    return cells, values

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def outputs_to_objects(outputs, img_size, class_idx2name):
    m = outputs['pred_logits'].softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = class_idx2name[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects

def objects_to_crops(img, objects, class_thresholds, padding=10):
    """Process the bounding boxes produced by the table detection model into
    cropped table images."""

    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue
        cropped_table = {}
        bbox = obj['bbox']
        bbox = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding]

        cropped_img = img.crop(bbox)

        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
        cropped_table['image'] = cropped_img
        table_crops.append(cropped_table)

    return table_crops

def draw_structure(img, det_tables):
    """For visualization of the objects like rows, columns, spanning cell 
    recognized by the structure recognizer model"""

    img = np.array(img)

    for det_table in det_tables:
        bbox = det_table['bbox']

        if det_table['label'] == 'table':
            edgecolor = (1, 0, 0.45)
            # edgecolor = (0, 0, 0)
            linewidth = 2
        elif det_table['label'] == 'table column':
            # edgecolor = (0.95, 0.6, 0.5)
            edgecolor = (0, 0, 0)
            linewidth = 2
        elif det_table['label'] == 'table row' and det_table['score'] == 1:
            # edgecolor = (0.95, 0.6, 0.5)
            edgecolor = (0, 0, 0)
            linewidth = 2
        elif det_table['label'] == 'table row':
            edgecolor = (0.95, 0.6, 0.5)
            # edgecolor = (0, 0, 0)
            linewidth = 2
        # elif det_table['label'] == 'table spanning cell':
        #     edgecolor = (0.95, 0.45, 0)
        #     # edgecolor = (0, 0, 0)
        #     linewidth = 2
        else:
            continue

        edgecolor_bgr = (int(edgecolor[2] * 255), int(edgecolor[1] * 255), int(edgecolor[0] * 255))

        # if det_table['label'] == 'table':
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), edgecolor_bgr, linewidth)


    return img

class TableExtractionPipeline(object):
    def __init__(self, det_device=None, str_device=None,
                 det_model=None, str_model=None,
                 str_model2=None, str_model_path2=None,
                 det_model_path=None, str_model_path=None,
                 det_config_path=None, str_config_path=None):

        self.det_device = det_device
        self.str_device = str_device

        self.det_config_path = det_config_path
        self.str_config_path = str_config_path
        self.det_model_path = det_model_path
        self.str_model_path = str_model_path

        self.str_model_path2 = str_model_path2

        self.det_class_name2idx = get_class_map('detection')
        self.det_class_idx2name = {v:k for k, v in self.det_class_name2idx.items()}
        self.det_class_thresholds = detection_class_thresholds

        self.str_class_name2idx = get_class_map('structure')
        self.str_class_idx2name = {v:k for k, v in self.str_class_name2idx.items()}
        self.str_class_thresholds = structure_class_thresholds

        with open(det_config_path, 'r') as f:
            det_config = json.load(f)
        det_args = type('Args', (object,), det_config)
        det_args.device = det_device
        self.det_model, _, _ = build_model(det_args)
        print("Detection model initialized.")

        self.det_model.load_state_dict(torch.load(det_model_path,
                                                map_location=torch.device(det_device)))
        self.det_model.to(det_device)
        self.det_model.eval()
        print("Detection model weights loaded.")

        with open(str_config_path, 'r') as f:
            str_config = json.load(f)
        str_args = type('Args', (object,), str_config)
        str_args.device = str_device
        self.str_model, _, _ = build_model(str_args)
        print("Structure model initialized.")
        self.str_model2, _, _ = build_model(str_args)
        print("Structure model initialized.")

        self.str_model.load_state_dict(torch.load(str_model_path,
                                                map_location=torch.device(str_device)))
        self.str_model.to(str_device)
        self.str_model.eval()
        print("Structure model weights loaded.")

        self.str_model2.load_state_dict(torch.load(str_model_path2,
                                        map_location=torch.device(str_device)))
        self.str_model2.to(str_device)
        self.str_model2.eval()
        print("Structure model weights loaded.")

    def __call__(self, page_image):
        return self.extract(self, page_image)

    def detect(self, img, crop_padding=10):
        """For detecting tables from images"""

        out_formats = {}

        # Transform the image how the model expects it
        img_tensor = detection_transform(img)

        # Run input image through the model
        outputs = self.det_model([img_tensor.to(self.det_device)])

        # Post-process detected objects, assign class labels
        objects = outputs_to_objects(outputs, img.size, self.det_class_idx2name)

        # Crop image for detected table
        tables_crops = objects_to_crops(img, objects, self.det_class_thresholds,
                                        padding=crop_padding)
        out_formats['crops'] = tables_crops

        return out_formats

    def recognize(self, img):
        """For tables with height less than 700"""

        out_formats = {}

        # Transform the image how the model expects it
        img_tensor = structure_transform(img)

        # Run input image through the model
        outputs = self.str_model([img_tensor.to(self.str_device)])

        # Post-process detected objects, assign class labels
        objects = outputs_to_objects(outputs, img.size, self.str_class_idx2name)
        out_formats['objects'] = objects

        return out_formats
    
    def recognize2(self, img):
        """For tables with height more than 700"""

        out_formats = {}

        part1 = img.crop((0, 0, img.width, img.height/2 + 50))
        part2 = img.crop((0, img.height/2 - 50, img.width, img.height))

        # Transform the image how the model expects it
        img_tensor1 = structure_transform(part1)
        img_tensor2 = structure_transform(part2)

        # Run input image through the model
        outputs1 = self.str_model([img_tensor1.to(self.str_device)])
        outputs2 = self.str_model([img_tensor2.to(self.str_device)])

        # Post-process detected objects, assign class labels
        objects1 = outputs_to_objects(outputs1, part1.size, self.str_class_idx2name)
        objects2 = outputs_to_objects(outputs2, part2.size, self.str_class_idx2name)
        # import pdb;pdb.set_trace()
        for j,obj in enumerate(objects1):
            if objects1[j]['label'] == 'table':
                objects1[j]['bbox'][3] += img.height/2 - 50
            # elif objects1[j]['label'] == 'table':

        for i,obj in enumerate(objects2):
            if objects2[i]['label'] == 'table':
                objects2.remove(obj)
            if objects2[i]['label'] == 'table row':
                objects2[i]['bbox'][1] += img.height/2 - 50
                objects2[i]['bbox'][3] += img.height/2 -50

        objects =objects1+objects2
        # objects =objects1

        out_formats['objects'] = objects

        return out_formats

def use_ocr(img):
    """Use Paddleocr to get the coordinates of text in the detected table images"""

    ocr = PaddleOCR(use_angle_cls=True, lang='es',show_log = False)
    img = np.array(img)
    # noise = np.random.normal(scale=25, size=img.shape)
    # img = img+noise
    # img = np.clip(img, 0, 255).astype(np.uint8)
    # x = Image.fromarray(img)
    # x.save('x.jpg')
    ocr_result = ocr.ocr(img, cls=True)
    
    tokens = []
    for i, res in enumerate(ocr_result[0]):

        xmin, ymin = res[0][0]
        xmax, ymax = res[0][0]

        for point in res[0][1:]:
            x, y = point
            xmin = min(xmin, x)
            ymin = min(ymin, y)
            xmax = max(xmax, x)
            ymax = max(ymax, y)

            bbox = [xmin, ymin, xmax, ymax]

        tokens.append({
            "bbox": bbox,
            "text": res[1][0],
            "flags": 0,
            "span_num": i,
            "line_num": 0,
            "block_num": 0
        })
    return tokens

def assign_text_to_cell(cells, tokens):
    """ If the coordinates of text(tokens) obtained from ocr and the coordinates of cells obtained from 
    table transformer structure recognition model overlap, then the token is assigned to that specific cell
    based on the overlap percentage"""

    # If text coordinates overlap with a cell, assign it to that cell
    for cell in cells: 
        for token_num, token in enumerate(tokens):
            rect1 = Rect(cell['bbox'][0],cell['bbox'][1],cell['bbox'][2],cell['bbox'][3])
            rect2 = Rect(token['bbox'][0],token['bbox'][1],token['bbox'][2],token['bbox'][3])
            intersect = rect1.intersect(rect2)
            if intersect != 0.0 and rect2.get_area() !=0.0:
                overlap = intersect.get_area()/rect2.get_area()
                if overlap:
                    if 'text' in cell:
                        cell['text'].append(token['text'])
                        cell['token_num'].append(token_num)
                        cell['conf_score'].append(overlap)
                    else:
                        cell['text']=[token['text']]     
                        cell['token_num'] = [token_num]
                        cell['conf_score'] = [overlap]
    
    # Based on the overlap percentage, remove duplicate text entries
    for id, i in enumerate(cells):  
        for jd, j in enumerate(cells):
            if 'text' in i and 'text' in j and id != jd:
                for itd, i_token_num in enumerate(i['token_num']):
                    for jtd, j_token_num in enumerate(j['token_num']):
                        if i_token_num == j_token_num:
                            if i['conf_score'][itd] >= j['conf_score'][jtd]:
                                cells[jd]['text'][jtd] = ''
                            else:
                                cells[id]['text'][itd] = ''
    return cells

def create_html(cells):
    """Based on the row and column number, create html table output"""

    num_rows = max(item['row_num'][0] for item in cells) + 1
    num_cols = max(max(item['col_num']) for item in cells) + 1

    table_data = [['' for _ in range(num_cols)] for _ in range(num_rows)]

    for cell_data in cells:
        row = cell_data['row_num'][0]
        col_nums = cell_data['col_num']

        # Merge cell text for spanning cells
        cell_text = ' '.join(cell_data.get('text', []))

        for col in col_nums:
            table_data[row][col] = cell_text

    # Create an empty HTML table
    html_table = '<table border="1">\n'

    # Iterate through rows and columns to populate the table
    for row in table_data:
        html_table += '<tr>\n'
        for cell_text in row:
            html_table += f'<td>{cell_text}</td>\n'
        html_table += '</tr>\n'

    html_table += '</table>'
    return html_table

def output_result(key, val, args, img_file, cropped_img, count):
    """Perform postprocessing on the objects, create tokens from the 
    images, create html output of the detected table"""
    
    cells, val = postprocess(val, structure_class_thresholds)
    if cells:
        if key == 'objects':
            if args.verbose:
                print(val)
            
            out_file = img_file.replace(".jpg", f"_objects_{count}.json")

            with open(os.path.join(args.out_dir, out_file), 'w') as f:
                json.dump(val, f)
         
            tokens = use_ocr(cropped_img) #Use ocr on cropped image
            cells = assign_text_to_cell(cells, tokens)
            
            html_table = create_html(cells)
            html_path = os.path.join(args.out_dir, img_file.replace('.jpg', f'_table_{count}.html'))
            with open(html_path, 'w') as f:
                f.write(html_table)
                    
            print(f'Successfully converted {img_file}_table_{count} to html')

            if args.visualize:
                structured_table = draw_structure(cropped_img, val)
                out_file = img_file.replace(".jpg", f"_table_{count}.jpg")
                out_image = os.path.join(args.out_dir, out_file)
                cv2.imwrite(out_image, structured_table)
    else:
        print(f'Rejected {img_file}_table_{count} during postprocessing')

def main():
    args = get_args()
    print(args.__dict__)
    print('-' * 100)

    if not args.out_dir is None and not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Create inference pipeline
    print("Creating inference pipeline")
    pipe = TableExtractionPipeline(det_device=args.detection_device,
                                   str_device=args.structure_device,
                                   det_config_path=args.detection_config_path, 
                                   det_model_path=args.detection_model_path,
                                   str_config_path=args.structure_config_path, 
                                   str_model_path=args.structure_model_path,
                                   str_model_path2=args.structure_model_path2,)

    # Load images
    img_files = os.listdir(args.image_dir)
    num_files = len(img_files)
    random.shuffle(img_files)

    for count, img_file in enumerate(img_files):
        print("({}/{})".format(count+1, num_files))
        img_path = os.path.join(args.image_dir, img_file)
        img = Image.open(img_path)
        print(f"Image loaded.{img_file}")

        detected_tables = pipe.detect(img)
        print("Table(s) detected.")
        table_count = len(detected_tables['crops'])
        print(f"No. of tables in  the image are: {table_count}")
        # if table_count>0:
        #     print('Title of the table is:', get_title(img))

        for key, val in detected_tables.items():
            count = 0   
            for image in val:
                cropped_img = image['image']
                if cropped_img.height <= 700:
                    extracted_table = pipe.recognize(cropped_img)
                    print('model 1')
                else:
                    extracted_table = pipe.recognize2(cropped_img)
                    print('model 2')
                print("Table(s) recognized.")

                for key, val in extracted_table.items():
                    output_result(key, val, args, img_file, cropped_img, count)
                count+=1

if __name__ == "__main__":
    main()
