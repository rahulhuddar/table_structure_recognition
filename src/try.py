import cv2
from PIL import Image
import numpy as np

img = Image.open('/home/lnv-25/Work/Andrea/Table_Extraction/table-transformer/src/input_1/01-09-2021-15-54-05_01-09-2021-15-54-01-530_Cencosud Orden de Compra 6600123731.pdf_page_1.jpg')
img = np.array(img)

for det_table in obj:
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

    if det_table['label'] == 'table row':
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), edgecolor_bgr, linewidth)

    elif det_table['label'] == 'table column':
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), edgecolor_bgr, linewidth)
        
    else:
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), edgecolor_bgr, linewidth)

img = Image.fromarray(img)
img.save('x.jpg')



































[
    {'label': 'table row', 'score': 0.9923446774482727, 'bbox': [8.911836624145508, 1166.060302734375, 1523.865478515625, 1243.48779296875]},
    {'label': 'table row', 'score': 0.9949324727058411, 'bbox': [9.275848388671875, 140.78553771972656, 1523.3695068359375, 278.5484313964844]},
    {'label': 'table column', 'score': 0.999356210231781, 'bbox': [105.83789825439453, 8.836332321166992, 231.91748046875, 1288.9869384765625]},
    {'label': 'table row', 'score': 0.999347984790802, 'bbox': [8.023035049438477, 1029.6319580078125, 1523.5711669921875, 1164.075927734375]},
    {'label': 'table row', 'score': 0.9993222951889038, 'bbox': [8.853187561035156, 9.222533226013184, 1524.8778076171875, 141.12416076660156]},
    {'label': 'table column', 'score': 0.9999765157699585, 'bbox': [833.6884765625, 7.746237754821777, 1057.978271484375, 1289.8145751953125]},
    {'label': 'table row', 'score': 0.9836503863334656, 'bbox': [8.846942901611328, 490.19439697265625, 1521.5321044921875, 586.912353515625]},
    {'label': 'table column', 'score': 0.9991036057472229, 'bbox': [234.1214141845703, 8.201025009155273, 517.4229125976562, 1290.0509033203125]},
    {'label': 'table row', 'score': 0.9789383411407471, 'bbox': [8.580894470214844, 968.7276611328125, 1524.1488037109375, 1033.6416015625]},
    {'label': 'table', 'score': 0.9999088048934937, 'bbox': [10.883609771728516, 8.025481224060059, 1524.9451904296875, 1291.225341796875]},
    {'label': 'table column', 'score': 0.9998888969421387, 'bbox': [7.937642574310303, 9.41117000579834, 104.25164794921875, 1288.2615966796875]},
    {'label': 'table row', 'score': 0.9740317463874817, 'bbox': [9.800313949584961, 280.8989562988281, 1523.1011962890625, 345.45941162109375]},
    {'label': 'table spanning cell', 'score': 0.5318634510040283, 'bbox': [8.672985076904297, 963.054931640625, 1525.8062744140625, 1027.3201904296875]},
    {'label': 'table column', 'score': 0.9993932247161865, 'bbox': [1325.47314453125, 9.424242973327637, 1523.37158203125, 1289.5140380859375]},
    {'label': 'table row', 'score': 0.9837881326675415, 'bbox': [8.427473068237305, 699.0889282226562, 1523.6656494140625, 771.9200439453125]},
    {'label': 'table row', 'score': 0.9933528900146484, 'bbox': [10.564231872558594, 1240.703857421875, 1526.62353515625, 1294.6943359375]},
    {'label': 'table column', 'score': 0.9996480941772461, 'bbox': [1165.833740234375, 9.533650398254395, 1321.4798583984375, 1289.507568359375]},
    {'label': 'table row', 'score': 0.9940820336341858, 'bbox': [8.871272087097168, 347.2542724609375, 1522.0308837890625, 481.6600341796875]},
    {'label': 'table column', 'score': 0.9360832571983337, 'bbox': [522.573486328125, 7.88946533203125, 834.7754516601562, 1290.3323974609375]},
    {'label': 'table row', 'score': 0.9979493021965027, 'bbox': [7.859668254852295, 766.3484497070312, 1522.157958984375, 962.0673217773438]},
    {'label': 'table row', 'score': 0.9948691129684448, 'bbox': [8.118778228759766, 573.8359985351562, 1521.0318603515625, 694.6190795898438]},
    {'label': 'table column', 'score': 0.998344898223877, 'bbox': [1062.87060546875, 10.045975685119629, 1162.1834716796875, 1289.78564453125]},
    {'label': 'table spanning cell', 'score': 0.7197341322898865, 'bbox': [7.38437032699585, 892.1986083984375, 1542.3919677734375, 961.5182495117188]},
    {'label': 'table row', 'score': 0.9841586351394653, 'bbox': [8.190332412719727, 682.95458984375, 1540.8927001953125, 772.4135131835938]},
    {'label': 'table row', 'score': 0.98377925157547, 'bbox': [7.803146839141846, 1158.4844970703125, 1540.138427734375, 1234.0618896484375
