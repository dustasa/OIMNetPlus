import scipy.io as sio
import os, glob
import xml.etree.ElementTree as ET
import xml.dom.minidom as DOM
import cv2
from pickle import OBJ
from posix import listdir
from matplotlib.pyplot import text

pwd = os.getcwd()
mat_input = r'/home/aousn/dataset/PRW_VOC/mat/'
xml_output = r'/home/aousn/dataset/PRW_VOC/xml_/'
img_input = r'/home/aousn/dataset/PRW_VOC/JPEGImages'


def read_mat(mat_input):
    file = os.listdir(mat_input)
    for i, file_name in enumerate(file):
        img = cv2.imread(glob.glob(os.path.join(img_input, file_name.replace('jpg.mat', 'jpg')))[0])
        print(f'img{img}')
        size_height = img.shape[0]
        size_width = img.shape[1]
        point_list = []
        xml_path = mat_input + file_name.replace('jpg.mat', 'xml')
        print("xml_path:", xml_path)
        mat_data = sio.loadmat(glob.glob(os.path.join(mat_input + file_name))[0])
        print(mat_data.keys(), type(mat_data))
        for key in mat_data.keys():
            if key == 'box_new':
                point_list.append(mat_data['box_new'])
        print(point_list)
        save_xml(file_name, point_list, size_h=size_height, size_w=size_width)


def save_xml(file_name, point_list, size_h, size_w):
    # 首先创建根节点
    root = ET.Element('annotation')
    # 添加子节点SubElement(父节点Element对象， Tag字符串格式， Attribute字典格式)
    folder1 = ET.SubElement(root, 'folder')
    filename1 = ET.SubElement(root, 'filename')
    # path1 = ET.SubElement(root, 'path')
    source1 = ET.SubElement(root, 'source')
    size1 = ET.SubElement(root, 'size')
    segmented1 = ET.SubElement(root, 'segmented')
    for i, value in enumerate(point_list[0]):
        if value[0] != -2.:
            object1 = ET.SubElement(root, 'object')
            name1 = ET.SubElement(object1, 'name')
            person_id = ET.SubElement(object1, 'personID')
            # # pose1 = ET.SubElement(object1, 'pose')
            # # truncated1 = ET.SubElement(object1, 'truncated')
            # # difficult1 = ET.SubElement(object1, 'difficult')
            bndbox = ET.SubElement(object1, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            ymin = ET.SubElement(bndbox, 'ymin')
            xmax = ET.SubElement(bndbox, 'xmax')
            ymax = ET.SubElement(bndbox, 'ymax')
            name1.text = 'person'
            # pose1.text = 'Unspecified'
            # truncated1.text = '0'
            # difficult1.text = '0'
            print('value[0]:', type(value[0]), value[0])  # id
            print('value[1]:', type(value[1]), value[1])  # left top X1
            print('value[2]:', type(value[2]), value[2])  # left top Y1
            print('value[3]:', type(value[3]), value[3])  # bbox width
            print('value[4]:', type(value[4]), value[4])  # bbox height
            x1 = int((round(value[1])))
            y1 = int((round(value[2])))
            bbox_w = int((round(value[3])))
            bbox_h = int((round(value[4])))
            x2 = x1 + bbox_w
            y2 = y1 + bbox_h

            person_id.text = str(int((round(value[0]))))
            # person id -2: unlabeled person, should be ignored

            xmin.text = str(x1)
            ymin.text = str(y1)
            xmax.text = str(x2)
            ymax.text = str(y2)

            # ymin.text = str(int((round(value[1]) * 416) / 600 + 9))
            # xmax.text = str(int((round(value[0]) * 416) / 600 + 9))
            # ymax.text = str(int((round(value[1]) * 416) / 600 - 9))

    # 添加子节点
    database1 = ET.SubElement(source1, 'database')
    width1 = ET.SubElement(size1, 'width')
    height1 = ET.SubElement(size1, 'height')
    depth1 = ET.SubElement(size1, 'depth')

    # 添加text
    folder1.text = 'PRW_VOC'
    filename1.text = file_name.replace('jpg.mat', 'jpg')
    # path1.text = img_input + file_name.replace('jpg.mat', 'jpg')
    database1.text = 'Unknow'
    width1.text = str(size_w)
    height1.text = str(size_h)
    depth1.text = '3'
    segmented1.text = '0'

    # 写入保存并格式化
    word = prettify(root)
    with open(xml_output + file_name.replace('jpg.mat', 'xml'), 'w') as f:
        f.write(word)

        # 将根目录转化为xml树状结构(即ElementTree对象)
        # tree = ET.ElementTree(root)
        # #在终端显示整个xml内容
        # ET.dump(root)
        # #写入xml文件
        # tree.write(xml_output+ '\\' + file_name.replace('mat','xml'), encoding="utf-8", xml_declaration=True,method='xml')


def prettify(elem):
    """将节点转换成字符串，并添加缩进。
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = DOM.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")


# resize图片，按需
def resize(img_list):
    for i in img_list:
        imgname = glob.glob(os.path.join(img_input, i))
        img = cv2.imread(imgname[0])
        print(img.shape[0])
        img = cv2.resize(img, (416, 416))
        cv2.imwrite(imgname[0], img)


# img_list = os.listdir(img_input)
# resize(img_list)


def main():
    read_mat(mat_input)


if __name__ == '__main__':
    main()
