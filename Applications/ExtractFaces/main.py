import cv2,os
import xml.etree.ElementTree as ET

path_to_dataset = r"D:\CampusX Mentorship Programme\Face Mask Detector\Dataset"
# print(len(os.listdir(r"D:\CampusX Mentorship Programme\Face Mask Detector\Dataset\original\images")))
# print(len(os.listdir(r"D:\CampusX Mentorship Programme\Face Mask Detector\Dataset\original\annotations")))
#
def parse_annotation_file(image_name,annotation_name, train=True, margin=5):

    boxes=[]
    # Parsing the xml file corresponsing to the image
    annot = ET.parse(os.path.join(path_to_dataset,r"original\annotations",annotation_name))
    root=annot.getroot()

    # Parsing all the bounding box info with the labels
    for object in root.findall('object'):
        label = object.find('name').text

        xmin = int(object.find('bndbox').find('xmin').text)
        ymin = int(object.find('bndbox').find('ymin').text)
        xmax = int(object.find('bndbox').find('xmax').text)
        ymax = int(object.find('bndbox').find('ymax').text)
        boxes.append([[xmin,ymin,xmax,ymax],label])

    # Cropping the image according to the bounding boxes
    img=cv2.imread(os.path.join(path_to_dataset,r"original\images",image_name))

    for index,box in enumerate(boxes):
        new_img = img[box[0][1]:box[0][3],box[0][0]:box[0][2]]
        new_img_name = image_name.split('.')[0]+"__"+str(index)+".png"

        if train:
            cv2.imwrite(os.path.join(path_to_dataset,"Cropped Image Dataset","train",box[1],new_img_name), new_img)
        else:
            cv2.imwrite(os.path.join(path_to_dataset, "Cropped Image Dataset", "test", box[1], new_img_name), new_img)





i=0
for image in os.listdir(os.path.join(path_to_dataset,r"original\images"))[:600]:
    print(f"Generating Training File: {i}")
    i+=1
    parse_annotation_file(image,image.split('.')[0]+".xml", train=True)
i=0
for image in os.listdir(os.path.join(path_to_dataset, r"original\images"))[600:]:
    print(f"Generating Testing File: {i}")
    i+=1
    parse_annotation_file(image, image.split('.')[0] + ".xml", train=False)

