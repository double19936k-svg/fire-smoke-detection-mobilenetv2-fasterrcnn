import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class FireDataset(Dataset):

    def __init__(self, root):

        self.image_dir = os.path.join(root, "images")
        self.label_dir = os.path.join(root, "labels")

        self.images = []

        for file in os.listdir(self.image_dir):
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                self.images.append(file)


        self.transform = T.Compose([
            T.ToTensor()
        ])


    def __len__(self):

        return len(self.images)


    def __getitem__(self, index):

        img_name = self.images[index]


        img_path = os.path.join(
            self.image_dir,
            img_name
        )


        label_path = os.path.join(
            self.label_dir,
            img_name.rsplit(".",1)[0]+".txt"
        )


        img = Image.open(img_path).convert("RGB")


        width,height = img.size


        boxes=[]
        labels=[]


        if os.path.exists(label_path):

            with open(label_path,"r") as f:

                for line in f.readlines():

                    data=line.strip().split()


                    if len(data)!=5:
                        continue


                    cls,x,y,w,h = map(float,data)


                    # YOLO格式转换

                    x1=(x-w/2)*width
                    y1=(y-h/2)*height

                    x2=(x+w/2)*width
                    y2=(y+h/2)*height


                    # 删除非法框

                    if x2<=x1 or y2<=y1:
                        continue


                    boxes.append(
                        [x1,y1,x2,y2]
                    )


                    # FasterRCNN类别从1开始

                    labels.append(
                        int(cls)+1
                    )



        if len(boxes)==0:

            boxes=torch.zeros(
                (0,4),
                dtype=torch.float32
            )

            labels=torch.zeros(
                (0,),
                dtype=torch.int64
            )


        else:

            boxes=torch.tensor(
                boxes,
                dtype=torch.float32
            )

            labels=torch.tensor(
                labels,
                dtype=torch.int64
            )



        target={

            "boxes":boxes,

            "labels":labels

        }


        img=self.transform(img)


        return img,target



def collate_fn(batch):

    return tuple(zip(*batch))