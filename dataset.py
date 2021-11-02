'''
code version 1.0 by hjc (from nju to ucas)
'''

import os
import cv2

from torch.utils.data import Dataset
from torchvision import transforms

from torch.utils.data import DataLoader
from tqdm import tqdm

class WCDataset(Dataset):

    def cal_for_img(self,imgin):

        img_return = (imgin*255-127.5)/128.
        return img_return

    def __init__(self, dataset_path):
        
        training_file=os.path.join(dataset_path,'FR_Train_dev.txt')
        #.replace('\\','/')
        #training_file=tf.replace('\\','/')
        with open(training_file) as f:
            #self.class_num = int(f.readline())
            self.class_num_in_txt=126
            self.class_names = []
            self.images = []
            class_count=0
            for i in range(self.class_num_in_txt):
                words = f.readline().split()
                class_name = ' '.join(words[:-2])
                if not os.path.exists(os.path.join(dataset_path, 'train', class_name)):
                    print(class_name,'not exist')
                    continue

                self.class_names.append(class_name)
                self.images += [
                    (os.path.join(dataset_path, 'train', class_name, 'C%05d.jpg'%(j+1)), class_count) for j in range(int(words[-2]))
                ] + [
                    (os.path.join(dataset_path, 'train', class_name, 'P%05d.jpg'%(j+1)), class_count) for j in range(int(words[-1]))
                ]
                class_count += 1
                # image not turn positive
            self.class_num=class_count
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((116, 100)),
            transforms.RandomCrop((112, 96)),
            transforms.ToTensor(),
            transforms.Lambda(self.cal_for_img),
            #transforms.Lambda(lambda img: (img*255-127.5)/128.),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, label = self.images[idx]
        image = cv2.imread(image_path, 1)
        assert image is not None, 'file %s dose not exist' % image_path

        return self.transform(image), label



if __name__ == '__main__':

    dataloader = DataLoader(
        WCDataset('./'),
        batch_size=10,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    for step, batch in enumerate(tqdm(dataloader, desc='test %s', unit='batch')):
        images,labels= batch
        print(images.shape)

    # for images, labels in dataloader:
    #     print(images.shape)
