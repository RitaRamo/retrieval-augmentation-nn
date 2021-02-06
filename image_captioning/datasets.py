import torch
from torch.utils.data import Dataset
import h5py
import json
import os
from PIL import Image
import torchvision.transforms as transforms
import faiss  
import numpy as np                 # make faiss available


class TrainRetrievalDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.data_folder=data_folder

        with open(os.path.join(data_folder, "TRAIN" + '_IMGPATHS_' + data_name + '.json'), 'r') as j:
            self.imgpaths = json.load(j)

        #self.imgpaths=self.imgpaths[:10]
        #print("self images", self.imgpaths)
        ##TODO:REMOVE

        # Total number of datapoints
        self.dataset_size = len(self.imgpaths)
        #print("this is the actual len on begin init", self.dataset_size)


        self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD
                                     std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image    
        img = Image.open(self.data_folder+"/"+self.imgpaths[i])
        img = self.transform(img)
        #print("i of retrieval dataset",i)
        return img, i
   
    def __len__(self):
        #print("this is the actual len on __len", self.dataset_size)
        return self.dataset_size



class ImageRetrieval():

    def __init__(self, dim_examples, encoder, train_dataloader_images, device):
        #print("self dim exam", dim_examples)
        self.datastore = faiss.IndexFlatL2(dim_examples) #datastore
        self.encoder= encoder

        #data
        self.device=device
        self.imgs_indexes_of_dataloader = torch.tensor([]).long().to(device)
        #print("self.imgs_indexes_of_dataloader type", self.imgs_indexes_of_dataloader)

        #print("len img dataloader", self.imgs_indexes_of_dataloader.size())
        self._add_examples(train_dataloader_images)
        #print("len img dataloader final", self.imgs_indexes_of_dataloader.size())
        #print("como ficou img dataloader final", self.imgs_indexes_of_dataloader)


    def _add_examples(self, train_dataloader_images):
        print("\nadding input examples to datastore (retrieval)")
        for i, (imgs, imgs_indexes) in enumerate(train_dataloader_images):
            #add to the datastore
            imgs=imgs.to(self.device)
            imgs_indexes = imgs_indexes.long().to(self.device)
            #print("img index type", imgs_indexes)
            encoder_output = self.encoder(imgs)

            encoder_output = encoder_output.view(encoder_output.size()[0], -1, encoder_output.size()[-1])
            input_img = encoder_output.mean(dim=1)
            
            self.datastore.add(input_img.cpu().numpy())

            if i%5==0:
                print("i and img index of ImageRetrival",i, imgs_indexes)
                print("n of examples", self.datastore.ntotal)
            self.imgs_indexes_of_dataloader= torch.cat((self.imgs_indexes_of_dataloader,imgs_indexes))



    def retrieve_nearest_for_train_query(self, query_img, k=2):
        #print("self query img", query_img)
        D, I = self.datastore.search(query_img, k)     # actual search
        #print("all nearest", I)
        #print("I firt", I[:,0])
        #print("if you choose the first", self.imgs_indexes_of_dataloader[I[:,0]])
        nearest_input = self.imgs_indexes_of_dataloader[I[:,1]]
        #print("the nearest input is actual the second for training", nearest_input)
        #nearest_input = I[0,1]
        #print("actual nearest_input", nearest_input)
        return nearest_input

    def retrieve_nearest_for_val_or_test_query(self, query_img, k=1):
        D, I = self.datastore.search(query_img, k)     # actual search
        nearest_input = self.imgs_indexes_of_dataloader[I[:,0]]
        #print("all nearest", I)
        #print("the nearest input", nearest_input)
        return nearest_input


class NearestCaptionAvgDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        print("entrei no infico aqui")

        self.data_folder=data_folder
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Captions per image
        self.cpi = 5

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Load image_id (completely into memory)
        with open(os.path.join(data_folder, self.split + '_IMGIDS_' + data_name + '.json'), 'r') as j:
            self.imgids = json.load(j)

        with open(os.path.join(data_folder, self.split + '_IMGPATHS_' + data_name + '.json'), 'r') as j:
            self.imgpaths = json.load(j)

        # Total number of datapoints
        self.dataset_size = len(self.captions)
        
        self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD
                                     std=[0.229, 0.224, 0.225])
        ])
 

    def __getitem__(self, i):
        print("\n nearest caption dataset: caption i",i)
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = Image.open(self.data_folder+"/"+self.imgpaths[i // self.cpi])
        print("path image i",self.imgpaths[i // self.cpi])

        img = self.transform(img)     
            
        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split == 'TRAIN':
            return img, caption, caplen, i// self.cpi #n+ da imagem
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            if self.split == "TEST":
                img_id = self.imgids[i]
                return img, nearest, caption, caplen, all_captions, img_id
            else:
                return img, nearest, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size

class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.data_folder=data_folder
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        # self.h = h5py.File(os.path.join(
        #     data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        # self.imgs = self.h['images']

        # Captions per image
        self.cpi = 5

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Load image_id (completely into memory)
        with open(os.path.join(data_folder, self.split + '_IMGIDS_' + data_name + '.json'), 'r') as j:
            self.imgids = json.load(j)

        with open(os.path.join(data_folder, self.split + '_IMGPATHS_' + data_name + '.json'), 'r') as j:
            self.imgpaths = json.load(j)

        # Total number of datapoints
        self.dataset_size = len(self.captions)

        self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD
                                     std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        #print("path image; index of dataset; img id", self.imgpaths[i // self.cpi], i, i // self.cpi)
        
        img = Image.open(self.data_folder+"/"+self.imgpaths[i // self.cpi])


        img = self.transform(img)
        
        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split == 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            if self.split == "TEST":
                img_id = self.imgids[i]
                return img, caption, caplen, all_captions, img_id
            else:
                return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
