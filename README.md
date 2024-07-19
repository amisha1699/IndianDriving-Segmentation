# IndianDriving-Segmentation

This project is focused on semantic segmentation using the Indian Driving Dataset. The primary objective is to fine-tune a pre-trained segmentation model [fcn_resnet50] (https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.fcn_resnet50.html) and assess its performance across multiple metrics. The project encompasses data preprocessing, model training, visualizing loss curves and mean IoU with TensorBoard, and presenting the class-wise performance of the test set based on pixel-wise accuracy, F1-Score, and IoU (Intersection Over Union).

## Dataset
The Indian Driving Dataset (IDD) contains images and their corresponding segmentation masks. The dataset is divided into three splits: train, validation, and test.

## Flow of Project

1. Downloading the dataset
* Visit the [IDD] (https://idd.insaan.iiit.ac.in/) Website and download the dataset. The dataset is also available in `data/IDDCLEAN` folder.

2. Unizip the dataset 
```
!tar -xzvf /Path/to/dataset/"IDD Dataset"/IDDSPLIT.tar.gz -C /content
```

3. Convert masks from RGB to grey scale
```
COLOR_DICT = {
    (128, 64, 128): 0,  # Road
    (244, 35, 232): 2,  # Sidewalk
    (220, 20, 60): 4,   # Person
    (255, 0, 0): 5,     # Rider
    (0, 0, 230): 6,     # Motorcycle
    (119, 11, 32): 7,   # Bicycle
    (0, 0, 142): 9,     # Car
    (0, 0, 70): 10,     # Truck
    (0, 60, 100): 11,   # Bus
    (0, 80, 100): 12,   # Train
    (102, 102, 156): 14 # Wall
}
```

4. Custom Dataset Class
```
class IDDataset(Dataset):
    def __init__(self, csv_file, img_dir, mask_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0])
        mask_name = os.path.join(self.mask_dir, self.data_frame.iloc[idx, 1])

        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('RGB')

        image = np.array(image)
        mask = np.array(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = rgb_to_index(mask)

        return image, mask

# Define transformations
transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
```

5. Data Loaders
Creatimg data loaders for train, validation, and test splits using PyTorch and resizing the images to `512 x 512` during preprocessing from the original `1920 x 1080`.

6. Train Model
Load and train the fcn_resnet50 model using pre-trained network weights. Defining loss function and optimizer.

7. Tensor Board Visualization
Visualizing the loss curves and mean IoU on TensorBoard for better insights into the training process.

8. Model Evaluation
Evaluating the model's performance on the test set through metrics such as pixel-wise accuracy, F1-Score, and IoU.
Calculating Precision, Recall, and Average Precision (AP) by using IoUs spanning from 0 to 1, with intervals of 0.1.