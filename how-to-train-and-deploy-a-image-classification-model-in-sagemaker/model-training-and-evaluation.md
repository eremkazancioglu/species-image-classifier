# Model training and evaluation

## Overview

For the multi-class image classification problem that I am trying to solve, where an image would be identified to belong either to one of the four target species or to a catch all "other" class, fine-tuning a pre-trained, neural network based image classification model, rather than training one from scratch seemed to be a good choice. There are a bunch of pre-trained models [out there](https://huggingface.co/models?pipeline_tag=image-classification) that use a Convolutional Neural Network (CNN) approach and exhibit good performance. These models have already incorporated a lot of information about high-level features of images such as edges, shapes etc. that is useful for any image classification task and there is no need spend resources to "re-learn" this information. Instead, by fine-tuning an existing image classification model, we can focus on re-training the model to focus on features that are relevant to the specific task at hand, i.e. determining if an image belongs to one of the four target species.

I will use this page to document my modeling approach and also to keep track of the resources that I used, for posterity, because this is my first hands-on experience with deep neural network models using PyTorch. Otherwise, I will forget what I did and why I did it.

Finally, in all stages of model fine tuning, deployment and evaluation, I collaborated with Claude both to generate Python scripts but also to inform myself about the specifics of how neural networks are trained and inner workings of PyTorch. I figured that it would help me a lot to have a working script first and then learn by poking around, experimenting and asking questions about specific parts, rather than trying to develop something from scratch and spend most of my time trying to debug rookie mistakes that I would inevitably make.

## Methodology

### Model and environment choice

My initial thought was that I would fine tune a model locally on my Apple M4 laptop to save time from figuring out how to set up an AWS SageMaker account, figure out costs, monitor runs etc. I decided I would start with [ResNet-50](https://huggingface.co/microsoft/resnet-50), a popular CNN-based image classification model that is available in PyTorch. During my experimentation with this model and PyTorch, it looked like training was painfully slow, not finishing over a few days. Various diagnostic trials showed that this is probably due to backward pass running very inefficiently on [Apple MPS](https://developer.apple.com/documentation/metalperformanceshaders) (specialized GPU libraries) but I was not able to address this issue.

Due to these performance issues, I first pivoted towards another family of CNN-based models called [EfficientNet](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b1.html) that offers higher computational efficiency as it is designed specifically to be run on resource-constrained environments such as mobile devices. Using the `efficientnet_b1`  model still showed slow backward pass performance, further indicating that it's probably something about the hardware architecture of my laptop that is causing issues. Therefore, I decided to run model training on AWS SageMaker and circumvent these issues. SageMaker also makes sense because I would want to make the final model through an API available anyway, which means that I would have needed to go to a service like SageMaker to host the model even if I was able to efficiently train a model locally.

### AWS SageMaker account setup

Getting AWS account ready to be able to run training and deploy model enpoints was relatively straightforward. I first created an IAM role using root access, and associated the following policy with this role.

<details>

<summary>IAM Policy</summary>

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "SageMakerFullAccess",
            "Effect": "Allow",
            "Action": [
                "sagemaker:*"
            ],
            "Resource": "*"
        },
        {
            "Sid": "S3AccessForTraining",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket",
                "s3:CreateBucket",
                "s3:GetBucketLocation",
                "s3:ListAllMyBuckets"
            ],
            "Resource": [
                "arn:aws:s3:::*sagemaker*",
                "arn:aws:s3:::*sagemaker*/*",
                "arn:aws:s3:::*training*",
                "arn:aws:s3:::*training*/*",
                "arn:aws:s3:::inaturalist-open-data",
                "arn:aws:s3:::inaturalist-open-data/*"
            ]
        },
        {
            "Sid": "CloudWatchAccess",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:DescribeLogGroups",
                "logs:DescribeLogStreams",
                "logs:PutLogEvents",
                "logs:GetLogEvents",
                "logs:FilterLogEvents"
            ],
            "Resource": "arn:aws:logs:*:*:*"
        },
        {
            "Sid": "ECRAccess",
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage"
            ],
            "Resource": "*"
        },
        {
            "Sid": "IAMPassRole",
            "Effect": "Allow",
            "Action": [
                "iam:GetRole",
                "iam:PassRole"
            ],
            "Resource": [
                "arn:aws:iam::*:role/service-role/AmazonSageMaker-ExecutionRole*",
                "arn:aws:iam::*:role/*SageMaker*"
            ]
        },
        {
            "Sid": "STSAccess",
            "Effect": "Allow",
            "Action": [
                "sts:GetCallerIdentity"
            ],
            "Resource": "*"
        }
    ]
}
```

</details>

Finally, I created an access key for this role which I will need to be able to run training scripts. When you generate a new key, and only at that time of creation, you are provided with a secret key which you need to provide during configuration of AWS CLI using `aws configure`.

### Model training and evaluation scripts

Training and evaluation is orchestrated through two scripts. The entry point to the training job in SageMaker is the script [train.py](https://github.com/eremkazancioglu/species-image-classifier/blob/main/sagemaker-run/train.py) which contains all functions to load a pre-trained model, prepare the training data to be used to fine tune the model, perform K-fold cross validation and prepare evaluation metrics. This script is referenced through another script, [launch\_training.py](https://github.com/eremkazancioglu/species-image-classifier/blob/main/sagemaker-run/launch_training.py), which also contains various configuration parameters for training and evaluation of the model as well as SageMaker resource utilization.

Below I will go through these scripts and highlight several critical parts that demonstrate how data preparation, training and evaluation are executed and also indicate design choices.

#### train.py

I will try and lay this process out as linearly as possible.

1. **Preparing the labeled dataset**

This is a straightforward step where we take the CSV file that we prepared at the end of [Training data collection](training-data-collection.md), encode labels of the four target species and the "other" class, and return a labeled dataset as well as label-species pairs. Below is the key part of the `prepare_data` function where label encoding happens.

```python
def prepare_data(csv_file, target_species, downsample_other=False, downsample_frac=0.1):
    (...)
    # Create binary classification: target species vs 'other'
    df['label_name'] = df['species_name'].apply(
        lambda x: x if x in target_species else 'other'
    )
    
    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label_name'])
    (...)
```

Note that there are two arguments `downsample_other` and `downsample_frac`. These allow us to use only a small, configurable fraction of the "other" class for training and evaluation, which is utilized only in the "dry run" mode that I wanted to have as an option to check quickly that the whole training and evaluation pipeline works properly without using the full dataset. This is configured through the script that launches the pipeline and I will get to that later.

2. **Preparing stratified folds for training and validation**

This is also a straightforward step where we determine indices of stratified folds that we we will use to train and validate the model. This happens within the `main` call of the script using `StratifiedKFold` imported from `sklearn`. I don't think there is anything else to point out here. Once we have the folds, we kick off the training process sequentially for each fold.

3. **Determining transformations that are to be applied to each image prior to training and validation**

For each of the folds that we determined, we will make multiple passes through the neural network as we train, validate and repeat where each pass is called an "epoch" (more on this later). At the beginning of each epoch, we perform a set of transformations on the training data which essentially aim both to standardize the images and resulting tensor data so they are compatible with the pre-training of the model (in this case `efficientnet_b1`), and to introduce some variation to images that we might expect to encounter in real-life scenarios (e.g. people will not always take pictures of images perfectly centered in one type of orientation). We perform transformations on the validation data as well, but these are just aimed towards making images and tensor data compatible with the underlying pre-trained model.

These transformations are set up by the function `get_transforms` below.

```python
 def get_transforms(img_size):
    """Define training and validation transforms"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform
```

First thing to note here is that we pass image size as an argument to this function. This ensures that resized images are compatible with the native resolution the underlying model is pre-trained with. Here, I am using `efficient_b1` which was pre-trained on 240x240 resolution images, so this argument will be set to 240 in this case.

Second, once we apply transformations and turn images into tensors, we normalize these tensors to the distribution of pixel values of the three channels (R, G, B) in ImageNet, which is what `efficientnet_b1` was pre-trained on. With this normalization, input tensor values are aligned with the information on low-level features learned by early layers of the pre-trained model, such as what kind of pixel values correspond to edges or certain types of textures. This is particularly important because, as I will show below, I utilize differential learning rates for earlier and later layers such that we focus more of the learning on features specific to the problem at hand. Without this normalization, the model might struggle to leverage low-level features from the pre-trained model.

4. **Train and validate the model for each fold**

This is where the pre-trained image classification model, `efficientnet_b1`, is fine tuned and evaluated using the training and validation datasets that are determined by each fold. There are a number of steps here that I was newly exposed through PyTorch so I will call those out here and document what is going on.

* **`SpeciesDataset` class:** This is a subclass of PyTorch's `Dataset` class that implements a function (`__getitem__`) that takes an index for an image file, grabs that image file from a specified location, performs any specified image transformations (see above), and returns the image as well as the label associated with that image. I used this class to generate training and validation datasets as shown below.

```python
train_dataset = SpeciesDataset(train_df, config['data_dir'], train_transform)
val_dataset = SpeciesDataset(val_df, config['data_dir'], val_transform)
```

* **`DataLoader` class:** Instances of this class is what actually utilize the `__getitem__` function of `SpeciesDataset` and puts the training and validation data together. Image data is processed in batches where, in each epoch, batches of images are passed forward and backward through the neural network, one batch at a time. For the training pass, images are first shuffled to ensure that each batch represents a variety of images and any inherent structure in the dataset (e.g. images of a species come one after the other) is removed. For the validation pass, no such shuffling is necessary as we are just calculating the performance of the model following the training pass. Data loaders are set up as below.

```python
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                          shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                        shuffle=False, num_workers=4, pin_memory=True)
```

