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



3. **Train and validate the model for each fold**

This is where the pre-trained image classification model, `efficientnet_b1`, is fine tuned and evaluated using the training and validation datasets that are determined by each fold. There are a number of steps here that I was newly exposed through PyTorch so I will call those out here and document what is going on.

* **`SpeciesDataset` class:** This is a subclass of PyTorch's `Dataset` class that implements a function (`__getitem__`) that takes an index for an image file, grabs that image file from a specified location, performs any specified image transformations, and returns the image as well as the label associated with that image. I used this class to generate training and validation datasets as shown below.

```python
train_dataset = SpeciesDataset(train_df, config['data_dir'], train_transform)
val_dataset = SpeciesDataset(val_df, config['data_dir'], val_transform)
```

