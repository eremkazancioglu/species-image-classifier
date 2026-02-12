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
