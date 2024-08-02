# python-face-lambda

##### python-face-lambda is a serverless face recognition application built using AWS Lambda. The project contains three main functions: reg, train, and rec, which are designed to capture, train, and recognize faces using a k-nearest neighbors (KNN) model.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Function Details](#function-details)
  - [reg](#reg)
  - [train](#train)
  - [rec](#rec)
- [Docker Support](#docker-support)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project provides a scalable solution for face recognition using AWS Lambda. It leverages a KNN model to perform face recognition tasks and includes functions to register new faces, train the model, and recognize faces in real-time.

## Features

- **Serverless**: Utilizes AWS Lambda for serverless deployment.
- **Face Registration**: Captures faces and stores them in a database.
- **Model Training**: Trains a KNN model with registered faces.
- **Face Recognition**: Recognizes and identifies faces using the trained model.
- **Dockerized**: Includes a Dockerfile for containerizing the functions and deploying to AWS Lambda.

## Architecture

![Architecture Diagram](path/to/architecture-diagram.png)

## Getting Started

### Prerequisites

- AWS Account
- AWS CLI
- Docker
- Python 3.8+

### Installation
1. Clone the repository:

```bash
git clone https://github.com/yourusername/python-face-lambda.git
cd python-face-lambda
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure AWS CLI:
```bash
aws configure
```

## Usage

### Deploying to AWS Lambda

#### Build the Docker image:

```bash
docker build -t python-face-lambda .
```

#### Push the Docker image to AWS ECR (Elastic Container Registry):
```bash
aws ecr create-repository --repository-name python-face-lambda
$(aws ecr get-login --no-include-email --region your-region)
docker tag python-face-lambda:latest your-account-id.dkr.ecr.your-region.amazonaws.com/python-face-lambda:latest
docker push your-account-id.dkr.ecr.your-region.amazonaws.com/python-face-lambda:latest
```

#### the Lambda function:
```bash
aws lambda create-function --function-name python-face-lambda --package-type Image --code ImageUri=your-account-id.dkr.ecr.your-region.amazonaws.com/python-face-lambda:latest --role your-lambda-execution-role-arn
```

## Function Details

### reg

The `reg` function captures faces and stores the data in a database.

- **Input**: Image data
- **Output**: Face data stored in the database

### train

The `train` function trains a KNN model using the registered faces.

- **Input**: Face data from the database
- **Output**: Trained KNN model stored in S3

### rec

The `rec` function recognizes faces using the trained KNN model.

- **Input**: Image data
- **Output**: Recognized face information

## Docker Support

The project includes a `Dockerfile` to containerize the functions and deploy them to AWS Lambda. The Docker image is built and pushed to AWS ECR, and then used to create the Lambda functions.

## Contributing

Contributions are welcome! Please read the contributing guidelines for more information.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

