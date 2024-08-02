# Use AWS official Python image as the base image
FROM public.ecr.aws/lambda/python:3.9

# Copy the requirements file into the container at /app
COPY requirements.txt ${LAMBDA_TASK_ROOT} 

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}" 

# Copy the rest of the application code into the container
COPY facerec_train_knn.py ${LAMBDA_TASK_ROOT}
#COPY .env ${LAMBDA_TASK_ROOT}

# Command to run the application
CMD ["facerec_train_knn.lambda_handler"]


# Usage
# docker build --platform linux/amd64 -t facerec_train_knn_lambda:test .
# docker run --platform linux/amd64 -p 9000:8080 facerec_train_knn_lambda:test


