# Use AWS official Python image as the base image
FROM public.ecr.aws/lambda/python:3.12

# Install system dependencies using dnf (Amazon Linux 2023 package manager)
RUN dnf -y update \
    && dnf -y install make gcc-c++ openssl-devel \
                     libtiff libjpeg libpng libwebp freetype-devel \
                     libSM libXrender libXext nano libglvnd-glx \
                     cmake git boost-devel \
    && dnf clean all

# Install any dependencies specified in requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install any other dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY lambda_lite_version/facerec_register.py ${LAMBDA_TASK_ROOT}
# COPY face_sample ${LAMBDA_TASK_ROOT}/face_sample
# COPY testfile ${LAMBDA_TASK_ROOT}/testfile

# Set the CMD to your handler
CMD ["facerec_register.lambda_handler"]

# Install cmake manually (if needed)
# RUN wget https://github.com/Kitware/CMake/releases/download/v3.29.3/cmake-3.29.3.tar.gz \
#     && tar -xvzf cmake-3.29.3.tar.gz \
#     && cd cmake-3.29.3 \
#     && ./bootstrap \
#     && make \
#     && make install \
#     && cd .. \
#     && rm -rf cmake-3.29.3 cmake-3.29.3.tar.gz

# Usage
# docker build --platform linux/amd64 -t facerec_register_lambda:test .
# docker run --platform linux/amd64 -p 9000:8080 facerec_register_lambda:test


