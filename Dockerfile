FROM tensorflow/tensorflow:1.5.0-rc0-devel

RUN pip install -U pip

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    python-dev \
    python-numpy \
    python-skimage \
    python-tk \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libjasper-dev \
    libdc1394-22-dev \
    qt5-default \
    wget \
    vim

RUN git clone https://github.com/opencv/opencv.git /root/opencv && \
	cd /root/opencv && \
        git checkout 2.4 && \
	mkdir build && \
	cd build && \
	cmake -DWITH_QT=ON -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_GDAL=ON -DWITH_XINE=ON -DBUILD_EXAMPLES=ON .. && \
	make -j"$(nproc)"  && \
	make install && \
	ldconfig && \
        echo 'ln /dev/null /dev/raw1394' >> ~/.bashrc

RUN ln /dev/null /dev/raw1394

RUN cd /root && git clone https://github.com/hiveml/tensorflow-grad-cam

WORKDIR /root/tensorflow-grad-cam

RUN cd imagenet && ./get_checkpoint.sh

CMD /bin/bash

