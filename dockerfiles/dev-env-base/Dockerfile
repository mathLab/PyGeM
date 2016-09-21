# Author Davide Baroli <davide.baroli@uni.lu>

FROM pygem-base:phusion

# Non-Python utilities and libraries
USER root
RUN apt-get update -y; apt-get install -y --force-yes --fix-missing --no-install-recommends curl git unzip tree subversion vim cmake bison g++ gfortran openmpi-bin pkg-config wget libpcre3-dev bison flex swig libglu1-mesa pyqt4-dev-tools    
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN id PyGeM
RUN chown -R PyGeM:PyGeM $HOME




RUN cd /tmp  && \
    wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh  && \
    chmod +x miniconda.sh && \
    bash miniconda.sh -b -p /usr/local/miniconda
ENV PATH=/usr/local/miniconda/bin:$PATH

RUN echo "PATH=/usr/local/miniconda/bin:$PATH" >> ~/.profile
RUN /bin/bash -c 'source  ~/.profile'

RUN  hash -r  && \
    conda config --set always_yes yes --set changeps1 no  && \
    conda update -q conda  
RUN  conda info -a  && \
    conda create --yes -n test python="2.7";

RUN /bin/bash -c 'source  activate test'  
RUN    conda install --yes numpy scipy matplotlib pip nose vtk  

RUN    conda install --yes -c https://conda.anaconda.org/dlr-sc pythonocc-core  && \
    pip install setuptools && \
    pip install enum34 && \
    pip install numpy-stl && \
    pip install coveralls && \
    pip install coverage
    
RUN cd $HOME  && \
    cd build && \
    git clone https://github.com/mathLab/PyGeM.git  && \ 
    cd PyGeM && \
    python setup.py install

USER PyGeM


