FROM phusion/baseimage:0.9.19

# Get Ubuntu updates
USER root
RUN apt-get update -q && \ 
    apt-get upgrade -y -o Dpkg::Options::="--force-confold" && \
    apt-get -y install sudo && \
    apt-get -y install locales && \
    echo "C.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set locale environment
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    LANGUAGE=C.UTF-8

# OpenBLAS threads should be 1 to ensure performance
RUN echo 1 > /etc/container_environment/OPENBLAS_NUM_THREADS && \
    echo 0 > /etc/container_environment/OPENBLAS_VERBOSE


# Set up user so that we do not run as root
RUN useradd -m -s /bin/bash -G sudo,docker_env PyGeM && \
    echo "PyGeM:docker" | chpasswd && \
    echo "PyGeM ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

RUN touch /etc/service/syslog-forwarder/down
COPY set-home-permissions.sh /etc/my_init.d/set-home-permissions.sh
RUN chmod +x /etc/my_init.d/set-home-permissions.sh

USER PyGeM
ENV HOME /home/PyGeM
RUN touch $HOME/.sudo_as_admin_successful && \
    mkdir $HOME/shared && \
    mkdir $HOME/build
VOLUME /home/PyGeM/shared

WORKDIR /home/PyGeM
ENTRYPOINT ["sudo","/sbin/my_init","--quiet","--","sudo","-u","PyGeM","/bin/bash","-l","-c"]
CMD ["/bin/bash","-i"]

# utilities and libraries
USER root
RUN apt-get update -y; apt-get install -y --force-yes --fix-missing --no-install-recommends curl git unzip tree subversion vim cmake bison g++ gfortran openmpi-bin pkg-config wget libpcre3-dev bison flex swig libglu1-mesa pyqt4-dev-tools    
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN id PyGeM
RUN chown -R PyGeM:PyGeM $HOME

RUN cd /tmp  && \
    wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh  && \
    chmod +x miniconda.sh && \
    bash miniconda.sh -b -p /usr/local/miniconda && \
    rm /tmp/*
ENV PATH=/usr/local/miniconda/bin:$PATH

RUN echo "PATH=/usr/local/miniconda/bin:$PATH" >> ~/.profile
RUN /bin/bash -c 'source  ~/.profile'

RUN  hash -r  && \
    conda config --set always_yes yes --set changeps1 no  && \
    conda update -q conda  
RUN  conda info -a  && \
    conda create --yes -n test python="2.7";

RUN /bin/bash -c 'source  activate test'  
# The default sip version has api that is not compatible with qt4.
RUN    conda install --yes numpy scipy matplotlib pip nose vtk sip=4.18 
RUN    conda install --yes -c https://conda.anaconda.org/dlr-sc pythonocc-core  &&\
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


