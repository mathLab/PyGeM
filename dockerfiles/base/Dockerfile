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
