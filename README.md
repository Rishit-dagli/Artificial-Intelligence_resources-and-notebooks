# Artificial-Intelligence_resources-and-notebooks
This repo contains various different datasets and codes with various different algorithms. It also contains code and demonstrations to run an Artificial Intelligence Algorithm on the edge. It also contains many datasets where one can practice AI. To be specific more focus here is on Computer Vision and there are mostly car photo datasets. I have also included some notebooks giving info about running applications on edge, using Intel Open Vino Toolkit. You can download the Open Vino toolkit [here](https://software.intel.com/en-us/openvino-toolkit/choose-download).
<h3>Install Open-Vino for Linux from Docker Image</h3>

Target Operating Systems

- Ubuntu* 16.04 long-term support (LTS), 64-bit
- CentOS* 7.4, 64-bit

Host Operating Systems

- Linux with installed GPU driver and with Linux kernel supported by GPU driver

<h4>For CPU</h4>

To build a Docker image, create a Dockerfile that contains defined variables and commands required to create an OpenVINO toolkit installation image.

Create your Dockerfile using the following example as a template: 

    FROM ubuntu:16.04
    ENV http_proxy $HTTP_PROXY
    ENV https_proxy $HTTPS_PROXY
    ARG DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/13231/l_openvino_toolkit_p_2019.0.000.tgz
    ARG INSTALL_DIR=/opt/intel/openvino
    ARG TEMP_DIR=/tmp/openvino_installer
    RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        cpio \
        sudo \
        lsb-release && \
        rm -rf /var/lib/apt/lists/*
    RUN mkdir -p $TEMP_DIR && cd $TEMP_DIR && \
        wget -c $DOWNLOAD_LINK && \
        tar xf l_openvino_toolkit*.tgz && \
        cd l_openvino_toolkit* && \
        sed -i 's/decline/accept/g' silent.cfg && \
        ./install.sh -s silent.cfg && \
        rm -rf $TEMP_DIR
    RUN $INSTALL_DIR/install_dependencies/install_openvino_dependencies.sh
    # build Inference Engine samples
    RUN mkdir $INSTALL_DIR/deployment_tools/inference_engine/samples/build && cd $INSTALL_DIR/deployment_tools/inference_engine/samples/build && \
        /bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh && cmake .. && make -j1"
        
To build a Docker* image for CPU, run the following command: 

    docker build . -t <image_name> \
    --build-arg HTTP_PROXY=<http://your_proxy_server.com:port> \
    --build-arg HTTPS_PROXY=<https://your_proxy_server.com:port>

To install the OpenVINO toolkit from the prepared Docker image, run the image with the following command: 

    docker run -it <image_name>
    
<h4>For GPU</h4>

Before building a Docker* image on GPU, add the following commands to the Dockerfile example for CPU above:

    COPY intel-opencl*.deb /opt/gfx/
    RUN cd /opt/gfx && \
       dpkg -i intel-opencl*.deb && \
       ldconfig && \
       rm -rf /opt/gfx
    RUN useradd -G video -ms /bin/bash user
    USER user

To build a Docker image for GPU:

Copy Intel® OpenCL™ driver for Ubuntu `(intel-opencl*.deb)` from `<OPENVINO_INSTALL_DIR>/install_dependencies` to the folder with the Dockerfile.

Run the following command to build a Docker image:

    docker build . -t <image_name> \
    --build-arg HTTP_PROXY=<http://your_proxy_server.com:port> \
    --build-arg HTTPS_PROXY=<https://your_proxy_server.com:port>

Run the Docker Image for GPU

To make GPU available in the container, attach the GPU to the container using `--device /dev/dri` option and run the container:

    docker run -it –device /dev/dri <image_name>
<br>

<h3>YUM</h3>

Note: You must be logged in as root to set up and install the repository. 
 Import the .repo file using the yum-config-manager:

`yum-utils` must be installed on your system. If it’s not currently installed, run the command:

    sudo yum install yum-utils
    
Add repository using the yum-config-manager:

    sudo yum-config-manager --add-repo https://yum.repos.intel.com/openvino/2019/setup/intel-openvino-2019.repo

Import the gpg public key for the repository:

    sudo rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-P

Run the following command to verify installation:

    yum repolist | grep -i openvino

Results:

    intel-openvino-2019 Intel(R) Distribution of OpenVINO 2019
    
To install the full runtime version of the OpenVINO package:

    sudo yum install intel-openvino-runtime-centos7    
    
<h4>APT</h4>


Download the public GPG key from https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB and save it to a file.
    
Add this key to the system keyring:

    sudo apt-key add <PATH_TO_DOWNLOADED_GPG_KEY>
    
Check the list of APT keys running the following command:
    
    sudo apt-key list

Add the APT Repository

Navigate to the repositories directory:
    
    cd /etc/apt/sources.list.d
    
Create a new source list file:

    sudo vi intel-openvino-2019.list

Add the following code:

    deb https://apt.repos.intel.com/openvino/2019/ all main
    
Save and close the file `intel-openvino-2019.list`.

To install a specific version of an OpenVINO package:

    sudo apt install intel-openvino-<PACKAGE_TYPE>-ubuntu<OS_VERSION>-<VERSION>.<UPDATE>.<BUIL

For additional installation guide refer [here](https://software.intel.com/en-us/openvino-toolkit/choose-download)
For some basic tutorials refer [here](https://software.intel.com/en-us/openvino-toolkit/documentation/get-started)
For documentation [here](https://software.intel.com/en-us/openvino-toolkit/documentation/featured)

<h1>Environment</h1>
The repository also contains an `environment.yml` file which also contains all the specific versions and libraries used for all the notebooks. There are also instructions for the same in a word document. >Notebooks > Installation Instructions.docx
<br>
<h3>To create the environment (Anaconda)</h3>
<br>
  Use the terminal or an Anaconda Prompt for the following steps:<br><br>
  Create the environment from the environment.yml file:<br>
    

    conda env create -f environment.yml


<br>
    The first line of the yml file sets the new environment's name. For details see Creating an environment file manually,    https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually .
    <br><br>
    Activate the new environment:
    <br>
    
    conda activate myenv
    
<br>
    Verify that the new environment was installed correctly:<br>
    
    conda env list
    
<br>
    You can also use 
    
    conda info --envs
    
