# Remove existing CuDA versions
sudo apt --purge remove "cublas*" "cuda*"
sudo apt --purge remove "nvidia*"
rm -rf /usr/local/cuda*
sudo apt-get autoremove && sudo apt-get autoclean

# Reboot to remove cached files
reboot

# Install build essentials
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
# Download CuDA from following path
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
# read as a DebianPackage
sudo apt-key adv - fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130–1_amd64.deb
sudo apt-key adv - fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
# install cuda-10.1
sudo apt install cuda-10.1


# copy the following files into the cuda toolkit directory.
echo 'export PATH=/usr/local/cuda-10.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
reboot


