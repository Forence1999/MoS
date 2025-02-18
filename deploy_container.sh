#!/bin/bash
set -e


# config container
apt-get update && apt install -y curl tmux git vim htop systemd wget unzip # when systemd is installed, select 6 and 31 sequentially.
touch ~/.tmux.conf && echo -e "set -g mouse on\n" >>~/.tmux.conf
mkdir -p ~/.config/htop && touch ~/.config/htop/htoprc && echo -e "# Beware! This file is rewritten by htop when settings are changed in the interface. \n# The parser is also very primitive, and not human-friendly. \nfields=0 48 17 18 38 39 40 2 46 47 49 1 \nsort_key=46 \nsort_direction=1 \nhide_threads=1 \nhide_kernel_threads=1 \nhide_userland_threads=1 \nshadow_other_users=0 \nshow_thread_names=0 \nshow_program_path=0 \nhighlight_base_name=1 \nhighlight_megabytes=0 \nhighlight_threads=1 \ntree_view=1 \nheader_margin=1 \ndetailed_cpu_time=0 \ncpu_count_from_zero=0 \nupdate_process_names=0 \naccount_guest_in_cpu_meter=0 \ncolor_scheme=0 \ndelay=15 \nleft_meters=CPU Memory Swap \nleft_meter_modes=1 1 1 \nright_meters=Tasks LoadAverage Uptime \nright_meter_modes=2 2 2 \n" >>~/.config/htop/htoprc

# install miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

echo "Please run \"source ~/.bashrc\" and close this terminal before continuing this script!"
exit

# source ~/.bashrc # should close the terminal and reopen it
conda config --add channels conda-forge
python --version # check python version, which should be Python 3.11.5

# special for open-instruct
apt update && apt install -y openjdk-8-jre-headless
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get -y install git-lfs
pip install --upgrade pip setuptools wheel
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install packaging
pip install flash-attn==2.2.2 --no-build-isolation
pip install pyarrow==13.0.0
pip install -r requirements.txt
chmod +x /workspace/scripts/*
chmod -R 777 /workspace/
python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token({HF_TOKEN})"

echo "Done!"
