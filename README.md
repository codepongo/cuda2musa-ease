# cuda2musa-ease
CUDAtoMUSA-Ease facilitates a seamless migration from CUDA to MUSA, highlighting MUSA's high compatibility and exceptional ease of use for developers.

## NVidia Environment
1. Setup on Host
   A. docker pull nvidia/cuda:12.3.1-runtime-ubuntu22.04
   B. docker run -it --name cuda2musa-smoothly --gpus all  -v $(pwd):/host -p 80:7860 -v models:/root/.cache/ sh-harbor.mthreads.com/cuda12.3.1-runtime-ubuntu22.04 bash
2. Setup on Container
   A.  apt install vim
   B. sh Anaconda3-2024.10-1-Linux-x86_64.sh -b
     a. /root/anaconda3/condabin/conda init
     b. source ~/.bashrc
     c. conda create --name py310 python=3.10
     d. pip install -r requirements.txt
3. python gradio_app.py
4. http://{IP}

