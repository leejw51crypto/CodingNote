MYENV="ml"
CONDA_SUBDIR=osx-arm64 conda create -n $MYENV python=3.9 -c conda-forge
conda env config vars set CONDA_SUBDIR=osx-arm64
conda activate "$MYENV" 
conda info --envs
conda list

echo "OK"
echo "conda activate  $MYENV"
echo "export TOKENIZERS_PARALLELISM=true"
echo "pip install -U --pre datasets transformers torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu"
