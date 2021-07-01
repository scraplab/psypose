cd models
mkdir model_weights
cd model_weights
gdown "https://drive.google.com/uc?id=1eyE-IIHpkswHhYnPXX3HByrZrSiXk00g"
gdown "https://drive.google.com/uc?id=1AkYZmHJ_LsyQYsML6k72A662-AdKwxsv"

cd ../../MEVA
mkdir -p data
cd data
gdown "https://drive.google.com/uc?export=download&id=1l5pUrV5ReapGd9uaBXrGsJ9eMQcOEqmD"
unzip meva_data.zip
rm meva_data.zip
cd ..
mkdir -p $HOME/.torch/models/
mkdir -p results/meva/vae_rec_2/models
mkdir -p results/meva/train_meva_2/

cp data/meva_data/model_1000.p results/meva/vae_rec_2/models/
mv data/meva_data/spin_model_checkpoint.pth.tar results/meva/train_meva_2/spin_model_checkpoint.pth.tar

