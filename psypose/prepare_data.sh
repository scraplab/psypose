cd models
mkdir model_weights
cd model_weights
gdown "https://drive.google.com/uc?id=1eyE-IIHpkswHhYnPXX3HByrZrSiXk00g"
gdown "https://drive.google.com/uc?id=1AkYZmHJ_LsyQYsML6k72A662-AdKwxsv"

cd ../../MEVA/scripts
mkdir -p data
cd data
gdown "https://drive.google.com/uc?id=14CjsrGqzZeQ_H76ZYCiWwX4e-VAyxt_Q"
unzip meva_data.zip
rm meva_data.zip
cd ..
mkdir -p $HOME/.torch/models/
mv data/meva_data/yolov3.weights $HOME/.torch/models/
mkdir -p results/meva/vae_rec_2/models
mkdir -p results/meva/train_meva_2/

rm results/meva/train_meva_2/model_best.pth.tar
rm results/meva/vae_rec_2/models/model_1000.p

cp data/meva_data/model_1000.p results/meva/vae_rec_2/models/
mv data/meva_data/model_best.pth.tar results/meva/train_meva_2/model_best.pth.tar
