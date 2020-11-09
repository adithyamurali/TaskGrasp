echo "Downloading config files"
# Below line not working (file size too big?)
wget "https://drive.google.com/uc?export=download&id=1vJfkaCCLJmvT8i5OB-qx_pOojgx2ouPf" -O cfg.zip
unzip cfg.zip -d ./
rm cfg.zip

echo "Downloading checkpoint files"
# Below line not working (file size too big?)
wget "https://drive.google.com/uc?export=download&id=1fasm-8MV6zBjdnbAHLbU8_8TZOkeABkR" -O checkpoints.zip
unzip checkpoints.zip -d ./
rm checkpoints.zip

echo "Downloading Dataset"
wget "https://drive.google.com/uc?export=download&id=1aZ0k43fBIZZQPPPraV-z6itpqCHuDiUU" -O data.zip
unzip data.zip -d ./
rm data.zip