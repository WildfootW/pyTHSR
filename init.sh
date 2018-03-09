pushd dataset/denoise/ > /dev/null
unzip all.zip
mv noise_caps trainA
mv clean_caps trainB
popd
pip install -r requirements.txt
