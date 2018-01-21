pushd dataset/denoise/ > /dev/null
unzip noise_caps.zip
unzip clean_caps.zip
mv noise_caps trainA
mv clean_caps trainB
popd
