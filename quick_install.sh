#!/bin/bash
# file adapted from MICA https://github.com/Zielon/MICA/

# URL encoding function
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# Username and password input
echo -e "\nIf you do not have an account you can register at https://flame.is.tue.mpg.de/ following the installation instruction."
username=$(urle $1)
password=$(urle $2)

echo -e "\nDownloading FLAME..."
mkdir -p data/FLAME2020/
curl -L -X POST -d "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' -o './FLAME2020.zip'
powershell -Command "Expand-Archive -Path FLAME2020.zip -DestinationPath data/FLAME2020/"
rm -rf FLAME2020.zip

echo -e "\nDownload pretrained SPECTRE model..."
gdown --id 1vmWX6QmXGPnXTXWFgj67oHzOoOmxBh6B
mkdir -p pretrained/
