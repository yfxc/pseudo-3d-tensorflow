#!/bin/bash


for folder in $1/*
do
    for file in "$folder"/*.avi
	
    do
		echo "${file%.avi}"
        if [[ ! -d "${file%.avi*}" ]]; then
            mkdir -p "${file%.avi*}"
        fi
        ffmpeg -i "$file" -vf fps=5 "${file%.avi*}"/%05d.jpg
        rm "$file"
    done
done
