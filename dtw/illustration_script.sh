#!/bin/bash

AT1='austria/33UVP/2017'
DK1='denmark/32VNH/2017'
FR1='france/31TCJ/2017'
FR2='france/30TXT/2017'


crops=("horsebeans" "spring_barley" "winter_triticale" "winter_barley")

MODEL_NAME="pseltae_DK1"
SOURCE="$DK1"
TARGET=$FR2

for crop in "${crops[@]}"; do
    python illustrate_pe.py -e $MODEL_NAME\_pe_recurrent --source $SOURCE --target $TARGET --pos_type rnn --crop_type $crop
done

MODEL_NAME="pseltae_AT1"
SOURCE="$AT1"
TARGET=$FR2

for crop in "${crops[@]}"; do
    python illustrate_pe.py -e $MODEL_NAME\_pe_recurrent --source $SOURCE --target $TARGET --pos_type rnn --crop_type $crop
done

MODEL_NAME="pseltae_FR1"
SOURCE="$FR1"
TARGET=$FR2

for crop in "${crops[@]}"; do
    python illustrate_pe.py -e $MODEL_NAME\_pe_recurrent --source $SOURCE --target $TARGET --pos_type rnn --crop_type $crop
done