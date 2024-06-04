#!/bin/bash
INPUT_FILE=$1
STEPS=$2

#spocitani radku ve vstupnim souboru
if [ ! -f $INPUT_FILE ]; then
    echo "Soubor $INPUT_FILE neexistuje"
    exit 1
fi

num_lines=$(wc -l < $INPUT_FILE)

#pocet procesoru nastaven podle poctu cisel (lze i jinak)
proc=$(echo "(l($num_lines)/l(2))" | bc -l | xargs printf "%1.0f") 

#preklad zdrojoveho souboru
mpic++ --prefix /usr/local/share/OpenMPI -o life life.cpp

#spusteni programu
mpirun --prefix /usr/local/share/OpenMPI -np $proc life $INPUT_FILE $STEPS	

#uklid
rm -f life