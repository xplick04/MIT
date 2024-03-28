all: flp-fun

flp-fun: main.hs Preprocessing.hs TreeOperations.hs DataTypes.hs
	ghc -Wall -prof -fprof-auto -rtsopts -o flp-fun main.hs Preprocessing.hs TreeOperations.hs DataTypes.hs

clean:
	rm -f flp-fun *.o *.hi
