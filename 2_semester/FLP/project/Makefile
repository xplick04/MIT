all: flp-fun

flp-fun: main.hs Preprocessing.hs TreeOperations.hs DataTypes.hs
	ghc -Wall -o flp-fun main.hs Preprocessing.hs TreeOperations.hs DataTypes.hs

clean:
	rm -f flp-fun *.o *.hi
