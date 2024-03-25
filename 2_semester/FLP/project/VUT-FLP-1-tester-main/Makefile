all: flp-fun

flp-fun: flp-fun.hs Preprocessing.hs TreeOperations.hs DataTypes.hs
	ghc -Wall -o flp-fun flp-fun.hs Preprocessing.hs TreeOperations.hs DataTypes.hs

clean:
	rm -f flp-fun *.o *.hi

