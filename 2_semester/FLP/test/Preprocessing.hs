module Preprocessing where

import DataTypes

import qualified Data.Text as T (stripSuffix, pack, unpack)
import Data.List (minimumBy, nub, partition, sort)
import Data.Ord (comparing)
import Data.Either (isLeft, isRight)


countStartSpaces :: String -> Int
countStartSpaces [] = 0
countStartSpaces (x:xs)
    | x == ' ' = 1 + countStartSpaces xs
    | otherwise = 0

makeTupleNode :: [String] -> Int -> Either String Tuple
makeTupleNode [_, idx, tresh] lvl = Right (Node (lvl `div` 2) (read idx) (read tresh))
makeTupleNode _ _ = Left "Wrong node format."


makeTupleLeaf :: [String] -> Int -> Either String Tuple
makeTupleLeaf [_, classs] lvl = Right (Leaf (lvl `div` 2) classs)
makeTupleLeaf _ _ = Left "Wrong leaf format."


makeTuple :: [String] -> Int -> Either String Tuple
makeTuple s@(x:_) lvl
    | x == "Node" = makeTupleNode s lvl
    | x == "Leaf" = makeTupleLeaf s lvl
    | otherwise = Left "Unknown format in input file 1."

--removes \n at the end of the input tree file if it is there
removeEndNewline :: String -> String
removeEndNewline str =
    case T.stripSuffix (T.pack "\n") (T.pack str) of
        Just trimmedStr -> T.unpack trimmedStr
        Nothing -> str


getTuples :: String -> Either String [Tuple]
getTuples c = 
    let
        l = lines (removeEndNewline c)
        spacesCount = map countStartSpaces l
        tuples = map (\(line, count) -> makeTuple line count) (zip (map words (map stripInput l)) spacesCount)
    in case filter isLeft tuples of
        (Left err : _) -> Left err
        _ -> Right (map (\(Right x) -> x) tuples)

createDato1 :: [String] -> [Float]
createDato1 x = map read x

--TASK 2

stripInput:: String -> String
stripInput [] = []
stripInput (x:xs)
    | x `elem` ['\n', '-', ',', ':'] = ' ' : stripInput xs
    | otherwise = x : stripInput xs 


-- Creates one dato
createDato2 :: [String] -> Dato
createDato2 strList =
    let features = map read (init strList)
        label = last strList
    in (features, label)

-- Calculates midpoints for one feature
calculateMidPoint :: [Float] -> [Float]
calculateMidPoint [] = []
calculateMidPoint [_] = []
calculateMidPoint (x:y:ys) = ((x + y) / 2) : calculateMidPoint (y:ys) 


-- Splits dataset based of midPoint and first feature value
filterByMidPoint :: Float -> String -> [Dato] -> [Dato]
filterByMidPoint midPoint "under" dataset =
    filter (\(features, _) -> (features !! 0) <= midPoint) dataset
filterByMidPoint midPoint "over" dataset =
    filter (\(features, _) -> (features !! 0) > midPoint) dataset
filterByMidPoint _ _ _ = []


getFeatureBestMP :: [Impurity] -> [TreshHold] -> Int -> MidPoint
getFeatureBestMP [] _ idx = (1.0, 0.0, idx) -- default value
getFeatureBestMP _ [] idx = (1.0, 0.0, idx) -- default value
getFeatureBestMP (x:xs) (y:ys) idx = 
    let (bestImpurity, mp, index) = getFeatureBestMP xs ys idx
    in if x < bestImpurity then (x, y, idx)
        else (bestImpurity, mp, index)


-- Calculate impurity at a given midpoint for a given subset
getMidPointImpurity :: [Dato] -> Float
getMidPointImpurity subset =
    let labels = nub (map snd subset)
        countForLabel label = length (filter (\(_, l) -> l == label) subset)
        totalCount = sum (map countForLabel labels)
        px = map (\label -> (fromIntegral (countForLabel label) / fromIntegral totalCount) ** 2) labels
    in 1 - sum px


-- Calculate best midpoint for a feature in the dataset
calculateFeatureBestMP :: [Dato] -> Int -> MidPoint
calculateFeatureBestMP dataset featureIdx =
    let sortedValues = sort (map (!! featureIdx) (map fst dataset))
        midPoints = calculateMidPoint sortedValues
        impurities = map (\midPoint -> getMidPointImpurity (filterByMidPoint midPoint "under" dataset) 
                                      + getMidPointImpurity (filterByMidPoint midPoint "over" dataset)) midPoints
        (minImpurity, minIdx) = minimumBy (comparing fst) (zip impurities [0..])
        bestMidPoint = midPoints !! minIdx
    in (minImpurity, bestMidPoint, featureIdx)



-- Get the best midpoints for all features in the dataset
getFeaturesBestMPs :: [Dato] -> Int -> [MidPoint]
getFeaturesBestMPs dataset idx
    | idx >= numFeatures = []
    | otherwise =
        let featureBestMP = calculateFeatureBestMP dataset idx
        in featureBestMP : getFeaturesBestMPs (map (dropFirstFeature) dataset) (idx + 1)
    where
        numFeatures = length (fst (head dataset))


-- Drop first feature in dato
dropFirstFeature :: Dato -> Dato
dropFirstFeature ((_:rest), label) = (rest, label)
dropFirstFeature (_, label) = ([], label)


-- Get best midPoint out of one column
getBestTuple :: [MidPoint] -> MidPoint
getBestTuple x = gbt x (-1.0, 0.0, 0)
    where
        gbt [] y = y
        gbt ( first@(x1, _, _) : xs ) second@(y1, _, _) 
            | x1 > y1 = gbt xs first
            | otherwise = gbt xs second

-- Split dataset based on midPoint
splitDataset :: [Dato] -> (Int, Float) -> ([Dato], [Dato])
splitDataset d (idx, mp) = partition (\x -> (((fst x) !! idx) < mp)) d

