-- Project:  FLP - project 1 
-- Author:   Maxim Plička (xplick04, 231813)
-- Date:     2024-03-23
module Preprocessing where

import DataTypes

import Data.List (minimumBy, nub, partition, sort)
import Data.Ord (comparing)


isLeft' :: Either a b -> Bool
isLeft' (Left _) = True
isLeft' (Right _) = False


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
makeTuple _ _ = Left "Unknown format in input file 1."

-- Get tuples from input tree
getTuples :: String -> Either String [Tuple]
getTuples c = 
    let
        c' = if last c == '\n' then init c else c   -- delete end \n
        l = lines c'
        spacesCount = map countStartSpaces l
        tuples = map (\(line, count) -> makeTuple line count) (zip (map words (map stripInput l)) spacesCount)
    in case filter isLeft' tuples of -- check if at least one error occourred
        (Left err : _) -> Left err
        _ -> Right (map (\(Right x) -> x) tuples)

-- converts float strings into floats
createDato1 :: [String] -> [Float]
createDato1 x = map read x

--TASK 2

stripInput:: String -> String
stripInput [] = []
stripInput (x:xs)
    | x `elem` ['\n', ',', ':'] = ' ' : stripInput xs
    | otherwise = x : stripInput xs 

-- Create Datos from input file
createDatos2 :: String -> [Dato]
createDatos2 c =
    let 
        c' = if last c == '\n' then init c else c   -- delete end \n
        linesList = lines c'
        strList = map (words . stripInput) linesList
        createDato strList' = (map read (init strList'), last strList')
    in map createDato strList

-- Calculates midpoints for one feature
calculateMidPoint :: [Float] -> [TreshHold]
calculateMidPoint [] = []
calculateMidPoint [_] = []
calculateMidPoint (x:y:ys) = ((x + y) / 2) : calculateMidPoint (y:ys) 


-- Splits dataset based of midPoint and first feature value
filterByMidPoint :: TreshHold -> String -> [Dato] -> [Dato]
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
    let uniqueLabels = nub (map snd subset)
        countForLabel label = length (filter (\(_, l) -> l == label) subset)
        totalCount = sum (map countForLabel uniqueLabels)
        px = map (\label -> (fromIntegral (countForLabel label) / fromIntegral totalCount) ** 2) uniqueLabels
    in 1 - sum px


-- Calculate best midpoint for a feature in the dataset
calculateFeatureBestMP :: [Dato] -> Int -> MidPoint
calculateFeatureBestMP dataset featureIdx =
    let sortedValues = nub (sort (map (!! 0) (map fst dataset))) --remove identical feature values
        midPoints = calculateMidPoint sortedValues
        under mp = filterByMidPoint mp "under" dataset 
        over mp = filterByMidPoint mp "over" dataset
        lenUnder mp = fromIntegral (length (under mp))
        lenOver mp = fromIntegral (length (under mp))
        impurities = map (\mp -> (getMidPointImpurity (under mp)) * (lenUnder mp) / ((lenUnder mp) + (lenOver mp)) + 
                                (getMidPointImpurity (over mp)) * (lenOver mp) / ((lenUnder mp) + (lenOver mp)) 
                        ) midPoints
        (minImpurity, minIdx) = minimumBy (comparing fst) (zip impurities [0..])
        bestMidPoint = midPoints !! minIdx
    in (minImpurity, bestMidPoint, featureIdx)



getFeaturesBestMPs :: [Dato] -> [MidPoint]
getFeaturesBestMPs dataset = gfbMP dataset 0
    where 
        gfbMP [] _ = []
        gfbMP _ idx 
            | idx >= numFeatures = []
        gfbMP d idx =
            let featureBestMP = calculateFeatureBestMP d idx
            in featureBestMP : gfbMP (map dropFirstFeature d) (idx + 1)

        numFeatures = length (fst (head dataset))


-- Drop first feature in dato
dropFirstFeature :: Dato -> Dato
dropFirstFeature ((_:rest), label) = (rest, label)
dropFirstFeature ([], label) = ([], label)


-- Get best midPoint out of one column
getBestTuple :: [MidPoint] -> MidPoint
getBestTuple x = gbt x (10.0, 0.0, 0) -- initial best impurity
    where
        gbt [] y = y
        gbt ( first@(x1, _, _) : xs ) second@(y1, _, _) 
            | x1 < y1 = gbt xs first
            | otherwise = gbt xs second

-- Split dataset based on midPoint
splitDataset :: [Dato] -> (Index, TreshHold) -> ([Dato], [Dato])
splitDataset d (idx, mp) = partition (\x -> (((fst x) !! idx) < mp)) d
