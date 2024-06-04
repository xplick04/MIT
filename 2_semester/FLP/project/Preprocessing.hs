-- Project:  FLP - project 1 
-- Author:   Maxim Plička (xplick04, 231813)
-- Date:     2024-03-23
module Preprocessing where

import DataTypes

import qualified Data.Map as Map
import Data.List (minimumBy, nub, partition)
import Data.Ord (comparing)


isLeft' :: Either a b -> Bool
isLeft' (Left _) = True
isLeft' (Right _) = False


-- Count start spaces to detetmine level of node
countStartSpaces :: String -> Int
countStartSpaces [] = 0
countStartSpaces (x:xs)
    | x == ' ' = 1 + countStartSpaces xs
    | otherwise = 0


makeTupleNode :: [String] -> Int -> Either String Tuple
makeTupleNode [_, idx, tresh] lvl = Right (Node (lvl `div` 2) (read idx) (read tresh))  -- lvl needs to be devides by 2
makeTupleNode _ _ = Left "Wrong node format."


makeTupleLeaf :: [String] -> Int -> Either String Tuple
makeTupleLeaf [_, classs] lvl = Right (Leaf (lvl `div` 2) classs) -- lvl needs to be devides by 2
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
        c' = if last c == '\n' then init c else c   -- delete end \n, otherwise lines create empty list as last elem
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
        --c' = if last c == '\n' then init c else c   -- delete end \n
        linesList = lines c
        strList = map (words . stripInput) linesList
        createDato strList' = (map read (init strList'), last strList')
    in map createDato strList


-- Splits dataset based of midPoint and first feature value
filterByMidPoint :: TreshHold -> String -> [Dato] -> [Dato]
filterByMidPoint midPoint "under" dataset =
    filter (\(features, _) -> case features of
                                (x:_) -> x <= midPoint
                                _     -> False) dataset
filterByMidPoint midPoint "over" dataset =
    filter (\(features, _) -> case features of
                                (x:_) -> x > midPoint
                                _     -> False) dataset
filterByMidPoint _ _ _ = []



-- Calculate impurity at a given midpoint for a given subset
getMidPointImpurity :: [Dato] -> Float
getMidPointImpurity subset =
    let labelCounts = foldl (\counts (_, label) -> Map.insertWith (+) label 1 counts) Map.empty subset -- get each label and their counts
        totalCount = fromIntegral ( sum (Map.elems labelCounts) :: Int)
        px = map (\count -> (fromIntegral count / totalCount) ** 2) (Map.elems labelCounts) -- compute px for each label
    in 1 - sum px



-- Calculate best midpoint for feature in the dataset
calculateFeatureBestMP :: [Dato] -> Int -> MidPoint
calculateFeatureBestMP dataset featureIdx =
    let midPoints = nub (init ((map (!! 0) (map fst dataset))))
        under mp = filterByMidPoint mp "under" dataset 
        over mp = filterByMidPoint mp "over" dataset
        lenUnder mp = fromIntegral (length (under mp))
        lenOver mp = fromIntegral (length (under mp))

        calculateImpurities [] = []
        calculateImpurities (mp:mps) =
            let impurityUnder = getMidPointImpurity (under mp)
                impurityOver = getMidPointImpurity (over mp)
                lenUnder' = lenUnder mp
                lenOver' = lenOver mp
                totalImpurity = impurityUnder * lenUnder' / (lenUnder' + lenOver') +
                     impurityOver * lenOver' / (lenUnder' + lenOver')

            in if totalImpurity == 0
                then [0]  -- No need to look for other
                else totalImpurity : calculateImpurities mps

        impurities = calculateImpurities midPoints

        (minImpurity, minIdx) = minimumBy (comparing fst) (zip impurities [0..])
        bestMidPoint = midPoints !! minIdx
    in (minImpurity, bestMidPoint, featureIdx)


-- Get each feature best mid points
getFeaturesBestMPs :: [Dato] -> [MidPoint]
getFeaturesBestMPs dataset = gfbMP dataset 0
    where 
        gfbMP _ idx
            | idx >= numFeatures = []
        gfbMP d idx =
            let featureValues = map (\dato -> extractFeatureAtIndex dato idx) d
                a@(gini, mp, newIdx) = calculateFeatureBestMP featureValues idx
            in if gini == 0
                then [(gini, mp, newIdx)]
                else a : gfbMP dataset (idx + 1)

        numFeatures = length (fst (head dataset))


-- Extract feature value at a specific index
extractFeatureAtIndex :: Dato -> Int -> Dato
extractFeatureAtIndex (features, label) idx = ([features !! idx], label) 


-- Get best midPoint out of one column
getBestTuple :: [MidPoint] -> MidPoint
getBestTuple x = gbt x (10.0, 0.0, 0) -- initial best impurity
    where
        gbt [] y = y
        gbt ( x1@(0.0, _, _) : _) _ = x1
        gbt ( first@(x1, _, _) : xs ) second@(y1, _, _) 
            | x1 < y1 = gbt xs first
            | otherwise = gbt xs second

-- Split dataset based on midPoint
splitDataset :: [Dato] -> (Index, TreshHold) -> ([Dato], [Dato])
splitDataset d (idx, mp) = partition (\x -> (((fst x) !! idx) <= mp)) d
