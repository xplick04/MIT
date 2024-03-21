module Preprocessing where

import DataTypes

import qualified Data.Text as T (stripSuffix, pack, unpack)
import Data.List (minimumBy, nub, partition, sort)
import Data.Ord (comparing)


modifyInput :: String -> String
modifyInput [] = []
modifyInput [x] = [x]
modifyInput (x:y:xs)
    | x `elem` [',', ':'] = '-' : modifyInput xs --remove extra space after ',', ':'
    | otherwise = x : modifyInput (y:xs)

countSpaceSequences :: String -> Int -> [Int]
countSpaceSequences [] i = [i] --end of file
countSpaceSequences (x:xs) i
    | x == ' ' = countSpaceSequences xs (i + 1)
    | x == '\n' = i : countSpaceSequences xs 0
    | otherwise = countSpaceSequences xs (i)


--removes \n at the end of the input tree file if it is there
removeEndNewline :: String -> String
removeEndNewline str =
    case T.stripSuffix (T.pack "\n") (T.pack str) of
        Just trimmedStr -> T.unpack trimmedStr
        Nothing -> str


stripInput:: String -> String
stripInput [] = []
stripInput (x:xs)
    | x `elem` ['\n', '-', ','] = ' ' : stripInput xs
    | otherwise = x : stripInput xs 


makeTuples :: [String] -> [Int] -> Either String [NodeTuple]
makeTuples [] [] = Right []
makeTuples ("Node":y:z:zs) (lvl:lvls) =
    case makeTuples zs lvls of
        Left err -> Left err
        Right rest -> Right ((Node, y, z, (lvl `div` 2)) : rest)
makeTuples ("Leaf":y:ys) (lvl:lvls) =
    case makeTuples ys lvls of
        Left err -> Left err
        Right rest -> Right ((Leaf, y, "", (lvl `div` 2)) : rest)
makeTuples _ _ = Left "Invalid input file format."


--TASK 2

createDato :: [String] -> Dato
createDato strList =
    let features = map read (init strList)
        label = last strList
    in (features, label)


calculateMidPoint :: [Float] -> [Float]
calculateMidPoint [] = []
calculateMidPoint [_] = []
calculateMidPoint (x:y:ys) = ((x + y) / 2) : calculateMidPoint (y:ys) 



filterByMidPoint :: Float -> String -> [Dato] -> [Dato]
filterByMidPoint midPoint "under" dataset =
    filter (\(features, _) -> (features !! 0) <= midPoint) dataset
filterByMidPoint midPoint "over" dataset =
    filter (\(features, _) -> (features !! 0) > midPoint) dataset
filterByMidPoint _ _ _ = []


getFeatureBestMP :: [Float] -> [Float] -> Int -> MidPoint
getFeatureBestMP [] _ idx = (1.0, 0.0, idx)
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



dropFirstFeature :: Dato -> Dato
dropFirstFeature ((_:rest), label) = (rest, label)
dropFirstFeature (_, label) = ([], label)


getBestTuple :: [MidPoint] -> MidPoint
getBestTuple x = gbt x (-1.0, 0.0, 0)
    where
        gbt [] y = y
        gbt ( first@(x1, _, _) : xs ) second@(y1, _, _) 
            | x1 > y1 = gbt xs first
            | otherwise = gbt xs second


splitDataset :: [Dato] -> (Int, Float) -> ([Dato], [Dato])
splitDataset d (idx, mp) = partition (\x -> (((fst x) !! idx) < mp)) d

