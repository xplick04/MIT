module Preprocessing where

import DataTypes
import qualified Data.Text as T (stripSuffix, pack, unpack)
import Data.List

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


countLabel :: [Dato] -> String -> Int
countLabel dataset label = length $ filter (\(_, b) -> b == label) dataset

-- 1 - (Class_count/count of less)
giniOfLess :: [Dato] -> (Float, Int)
giniOfLess dataset =
    let uniqueLabels = nub (map snd dataset)
    let lesser = map (\midPoint -> filterByMidPoint midPoint "under" dataset) midPoints
    let lesserCount = transpose (map (\label -> map (\elem -> countLabel elem label) lesser) uniqueLabels)
    let countOfLess = length dataset
        in 1 - 


-- 1 - (Class_count/count of less)
--giniOfGreater :: [Dato] -> [String] -> (Float, Int)
--giniOGreater dataset uniqueLabels



-- (totalOfLess/Total)*GiniOfLess + (totalOfGreater/Total)*giniOfGreater
calculateGini :: (Float, Int) -> (Float, Int) -> Float
calculateGini (a,b) (c,d) = (((b' / (b' + d')) * a) + ((d' / (b' + d')) * c))
    where
        b' = fromIntegral b
        d' = fromIntegral d

--split dataset based of midpoints in float array
--splitDataset :: [Dato] -> [Float] -> ([Dato], [Dato])


dropFirstNumber :: Dato -> Dato
dropFirstNumber ((_:rest), label) = (rest, label)


filterByMidPoint :: Float -> String ->[Dato] -> [Dato]
filterByMidPoint midPoint "under" dataset =
    filter (\(features, label) -> (features !! 0) < midPoint) dataset
filterByMidPoint midPoint "over" dataset =
    filter (\(features, label) -> (features !! 0) > midPoint) dataset
filterByMidPoint _ _ _ = []

