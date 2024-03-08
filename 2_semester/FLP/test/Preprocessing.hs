module Preprocessing where

import DataTypes


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


stripInput:: String -> String
stripInput [] = []
stripInput (x:xs)
    | x `elem` y = ' ' : stripInput xs
    | otherwise = x : stripInput xs 
    where y = ['\n', '-']


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