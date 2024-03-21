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


countLabel :: [Dato] -> String -> Float
countLabel dataset label = fromIntegral (length $ filter (\(_, b) -> b == label) dataset)

filterByMidPoint :: Float -> String ->[Dato] -> [Dato]
filterByMidPoint midPoint "under" dataset =
    filter (\(features, label) -> (features !! 0) <= midPoint) dataset
filterByMidPoint midPoint "over" dataset =
    filter (\(features, label) -> (features !! 0) > midPoint) dataset
filterByMidPoint _ _ _ = []


getFeatureBestMP :: [Float] -> [Float] -> Int -> (Float, Float, Int)
getFeatureBestMP [] _ idx = (1,0,idx)
getFeatureBestMP (x:xs) (y:ys) idx = 
    let (bestImpurity, mp, index) = getFeatureBestMP xs ys idx
    in if x < bestImpurity
        then (x, y, idx)
        else (bestImpurity, mp, index)


calculateFeatureBestMP :: [Dato] -> Int -> (Float, Float, Int)
calculateFeatureBestMP dataset featureIdx = bestMidPoint
  where
    uniqueLabels = nub (map snd dataset)    
    sortedVector = (map sort (transpose (map fst dataset))) !! 0
    midPoints = calculateMidPoint sortedVector
    lesser = map (\midPoint -> filterByMidPoint midPoint "under" dataset) midPoints
    greater = map (\midPoint -> filterByMidPoint midPoint "over" dataset) midPoints
    lesserCount = transpose (map (\label -> map (\lst -> countLabel lst label) lesser) uniqueLabels)
    greaterCount = transpose (map (\label -> map (\grt -> countLabel grt label) greater) uniqueLabels)
    px = map (\sublist -> map (\x -> (x / (sum sublist)) ** 2) sublist) lesserCount
    py = map (\sublist -> map (\x -> (x / (sum sublist)) ** 2) sublist) greaterCount
    giniX = map sum px
    giniY = map sum py
    giniImpurityX = zipWith (\x g -> x - g) [1,1..] giniX
    giniImpurityY = zipWith (\x g -> x - g) [1,1..] giniY
    countInLesserSubset = map sum lesserCount
    countInGreaterSubset = map sum greaterCount
    total = zipWith (\x g -> x + g) countInGreaterSubset countInLesserSubset
    weightedImpurity = zipWith5 (\l g t x y -> (l/t) * x + (g/t) * y) countInLesserSubset countInGreaterSubset total giniImpurityX giniImpurityY
    bestMidPoint = getFeatureBestMP weightedImpurity midPoints featureIdx


getFeaturesBestMPs :: [Dato] -> Int -> [(Float, Float, Int)]
getFeaturesBestMPs dataset idx
    | idx > numFeatures = []    
    | True = (calculateFeatureBestMP dataset idx) : getFeaturesBestMPs (map dropFirstFeature dataset) (idx+1)
  where
    numFeatures = length (fst (head dataset))



dropFirstFeature :: Dato -> Dato
dropFirstFeature ((_:rest), label) = (rest, label)
dropFirstFeature (_, label) = ([], label)


getBestTuple :: [(Float, Float, Int)] -> (Float, Float, Int) -> (Float, Float, Int)
getBestTuple [] y = y
getBestTuple (x@(x1, _, _) : xs) y@(y1, _, _) 
    | x1 > y1 = getBestTuple xs x
    | otherwise = getBestTuple xs y


-- second arg is best mp(impurity, MP, index), TODO impurity not needed
splitDataset :: [Dato] -> (Int, Float) -> ([Dato], [Dato])
splitDataset d (idx, mp) = partition (\x -> (((fst x) !! idx) < mp)) d


--todo remake to either error if empty or tree
buildTree :: [Dato] -> BTree
buildTree [] = EmptyBTree
buildTree [d] = (makeLeaf d)
buildTree d = (makeNode d)


makeLeaf :: Dato -> BTree
makeLeaf d = BLeaf (snd d)


makeNode :: [Dato] -> BTree
makeNode d =
    let 
    impurity = first (getBestTuple (getFeaturesBestMPs d 0) (-1,0,0))
    uniqueLabels = nub (map snd d)  
    idx = third (getBestTuple (getFeaturesBestMPs d 0) (-1,0,0))
    mp = second (getBestTuple (getFeaturesBestMPs d 0) (-1,0,0))
    left = buildTree (fst (splitDataset d (idx, mp)))
    right = buildTree (snd (splitDataset d (idx, mp)))
    in if impurity == 0 && (length uniqueLabels == 1) then makeLeaf (d !! 0)
        else BNode idx mp left right


second :: (Float, Float, Int) -> Float
second (_,a,_) = a


third :: (Float, Float, Int) -> Int
third (_,_,a) = a

first :: (Float, Float, Int) -> Float
first (a,_,_) = a