module TreeOperations where

import DataTypes
import Preprocessing

import Data.List (nub)

getNodeLvl:: NodeTuple -> Int
getNodeLvl (_,_,_,x) = x


--takes reversed list and produces double of reversed lists (first for left subtree,..)
splitList2 :: [NodeTuple] -> Int -> ([NodeTuple], [NodeTuple])
splitList2 [] _ = ([], [])
splitList2 (x:xs) b
    | getNodeLvl x > b = (x : smaller, other)
    | getNodeLvl x == b = ([x], xs)
    | otherwise = (smaller, x : other)
    where
        (smaller, other) = splitList2 xs b


--return element from double (list was reversed, so it return opposite)
getElem :: ([NodeTuple], [NodeTuple]) -> Int -> [NodeTuple]
getElem (x,_) 2 = reverse x 
getElem (_,y) 1 = reverse y 
getElem _ _ = []


helper :: [NodeTuple] -> Int -> ([NodeTuple], [NodeTuple])  
helper x y = splitList2 (reverse x) (y + 1)


makeTree :: [NodeTuple] -> BTree
makeTree ((Node, x, y, lvl):xs) = BNode (read x) (read y) (makeTree (getElem (helper xs lvl) 1)) (makeTree (getElem (helper xs lvl) 2))
makeTree ((Leaf, x, _, _):_) = BLeaf x
makeTree _ = EmptyBTree


--function calls findTree for each element
printResult1 :: [[Float]] -> BTree -> IO ()
printResult1 [] _ = pure ()
printResult1 (x:xs) y = do
    putStrLn (findTree x y)
    printResult1 xs y  


findTree :: [Float] -> BTree -> String
findTree _ EmptyBTree = "Could not find class for data"  -- Handle the case of an empty tree
findTree x (BNode i0 t0 left right)
    | i0 < length x = 
        if x !! i0 > t0
            then findTree x right
            else findTree x left
    | otherwise = "Index is out of bound"  -- Handle out-of-bound index
findTree _ (BLeaf i) = show i  -- Convert the float value to a string


-- TASK 2

buildTree :: [Dato] -> BTree
buildTree [] = EmptyBTree
buildTree [d] = BLeaf (snd d)
buildTree d = (makeNode d)


makeLeaf :: Dato -> BTree
makeLeaf d = BLeaf (snd d)


makeNode :: [Dato] -> BTree
makeNode d =
    let uniqueLabels = nub (map snd d)  
        idx = third (getBestTuple (getFeaturesBestMPs d 0))
        mp = second (getBestTuple (getFeaturesBestMPs d 0))
        left = buildTree (fst (splitDataset d (idx, mp)))
        right = buildTree (snd (splitDataset d (idx, mp)))
    in if (length uniqueLabels == 1) then makeLeaf (d !! 0)
        else BNode idx mp left right


first :: MidPoint -> Float
first (a,_,_) = a

second :: MidPoint -> Float
second (_,a,_) = a

third :: MidPoint -> Int
third (_,_,a) = a