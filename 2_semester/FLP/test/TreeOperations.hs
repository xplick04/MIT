module TreeOperations where

import DataTypes
import Preprocessing

import Data.List (nub)
import Debug.Trace


-- left and right tree gets its subtree nodes by splitting current node into two parts by their level
-- to do this i reversed current node list and took all nodes untill i reached level of current node children
-- second children gets rest of the current node list. After that childrens lists needs to be reversed to normal order
getTupleLvl :: Tuple -> Int
getTupleLvl (Leaf x _) = x
getTupleLvl (Node x _ _) = x


-- Return element from double (list was reversed, so it return opposite)
getElem :: ([Tuple], [Tuple]) -> Int -> [Tuple]
getElem (x,_) 2 = reverse x 
getElem (_,y) 1 = reverse y 
getElem _ _ = []


-- Take reversed list and produce double of reversed lists (first for left subtree,..)
listSplit :: [Tuple] -> Int -> ([Tuple], [Tuple])
listSplit [] _ = ([], [])
listSplit (x:xs) b
    | getTupleLvl x > b = (x : smaller, other)
    | getTupleLvl x == b = ([x], xs)
    | otherwise = (smaller, x : other)
    where
        (smaller, other) = listSplit xs b


helper :: [Tuple] -> Int -> ([Tuple], [Tuple])  
helper x y = listSplit (reverse x) (y + 1)


buildTree1 :: [Tuple] -> BTree
buildTree1 ((Node lvl x y):xs) = BNode x y (buildTree1 (getElem (helper xs lvl) 1)) (buildTree1 (getElem (helper xs lvl) 2))
buildTree1 ((Leaf _ x):_) = BLeaf x
buildTree1 _ = EmptyBTree


-- Get class by decision tree
findTree :: [Float] -> BTree -> String
findTree _ EmptyBTree = "Could not find class for data"  -- Handle the case of an empty tree
findTree x (BNode i t left right)
    | i < length x = 
        if (x !! i) > t
            then findTree x right
            else findTree x left
    | otherwise = "Index is out of bound"  -- Handle out-of-bound index
findTree _ (BLeaf i) = i


-- Call findTree for each element
printResult1 :: [[Float]] -> BTree -> IO ()
printResult1 [] _ = pure () -- do nothing
printResult1 (x:xs) y = do
    putStrLn (findTree x y)
    printResult1 xs y  


-- TASK 2
buildTree2 :: [Dato] -> BTree
buildTree2 [] = EmptyBTree
buildTree2 [d] = (makeLeaf d)
buildTree2 d = (makeNode d)


makeLeaf :: Dato -> BTree
makeLeaf d = BLeaf (snd d)


makeNode :: [Dato] -> BTree
makeNode d =
    let uniqueLabels = nub (map snd d) 
        idx = third (getBestTuple (getFeaturesBestMPs d 0))
        mp = second (getBestTuple (getFeaturesBestMPs d 0))
        left = buildTree2 (fst (splitDataset d (idx, mp)))
        right = buildTree2 (snd (splitDataset d (idx, mp)))
    in if (length uniqueLabels == 1) then makeLeaf (d !! 0)
        else BNode idx mp left right


first :: MidPoint -> Float
first (a,_,_) = a

second :: MidPoint -> Float
second (_,a,_) = a

third :: MidPoint -> Int
third (_,_,a) = a