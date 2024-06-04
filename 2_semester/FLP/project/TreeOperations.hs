-- Project:  FLP - project 1 
-- Author:   Maxim Plička (xplick04, 231813)
-- Date:     2024-03-23

module TreeOperations where

import DataTypes
import Preprocessing

import Data.List (nub, maximumBy)
import Data.Ord (comparing)

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


-- Build tree for first task
buildTree1 :: [Tuple] -> BTree
buildTree1 ((Node lvl x y):xs) = BNode x y (buildTree1 (getElem (helper xs lvl) 1)) (buildTree1 (getElem (helper xs lvl) 2))
    where
        helper a b = listSplit (reverse a) (b + 1)
buildTree1 ((Leaf _ x):_) = BLeaf x
buildTree1 _ = EmptyBTree


-- Get dato class by decision tree
findTree :: [Float] -> BTree -> String
findTree _ EmptyBTree = "Could not find class for data"  -- Handle the case of an empty tree
findTree x (BNode i t left right)
    | i < length x = 
        if (x !! i) > t
            then findTree x right
            else findTree x left
    | otherwise = "Index is out of bound"  -- Handle out-of-bound index
findTree _ (BLeaf i) = i


-- TASK 2

-- Build tree for second task
buildTree2 :: [Dato] -> BTree
buildTree2 [] = EmptyBTree
buildTree2 [d] = (makeLeaf d)
buildTree2 d = (makeNode d)


makeLeaf :: Dato -> BTree
makeLeaf d = BLeaf (snd d)


makeNode :: [Dato] -> BTree
makeNode d =
    let uniqueLabels = nub (map snd d) 
        idx = third (getBestTuple (getFeaturesBestMPs d))
        mp = second (getBestTuple (getFeaturesBestMPs d))
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



-- TASK2 - Cost complexity post-pruning

-- Prune if tree complexity is bigger than alpha
pruneTree :: Int -> BTree -> BTree
pruneTree alpha tree = let prunedTree = pruneSubtrees alpha tree
                       in if complexity(tree) > alpha then prunedTree else tree


-- Calculate cost of tree
complexity :: BTree -> Int
complexity EmptyBTree = 0
complexity (BLeaf _) = 0
complexity (BNode _ _ left right) = 1 + (complexity left) + (complexity right)


-- Replace node by majority label leaf if complexity is bigger than alpha
pruneSubtrees :: Int -> BTree -> BTree
pruneSubtrees _ EmptyBTree = EmptyBTree
pruneSubtrees _ (BLeaf label) = BLeaf label
pruneSubtrees alpha curr@(BNode idx mp left right) =
  let prunedLeft = pruneSubtrees alpha left
      prunedRight = pruneSubtrees alpha right
  in if complexity (BNode idx mp left right) > alpha
        then (BNode idx mp prunedLeft prunedRight)
        else (BLeaf (majorityLabel curr))


-- Get majority label
majorityLabel :: BTree -> String
majorityLabel tree =
    let labels = getLabels tree
        labelCounts = map (\label -> (label, length (filter (== label) labels))) (nub labels)
        (best, _) = maximumBy (comparing snd) labelCounts
    in best

-- Get list of all tree labels
getLabels :: BTree -> [String]
getLabels EmptyBTree = []
getLabels (BLeaf label) = [label]
getLabels (BNode _ _ left right) = (getLabels left) ++ (getLabels right)
