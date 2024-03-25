-- Project:  FLP - project 1 
-- Author:   Maxim Plička (xplick04, 231813)
-- Date:     2024-03-23

module DataTypes where

data Arguments = Arguments {
    task :: Int,
    input1 :: String,
    input2 :: String 
} deriving (Show, Eq)


-- TASK 1

type Level = Int
type TreshHold = Float
type Class = String
type Index = Int

data Tuple = Node Level Index TreshHold | Leaf Level Class deriving (Show)


-- TASK 2
data BTree = EmptyBTree | BNode Index TreshHold BTree BTree | BLeaf Class deriving (Eq)

instance Show BTree where
    show tree = showTree tree 0
        where
            showTree EmptyBTree s = replicate (s * 2) ' ' ++ "EmptyTree" 
            showTree (BNode x y l r) s = 
                let 
                    helper = replicate (s * 2) ' ' ++ "Node " ++ show x ++ " " ++ show y ++
                            showTree l (s + 1) ++
                            showTree r (s + 1)
                in
                    if s == 0 then -- first print (level == 0)
                        helper
                    else 
                        "\n" ++ helper
            showTree (BLeaf x) s = 
                let 
                    helper2 = replicate (s * 2) ' ' ++ "Leaf " ++ id x  -- id shows string without quote
                in
                    if s == 0 then  -- first print (level == 0)
                        helper2
                    else 
                        "\n" ++ helper2

type Impurity = Float
type Dato = ([Float], String)
type MidPoint = (Impurity, TreshHold, Index)