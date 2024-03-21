
module DataTypes where

data Arguments = Arguments {
    task :: Int,
    input1 :: String,
    input2 :: String 
} deriving (Show, Eq)


-- TASK 1
data NodeType = Node | Leaf | EndFileChar deriving (Show, Eq)

type NodeTuple = (NodeType, String, String, Int)


-- TASK 2
data BTree = EmptyBTree | BNode Int Float BTree BTree | BLeaf String deriving (Eq)


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
                    if s == 0 then 
                        helper
                    else 
                        "\n" ++ helper
            showTree (BLeaf x) s = "\n" ++ replicate (s * 2) ' ' ++ "Leaf " ++ show x


type Dato = ([Float], String)

type MidPoint = (Float, Float, Int)