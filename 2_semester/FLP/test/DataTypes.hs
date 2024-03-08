
module DataTypes where


data Arguments = Arguments {
    task :: Int,
    input1 :: String,
    input2 :: String 
} deriving (Show, Eq)


data NodeType = Node | Leaf deriving (Show, Eq)

type NodeTuple = (NodeType, String, String, Int)

--data BTree i t = EmptyBTree | BNode i t (BTree i t) (BTree i t) | BLeaf i deriving (Show, Eq)
data BTree = EmptyBTree | BNode Int Float BTree BTree | BLeaf String deriving (Show, Eq)