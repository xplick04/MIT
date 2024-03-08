
module DataTypes where


data Arguments = Arguments {
    task :: Integer,
    input1 :: String,
    input2 :: String 
} deriving (Show, Eq)

data Tree k v = EmptyTree | Node k v (Tree k v) (Tree k v)
