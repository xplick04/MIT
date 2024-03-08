module TreeOperations where

import DataTypes

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


makeTree :: [NodeTuple] -> BTree String String
makeTree [] = EmptyBTree
makeTree ((Node, x, y, lvl):xs) = BNode x y (makeTree (getElem (helper xs lvl) 1) ) (makeTree (getElem (helper xs lvl) 2))
makeTree ((Leaf, x, _, lvl):xs) = BLeaf x
makeTree _ = EmptyBTree


printTree :: BTree String String -> IO ()
printTree EmptyBTree = putStrLn ""
printTree (BNode x y l r) = do
    putStrLn $ "Node " ++ x ++ " " ++ y
    printTree l
    printTree r
printTree (BLeaf x) = putStrLn $ "Leaf " ++ x


