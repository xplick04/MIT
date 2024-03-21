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


makeTree :: [NodeTuple] -> BTree
makeTree ((Node, x, y, lvl):xs) = BNode (read x) (read y) (makeTree (getElem (helper xs lvl) 1)) (makeTree (getElem (helper xs lvl) 2))
makeTree ((Leaf, x, _, _):_) = BLeaf x
makeTree _ = EmptyBTree



printTree :: BTree -> IO ()
printTree EmptyBTree = putStrLn "ET"
printTree (BNode x y l r) = do
    putStrLn $ "Node " ++ (show x) ++ " " ++ (show y)
    printTree l
    printTree r
printTree (BLeaf x) = putStrLn $ "Leaf " ++ x


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