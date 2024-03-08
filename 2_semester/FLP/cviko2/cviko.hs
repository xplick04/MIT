--zipWith (\x y -> (x,y)) [1..10] [1..10] --zip s pomoci zipWith


filter1 :: (a -> Bool) -> [a] -> [a]
filter1 f = foldr (\x acc -> if f x then x:acc else acc) []

data Tree a = EmptyTree | Node a (Tree a) (Tree a)

treeInsert :: (Ord a) => a -> Tree a -> Tree a
treeInsert a EmptyTree = Node a EmptyTree EmptyTree
treeInsert a (Node b l r)
    | a > b = Node b (treeInsert a l) r
    | a < b = Node b l (treeInsert a r)
    | True = Left "Erroro"

treeElem :: (Ord a) => a -> Tree a -> Bool
treeElem a (Tree a Tree l Tree r)
    | a > b = treeElem a l
    | a < b = treeElem a r
    | True = True


data KTree k v
    = EmptyTree
    | KNode k v (KTree k v) (KTree k v)
    deriving (Show)

KtreeInsert :: (Ord k) => k -> KTree k v -> KTree k v
treeInsert a EmptyTree = Node a EmptyTree EmptyTree
ktreeInsert k EmptyTree _ = (KNode k0 v0 EmptyTree EmptyTree)
KtreeInsert k (KNode k0 v0 Ktree k1 k1 Ktree k2 v2)
    | k < k0 =  KtreeInsert (k1 k1)
    | k > k0 =  KtreeInsert (k2 k2)
    | True = Left "."

KtreeElem :: (Ord k) => k -> KTree k v -> Mabye v
ktreeElem k EmptyTree _ = Nothing
KtreeElem k (KNode k0 v0 Ktree k1 k1 Ktree k2 v2)
    | k < k0 =  KtreeElem (k1 k1)
    | k > k0 =  KtreeElem (k2 k2)
    | k == k0 = Just 



instance (Show a) => Show (Tree a) where
    Show t = "--" ++ helper t ++ "--"
    where 
        helper EmptyTree = ""
        helper (Node a Tree l Tree r) = show x ++ helper l ++ helper r


main :: IO ()
main = do print()