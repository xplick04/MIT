--zipWith (\x y -> (x,y)) [1..10] [1..10] --zip s pomoci zipWith

{-
filter1 :: (a -> Bool) -> [a] -> [a]
filter1 f = foldr (\x acc -> if f x then x:acc else acc) []

data Tree a = EmptyTree | Node a (Tree a) (Tree a)
-}
{-
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
-}

data Tree k v = EmptyTree | Node k v (Tree k v) (Tree k v) deriving (Show)

kTreeAdd :: (Ord k) => (k, v) -> Tree k v -> Either String (Tree k v) 
kTreeAdd (k, v) EmptyTree = Right (Node k v EmptyTree EmptyTree)
kTreeAdd x@(k, v) (Node kN vN left right)
    | k < kN = case kTreeAdd x left of
                 Left err -> Left err
                 Right newLeft -> Right (Node kN vN newLeft right)
    | k > kN = case kTreeAdd x right of
                 Left err -> Left err
                 Right newRight -> Right (Node kN vN left newRight)
    | otherwise = Left "TralalTrololo"

removeRight (Right a) = a


--kTreeAdd (key, value) 


{-
KtreeElem :: (Ord k) => k -> KTree k v -> Mabye v
ktreeElem k EmptyTree _ = Nothing
KtreeElem k (KNode k0 v0 Ktree k1 k1 Ktree k2 v2)
    | k < k0 =  KtreeElem (k1 k1)
    | k > k0 =  KtreeElem (k2 k2)
    | k == k0 = Just 
-}

{-
instance (Show a) => Show (Tree a) where
    Show t = "--" ++ helper t ++ "--"
    where 
        helper EmptyTree = ""
        helper (Node a Tree l Tree r) = show x ++ helper l ++ helper r
-}


instance Functor (Tree k) where
    fmap _ EmptyKtree = EmptyKtree
    fmap f (Knode k v left right) = Knode k (f v) (fmap f left) (fmap f right)


main :: IO ()
main = do 
    print("nic")
    
