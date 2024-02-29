{-
test :: Int -> Int -> Int
test x y = x + y

main :: IO ()
main = do print(test 1 2)
-}

elem2 :: (Eq a) => a -> [a] -> Bool
elem2 _ [] = False
elem2 x (y:ys) = x == y || elem2 x ys


nub2 :: (Eq a) => [a] -> [a]
nub2 [] = []
nub2 (x:xs)
    | elem2 x xs = nub2 xs
    | True = x : nub2 xs


isAsc2 :: [Int] -> Bool
isAsc2 [] = True
isAsc2 [_] = True
isAsc2 (x1:x2:xs)
    | x1 <= x2 = isAsc2 (x2:xs)
    | True = False


hasPath2 :: [(Int, Int)] -> Int -> Int -> Bool
hasPath2 [] x y = x == y
hasPath2 xs x y
    | x == y = True
    | True =
        let xs' = [ (n,m) | (n,m) <- xs, n /= x ] in 
        or [ hasPath2 xs' m y | (n,m) <- xs, n == x]


main :: IO ()
main = do print(hasPath2 [(1,1), (1,2), (2,3)] 1 3)


