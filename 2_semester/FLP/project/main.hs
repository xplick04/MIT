-- Project:  FLP - project 1 
-- Author:   Maxim Plička (xplick04, 231813)
-- Date:     2024-03-23

import DataTypes
import Preprocessing
import TreeOperations

import System.IO
import System.Environment (getArgs)

parseArgs :: [String] -> Arguments -> Int -> Arguments -- user input list, parsed arguments, argument number -> parsed arguments
parseArgs [] args _ = args  -- end case
parseArgs ("-1":xs) (Arguments _ _ i2) 0 = parseArgs xs (Arguments 1 "" i2) 1   -- first argument
parseArgs ("-2":xs) (Arguments _ _ i2) 0 = parseArgs xs (Arguments 2 "" i2) 1   -- first argument
parseArgs (x:xs) (Arguments t _ i2) 1 = parseArgs xs (Arguments t x i2) 2  -- second argument
parseArgs (x:xs) (Arguments 1 i1 _) 2 = parseArgs xs (Arguments 1 i1 x) 3  -- third argument (only for task 1)
parseArgs _ _ _ = Arguments 0 "" "" -- error



main :: IO ()
main = do
    args <- getArgs
    let parsedArgs = parseArgs args (Arguments 0 "" "") 0
    
    case parsedArgs of
        Arguments 1 i1 i2 -> do
            -- Input file 1
            f1 <- openFile i1 ReadMode
            c1 <- hGetContents f1
            -- Input file 2
            f2 <- openFile i2 ReadMode
            c2 <- hGetContents f2

            let result = getTuples c1
            case result of  -- check if input file format matches
                Right tuples -> do
                    let tree = buildTree1 tuples
                    let dataset = map createDato1 (map words (map stripInput (lines c2)))
                    mapM_ (\d -> putStrLn (findTree d tree)) dataset -- print result
                Left err -> print err

            hClose f1
            hClose f2

        Arguments 2 i1 _ -> do
            f1 <- openFile i1 ReadMode
            c1 <- hGetContents f1

            let dataset = createDatos2 c1

            let tree = buildTree2 dataset
            
            let alpha = 0 --pruning alpha value (0 does nothing, the larger value the more pruning)
            let pruned = pruneTree alpha tree
            print pruned

            hClose f1

        _ -> putStrLn "Invalid input"