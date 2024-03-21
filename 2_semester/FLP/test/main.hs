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
        Arguments 1 input1 input2 -> do
            f1 <- openFile input1 ReadMode
            c1 <- hGetContents f1
            f2 <- openFile input2 ReadMode
            c2 <- hGetContents f2

            let result = getTuples c1
            case result of
                Right tuples -> do
                    let tree = buildTree1 tuples
                    let dataset = map createDato1 (map words (map stripInput (lines c2)))
                    printResult1 dataset tree
                Left err -> print err
      
            hClose f1
            hClose f2

        Arguments 2 input1 _ -> do
            f1 <- openFile input1 ReadMode
            c1 <- hGetContents f1

            let dataset = map createDato2 (map words (map stripInput (lines c1)))
            let tree = buildTree2 dataset

            print tree
            hClose f1

        _ -> putStrLn "Invalid input"