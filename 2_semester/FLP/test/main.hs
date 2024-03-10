import DataTypes
import Preprocessing
import TreeOperations

import System.IO
import System.Environment
import Data.List


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
            f2 <- openFile input2 ReadMode
            c1 <- hGetContents f1
            c2 <- hGetContents f2
            --input1 processing
            let inputM = removeEndNewline (modifyInput c1)
            let nodeLevels = countSpaceSequences inputM 0
            let s = stripInput inputM
                list = words s
            case makeTuples list nodeLevels of
                Left err -> putStrLn err
                Right tuples -> do
                    let tree = makeTree tuples
                    --input2 processing
                    let input2M = map (\x -> if x == ',' then ' ' else x) c2
                    let a = lines input2M
                    let b = map words a
                    let c = map (map read) b
                    printResult1 c tree
            hClose f1
            hClose f2

        Arguments 2 input1 _ -> do
            --putStrLn "Task 2"
            f1 <- openFile input1 ReadMode
            c1 <- hGetContents f1
            let dataset = map createDato (map words (map stripInput (lines c1)))
            print (map fst dataset)
            let vectors = map sort (transpose (map fst dataset))
            print (vectors)
            let midPoints = map calculateMidPoint vectors
            print (midPoints)
            
            
            hClose f1
        _ -> putStrLn "Invalid input"