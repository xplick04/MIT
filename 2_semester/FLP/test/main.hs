import DataTypes
import Preprocessing
import TreeOperations

import System.IO
import System.Environment


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
            fileHandle <- openFile input1 ReadMode
            contents <- hGetContents fileHandle
            let inputM = modifyInput contents
            let nodeLevels = countSpaceSequences inputM 0
            let strippedInput = stripInput inputM
                list = words strippedInput

            case makeTuples list nodeLevels of
                Left err -> putStrLn err
                Right tuples -> do
                    let b = helper tuples 1

                    print (tuples)
                    --print (a)
                    --print (getElem b 1)
                    --print (getElem b 2)
                    let tree = makeTree tuples
                    printTree tree
            hClose fileHandle

        Arguments 2 input1 input2 -> putStrLn "Task 2"

        _ -> putStrLn "Invalid input"
    