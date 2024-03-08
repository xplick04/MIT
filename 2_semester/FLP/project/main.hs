import FileParser
import DataTypes

import System.IO
import System.Environment
import Text.Parsec

parseArgs :: [String] -> Arguments -> Integer -> Arguments -- user input list, parsed arguments, argument number -> parsed arguments
parseArgs [] args _ = args  -- end case
parseArgs ("-1":xs) (Arguments _ _ i2) 0 = parseArgs xs (Arguments 1 "" i2) 1   -- first argument
parseArgs ("-2":xs) (Arguments _ _ i2) 0 = parseArgs xs (Arguments 2 "" i2) 1   -- first argument
parseArgs (x:xs) (Arguments t _ i2) 1 = parseArgs xs (Arguments t x i2) 2  -- second argument
parseArgs (x:xs) (Arguments t i1 _) 2  -- third argument (only for task 1)
    | t == 1    = parseArgs xs (Arguments t i1 x) 3

parseArgs _ _ _ = Arguments 0 "" "" -- error


main :: IO ()
main = do
    args <- getArgs
    let parsedArgs = parseArgs args (Arguments 0 "" "") 0
    
    case parsedArgs of
        Arguments 1 _ _ -> do
            let filename = input1 parsedArgs
            result <- parseInputFile filename
            putStrLn $ show result
        Arguments 2 _ _ -> do
            putStrLn "Task 2"
        _ ->
            error "Invalid input"
    