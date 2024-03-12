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
            --print (dataset)

            let sortedVector = (map sort (transpose (map fst dataset))) !! 0
            --print (sortedVector)
            let midPoints = calculateMidPoint (nub sortedVector) --map nub removes duplicate values
            --print (midPoints)
            let uniqueLabels = nub (map snd dataset)
            --print (uniqueLabels)
            
            let lesser = map (\midPoint -> filterByMidPoint midPoint "under" dataset) midPoints
            let greater = map (\midPoint -> filterByMidPoint midPoint "over" dataset) midPoints

            
            let lesserCount = transpose (map (\label -> map (\lst -> countLabel lst label) lesser) uniqueLabels)
            let greaterCount = transpose (map (\label -> map (\grt -> countLabel grt label) greater) uniqueLabels)
            --print (lesserCount)

            let px = map (\sublist -> map (\x -> (x / fromIntegral (length sublist)) ** 2) sublist) lesserCount
            let py = map (\sublist -> map (\x -> (x / fromIntegral (length sublist)) ** 2) sublist) greaterCount

            --print (px)
            let giniX = map sum px
            let giniY = map sum py

            --print (giniX)
            let giniImpurityX = zipWith (\x g -> x - g) [1,1..] giniX
            let giniImpurityY = zipWith (\x g -> x - g) [1,1..] giniY
            print $ "X:" ++ (show giniImpurityX)
            print $ "Y:" ++ (show giniImpurityY)

            let countInLesserSubset = map sum lesserCount
            let countInGreaterSubset = map sum greaterCount
            print $ "G:" ++ (show countInLesserSubset)
            print $ "L:" ++ (show countInGreaterSubset)

            let total = zipWith (\x g -> x + g) countInGreaterSubset countInLesserSubset
            print (total)

            let weightedImpurity = zipWith5 (\l g t x y -> (l/t) * x + (g/t) * y) countInLesserSubset countInGreaterSubset total giniImpurityX giniImpurityY
            print (weightedImpurity)

            print (midPoints)

            let bestMidPoint = getFeatureBestMP weightedImpurity midPoints 0
        
            {-
            let datasetNext = map dropFirstNumber dataset
            --print ((map fst dataset))-}
            hClose f1
        _ -> putStrLn "Invalid input"