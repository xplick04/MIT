module FileParser where

import DataTypes
import Text.Parsec (ParseError)
import Text.Parsec.String (Parser, parseFromFile)
import Text.Parsec.Char (string, oneOf, char, digit, satisfy, letter)
import Text.Parsec.Combinator (choice, many1, option)
import Control.Applicative (many)
import Control.Monad (void)



skipDelimiters :: Parser ()
skipDelimiters = void $ many $ oneOf ",: "

readIndex :: Parser Integer
readIndex = do
    skipDelimiters
    int <- many1 $ digit
    return $ read int

readTresh :: Parser Float
readTresh = do
    skipDelimiters
    integerPart <- many1 digit
    option "" (string ".")
    fractionalPart <- many digit
    if fractionalPart == "" then
        return $ read (integerPart ++ ".0")
    else
        return $ read (integerPart ++ "." ++ fractionalPart)

data MyData = MyData { field1 :: Integer, field2 :: Float, field3 :: String } deriving Show

fileParser :: Parser MyData
fileParser = do
    skipDelimiters
    nodeType <- choice [string "Node", string "Leaf"]
    i <- readIndex
    t <- readTresh
    
    return (MyData i t nodeType)

parseInputFile :: FilePath -> IO (Either ParseError MyData)
parseInputFile filename = parseFromFile fileParser filename

