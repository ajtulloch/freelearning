module Layers where

import           Text.Printf

data Pointwise = ReLU | DropOut deriving (Show)
data Criterion = LogSoftMax deriving (Show)

data Layer = FC [[Float]]
           | Pointwise Pointwise
           | Criterion Criterion
           | Convolution
           | DPJoin Int
           | Input
           | DPFork Int
           | MaxPool
           deriving (Show)

fprop :: Layer -> [Float] -> [Float]
fprop (FC weights) input = map (dot input) weights
    where
      dot a b = sum (zipWith (*) a b)
fprop (Pointwise DropOut) input = input
fprop (Pointwise ReLU) input = map (max 0) input

fprop Convolution input = input
fprop (DPFork _) input = input
fprop Input input = input
fprop (DPJoin _) input = input
fprop (Criterion LogSoftMax) input = input
fprop MaxPool input = input

outputSize :: Layer -> [Int] -> [Int]
outputSize (FC weights) [b, _] = [b, length weights]
outputSize (FC weights) [_] = [length weights]
outputSize (Pointwise _) inputSize = inputSize
outputSize Convolution inputSize = inputSize
outputSize (DPJoin n) (x:xs) = (n * x):xs
outputSize (Criterion LogSoftMax) [b, _] = [b, 1]
outputSize (Criterion LogSoftMax) [_] = [1]
outputSize (DPFork n) (x:xs) = outer:xs
    where
      outer :: Int
      outer = ceiling ((fromIntegral x :: Double) / fromIntegral n)
outputSize MaxPool inputSize = inputSize
outputSize Input inputSize = inputSize
outputSize layer inputSize =
    error $ printf "Layer: %s, Unhandled size: %s" (show layer) (show inputSize)


data PrettyLayer = P Layer

instance Show PrettyLayer where
    show (P (FC weights)) = printf "FC(%d)" $ length weights
    show (P l) = show l
