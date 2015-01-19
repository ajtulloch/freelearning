{-# LANGUAGE RecordWildCards #-}
module Layers where

import           Text.Printf

data Pointwise = ReLU | DropOut deriving (Show)
data Criterion = LogSoftMax deriving (Show)

data CP = CP { nOutput :: Int, kW :: Int, kH :: Int} deriving (Show)
data MP = MP { steps :: [Int] } deriving (Show)

data Layer = FC [[Float]]
           | Pointwise Pointwise
           | Criterion Criterion
           | Convolution
           | ModelParallelJoin Int
           | Input
           | ModelParallelFork Int
           | MaxPool MP
           deriving (Show)

fprop :: Layer -> [Float] -> [Float]
fprop (FC weights) input = map (dot input) weights
    where
      dot a b = sum (zipWith (*) a b)
fprop (Pointwise DropOut) input = input
fprop (Pointwise ReLU) input = map (max 0) input

fprop (Convolution) input = input
fprop (ModelParallelFork _) input = input
fprop Input input = input
fprop (ModelParallelJoin _) input = input
fprop (Criterion LogSoftMax) input = input
fprop (MaxPool _) input = input


divUp :: (Integral a, Integral b, Integral c) => a -> b -> c
divUp num denom = ceiling ((fromIntegral num :: Double) / fromIntegral denom)

-- Given *one* of the inputs, what output is created on *one* of the
-- edges?
outputSize :: Layer -> [Int] -> [Int]
outputSize (FC weights) [b, _] = [b, length weights]
outputSize (FC weights) [_] = [length weights]
outputSize (Pointwise _) inputSize = inputSize
outputSize (Convolution) inputSize = inputSize
outputSize (ModelParallelJoin n) (x:xs) = (n * x):xs
outputSize (Criterion LogSoftMax) [b, _] = [b, 1]
outputSize (Criterion LogSoftMax) [_] = [1]
outputSize (ModelParallelFork n) (x:xs) = outer:xs
    where
      outer :: Int
      outer = x `divUp` n
outputSize layer@(MaxPool MP{..}) inputSize = reverse (go (reverse inputSize))
    where
      go revInputSize
          | length steps > length revInputSize
              = error $ printf "Layer: %s, Unhandled size: %s" (show layer) (show inputSize)
          | otherwise = zipWith divUp revInputSize steps ++ drop (length steps) revInputSize

outputSize Input inputSize = inputSize
outputSize layer inputSize =
    error $ printf "Layer: %s, Unhandled size: %s" (show layer) (show inputSize)

data PrettyLayer = P Layer

instance Show PrettyLayer where
    show (P (FC weights)) = printf "FC %d" $ length weights
    show (P l) = show l
