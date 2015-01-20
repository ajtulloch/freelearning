{-# LANGUAGE RecordWildCards #-}
module Layers where

import           Data.List
import           Text.Printf

data Pointwise = ReLU | DropOut deriving (Show)
data Criterion = LogSoftMax deriving (Show)
data LS = LSInput | LSOutput | LSOutputGate | LSForgetGate | LSInputGate | LSCell deriving (Show)

data CD = CD { width :: Int, stride :: Int} deriving (Show)
data CP = CP { nOutput :: Int, kernel :: [CD]} deriving (Show)
data MP = MP { steps :: [Int] } deriving (Show)
data FC = FC { nHidden :: Int, weights :: [[Float]] } deriving (Show)

type Tensor = [Float]

data Layer = FullyConnected FC
           | Pointwise Pointwise
           | Criterion Criterion
           | Convolution CP
           | ModelParallelJoin Int | ModelParallelFork Int
           | Input | Output
           | Reshape
           | MaxPool MP
           | LSTM LS
           deriving (Show)


fprop :: Layer -> Tensor -> Tensor
fprop (FullyConnected FC{..}) input = map (dot input) weights
    where
      dot a b = sum (zipWith (*) a b)
fprop (Pointwise DropOut) input = input
fprop (Pointwise ReLU) input = map (max 0) input

fprop (Convolution CP{..}) input = input
fprop (ModelParallelFork _) input = input
fprop Input input = input
fprop (ModelParallelJoin _) input = input
fprop (Criterion LogSoftMax) input = input
fprop (MaxPool _) input = input
fprop (Reshape) input = input
fprop Output input = input
divUp :: (Integral a, Integral b, Integral c) => a -> b -> c
divUp num denom = ceiling ((fromIntegral num :: Double) / fromIntegral denom)

-- Given *one* of the inputs on an edge, for *one* batch size, what
-- output is created on *one* of the edges?


outputSize :: Layer -> [Int] -> [Int]
outputSize (FullyConnected FC{..}) [_] = [nHidden]
outputSize (Pointwise _) inputSize = inputSize
outputSize layer@(Convolution CP{..}) inputSize = setOutputs (reverse (go (reverse inputSize)))
    where
      go revInputSize
          | length kernel > (length revInputSize - 1) = outputError layer inputSize
          | otherwise = zipWith outputWidth revInputSize kernel ++ drop (length kernel) revInputSize
      setOutputs (_:xs) = nOutput:xs
      setOutputs [] = []
      outputWidth input CD{..} = (input - 2 * width) `divUp` stride
outputSize (ModelParallelJoin n) (x:xs) = (n * x):xs
outputSize (Criterion LogSoftMax) [_] = [1]
outputSize Reshape inputSize = [product inputSize]
outputSize (ModelParallelFork n) (x:xs) = x `divUp` n:xs
outputSize layer@(MaxPool MP{..}) inputSize = reverse (go (reverse inputSize))
    where
      go revInputSize
          | length steps > length revInputSize = outputError layer inputSize
          | otherwise = zipWith divUp revInputSize steps ++ drop (length steps) revInputSize
outputSize Output inputSize = inputSize
outputSize Input inputSize = inputSize

-- Catch all
outputSize layer inputSize = outputError layer inputSize

outputError :: Layer -> [Int] -> a
outputError layer inputSize =
    error $ printf "Layer: %s, Unhandled size: %s" (show layer) (show inputSize)

newtype PrettyLayer = P Layer

instance Show PrettyLayer where
    show (P (FullyConnected FC{..})) = printf "FullyConnected %d" nHidden
    show (P (MaxPool MP{..})) = printf "MaxPool %s" $ intercalate "x" (map show steps)
    show (P (Convolution CP{..})) = printf "Convolution %sx%s" (show nOutput) kernels
        where
          kernels :: String
          kernels = intercalate "x" (map (show . width) kernel)
    show (P l) = show l
