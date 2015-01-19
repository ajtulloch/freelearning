module Layers where

import           Text.Printf

data Layer = FC [[Float]]
           | Input
           | Convolution
           | Join
           | DropOut
           | LogSoftMax
           | Fork
           | MaxPool
           | ReLU
           deriving (Show)

fprop :: Layer -> [Float] -> [Float]
fprop (FC weights) input = map (dot input) weights
    where
      dot a b = sum (zipWith (*) a b)

fprop ReLU input = map (max 0) input
fprop Input input = input
fprop Convolution input = input
fprop DropOut input = input
fprop LogSoftMax input = input
fprop Fork input = input
fprop MaxPool input = input
fprop Join input = input

outputSize :: Layer -> Int -> [Int] -> [Int]
outputSize (FC weights) n [_] = [n * length weights]
outputSize (FC _)  _ _ = error "Must be one or two dimensional input"

outputSize ReLU n inputSize = map (* n) inputSize
outputSize Convolution n inputSize = map (* n) inputSize
outputSize Join n inputSize = map (* n) inputSize
outputSize LogSoftMax n inputSize = map (* n) inputSize
outputSize Fork n inputSize = map (* n) inputSize
outputSize MaxPool n inputSize = map (* n) inputSize
outputSize Input n inputSize = map (* n) inputSize
outputSize DropOut n inputSize = map (* n) inputSize

data PrintLayer = P Layer

instance Show PrintLayer where
    show (P (FC weights)) = printf "FC(%d)" $ length weights
    show (P l) = show l
