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

outputSize :: Layer -> [Int] -> [Int]
outputSize (FC weights) [input] = [length weights]
outputSize (FC _) _ = error "Must be one or two dimensional input"

outputSize ReLU inputSize = inputSize
outputSize Convolution inputSize = inputSize
outputSize Join inputSize = inputSize
outputSize LogSoftMax inputSize = inputSize
outputSize Fork inputSize = inputSize
outputSize MaxPool inputSize = inputSize
outputSize Input inputSize = inputSize
outputSize DropOut inputSize = inputSize

data PrintLayer = P Layer

instance Show PrintLayer where
    show (P (FC weights)) = printf "FC(%d)" $ length weights
    show (P l) = show l
