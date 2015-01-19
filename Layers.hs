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
fprop _ input = input

data PrintLayer = P Layer

instance Show PrintLayer where
    show (P (FC weights)) = printf "FC(%d)" $ length weights
    show (P l) = show l
