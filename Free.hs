{-# LANGUAGE DeriveFoldable      #-}
{-# LANGUAGE DeriveFunctor       #-}
{-# LANGUAGE DeriveTraversable   #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Main where

import           Control.Monad
import           Control.Monad.Free
import           Control.Monad.IO.Class
import           Control.Monad.Random

import qualified Data.Foldable          as F
import           Data.Graph.Free
import           Data.GraphViz
import qualified Data.Traversable       as T
import           System.Process

import           Layers

data NNOperator next = Node Layer next
                     | Split [next]
                     | End
                       deriving (Functor, F.Foldable, T.Traversable)

type NN = Free NNOperator

layer :: Layer -> NN ()
layer l = liftF $ Node l ()

end :: NN a
end = liftF End

alexNet :: NN ()
-- alexNetColumns = Free (Split $ replicate 2 $ Free (Node ReLU (Free End)))
alexNet = Free (Split $ replicate 2 (mapM_ layer features >> mapM_ layer classifier))


features :: [Layer]
features = [
           Convolution CP{nOutput=10, kernel=[CD{width=3, stride=2}]},
           Pointwise ReLU, MaxPool MP{steps=[2]},
           -- Convolution, Pointwise ReLU, MaxPool,
           -- Convolution, Pointwise ReLU,
           -- Convolution, Pointwise ReLU,
           Convolution CP{nOutput=10, kernel=[CD{width=3, stride=2}]},
           Pointwise ReLU, MaxPool MP{steps=[2]}]

classifier :: [Layer]
classifier = [
           Reshape,
           -- DropOut, FullyConnected [[1]], ReLU,
           -- DropOut, FullyConnected [[1]], ReLU,
           FullyConnected FC{nHidden=5, weights=[[1] | _ <- [1..5::Integer]]},
           Criterion LogSoftMax
          ]

slp :: [[Float]] -> NN ()
slp weights_ = do
    layer (FullyConnected FC{nHidden=length weights_, weights=weights_})
    layer (Pointwise ReLU)

mlp :: [[[Float]]] -> NN ()
mlp = mapM_ slp

activations :: MonadIO m => [Float] -> NN a -> m ()
activations input (Free (Node l k)) = do
  let output = fprop l input
  liftIO (print output)
  activations output k
activations _ (Free End) = return ()
activations input (Free (Split xs)) = mapM_ (activations input) xs
activations _ (Pure _) = return ()

randomWeights
  :: (MonadRandom m, Random a, Fractional a) => Int -> Int -> m [[a]]
randomWeights nInput nOutput_ = do
  rs <- replicateM nOutput_ (getRandomRs (-1.0, 1.0))
  return $ map (take nInput) rs

generateMlp :: NN ()
generateMlp = alexNet >> end

attributes :: NN a -> Attributes
attributes (Free (Node l _)) = [(toLabel . show . P) l, color Red]
attributes (Free End) = [toLabel "End", color Green]
attributes (Free (Split _)) = [toLabel "Split", color Blue]
attributes (Pure _) = []

main :: IO ()
main = do
  let nn = generateMlp
  activations [1, 1] nn
  generate (freeFoldableGraph nn) attributes "nn"
  _ <- system "open nn.png &"
  return ()

