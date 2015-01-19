{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Graph where

import           Control.Monad.State.Strict

import           Control.Arrow
import           Data.Graph.Inductive.Dot
import           Data.Graph.Inductive.Graph
import           Data.Graph.Inductive.PatriciaTree
import           Data.Graph.Inductive.Query
import qualified Data.HashMap.Strict               as HM
import           Data.Maybe
import           System.Process

import           Layers

type G = Gr Layer ()
type GS = State (Int, G)

layer :: Layer -> GS Int
layer l = do
  (gid, s) <- get
  put (gid + 1, insNode (gid, l) s)
  return gid

edge :: Int -> Int -> GS ()
edge from to = modify (second (insEdge (from, to, ())))

column :: [Layer] -> GS (Int, Int)
column layers = do
  ids <- mapM layer layers
  mapM_ (uncurry edge) (pairs ids)

  return (head ids, last ids)
    where
      pairs [] = []
      pairs xs = zip (init xs) (tail xs)

fork :: [[Layer]] -> GS (Int, Int)
fork columns = do
  splitId <- layer Fork

  ids <- mapM column columns
  mapM_ (edge splitId . fst) ids

  joinId <- layer Join
  mapM_ ((`edge` joinId) . snd) ids

  return (splitId, joinId)

toGraph :: GS a -> G
toGraph f = snd (execState f (0, empty))

activation :: G -> Node -> State (HM.HashMap Int [Float]) (Layer, [Float])
activation graph gid = do
  let inputIds = pre graph gid
  let l = fromJust (lab graph gid)
  acts <- get
  let inputActs = map (\k -> fromJust (HM.lookup k acts)) inputIds
  let outputActivation = fprop l (concat inputActs)
  put (HM.insert gid outputActivation acts)
  return (l, outputActivation)

activations :: G -> [Float]  -> [(Layer, [Float])]
activations graph input = evalState (mapM step layers) (HM.singleton inputLayer input)
    where
      (inputLayer:layers) = topsort graph
      step = activation graph

simpleNet :: G
simpleNet = toGraph $
            column [Input,
                    FC [[1.0, 1.0], [-1.0, -1.0]], ReLU,
                    FC [[1.0, 1.0], [-1.0, -1.0]], ReLU,
                    LogSoftMax]

main :: IO ()
main = do
  mapM_ (print . first P) $ activations alexNet [1.0, 2.0]
  let dot = (showDot . fglToDotString . emap (const "") . nmap (show . P)) alexNet
  writeFile "file.dot" dot
  _ <- system "dot -Tpng -ofile.png file.dot"
  _ <- system "open file.png &"
  return ()

alexNet :: G
alexNet = toGraph $ do
  inputId <- layer Input
  (splitId, joinId) <- fork (replicate 2 alexNetColumn)
  edge inputId splitId

  (classifierId, _) <- column classifierColumn
  edge joinId classifierId
  return ()
    where
      alexNetColumn = [
           Convolution, ReLU, MaxPool,
           Convolution, ReLU, MaxPool,
           Convolution, ReLU,
           Convolution, ReLU,
           Convolution, ReLU, MaxPool]

      classifierColumn = [
           DropOut, FC [[1]], ReLU,
           DropOut, FC [[1]], ReLU,
           FC [[1]], LogSoftMax
          ]
