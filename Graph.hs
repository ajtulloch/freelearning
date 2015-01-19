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

import           Debug.Trace
import           Layers
import           Text.Printf

type G = Gr Layer
type GS = State (Int, G ())

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

toGraph :: GS a -> G ()
toGraph f = snd (execState f (0, empty))

type EdgeActivations a = HM.HashMap (Int, Int) a

updateActivation :: G b -> (Layer -> [a] -> a) -> Node -> State (EdgeActivations a) ()
updateActivation graph f gid = do
  let inputIds = pre graph gid
  let l = fromJust (lab graph gid)
  acts <- get
  let inputActs = map (\k -> fromJust (HM.lookup (k, gid) acts)) inputIds
  let outputAct = f l inputActs
  put (setOutgoingActivations graph gid outputAct acts)

setOutgoingActivations :: G b -> Node -> a -> EdgeActivations a -> EdgeActivations a
setOutgoingActivations graph gid outputAct acts = foldl step acts outputIds where
    outputIds = suc graph gid
    step m outputId = HM.insert (gid, outputId) outputAct m

updateEdges :: G b -> EdgeActivations a -> G a
updateEdges graph acts = gmap update graph
    where
      update (pre, gid, l, suc) = (newpre, gid, l, newsucc)
          where
            newpre = map (\(a, preId) -> (actAt acts preId gid, preId)) pre
            newsucc = map (\(a, succId) -> (actAt acts gid succId, succId)) suc

actAt :: EdgeActivations a -> Node -> Node -> a
actAt acts from to = fromJust (HM.lookup (from, to) acts)

activations :: G b -> (Layer -> [a] -> a) -> a -> EdgeActivations a
activations graph f input = execState (mapM step layers) initialState
    where
      (inputLayer:layers) = topsort graph
      step = updateActivation graph f
      initialState = setOutgoingActivations graph inputLayer input HM.empty

fpropAll :: Layer -> [[Float]] -> [Float]
fpropAll l inputs = fprop l (concat inputs)

sizeAll :: Layer -> [[Int]] -> [Int]
sizeAll l inputs = trace (printf "L: %s, I: %s" (show (P l)) (show inputs)) $ outputSize l (length inputs) (head inputs)

simpleNet :: G ()
simpleNet = toGraph $
            column [Input,
                    FC [[1.0, 1.0], [-1.0, -1.0]], ReLU,
                    FC [[1.0, 1.0], [-1.0, -1.0]], ReLU,
                    LogSoftMax]

-- graphToDot :: G a -> (a -> String) -> String

-- main :: IO ()
main = do
  -- mapM_ (print . first P) $ activations alexNet [1.0, 2.0]
  let edgeActivatedGraph = updateEdges alexNet (activations alexNet fpropAll [1.0, 2.0])
  let dot = (showDot . fglToDotString . emap (\e -> printf "%d" $ length e) . nmap (show . P)) edgeActivatedGraph
  writeFile "file.dot" dot
  _ <- system "dot -Tpng -ofile.png file.dot"
  _ <- system "open file.png &"
  return ()

main2 = do
  -- mapM_ (print . first P) $ activations alexNet [1.0, 2.0]
  let edgeActivatedGraph = updateEdges alexNet (activations alexNet sizeAll [2])
  let dot = (showDot . fglToDotString . emap (\e -> printf "%s" $ (show . product) e) . nmap (show . P)) edgeActivatedGraph
  writeFile "file.dot" dot
  _ <- system "dot -Tpng -ofile.png file.dot"
  _ <- system "open file.png &"
  return ()


alexNet :: G ()
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
           -- Convolution, ReLU, MaxPool,
           -- Convolution, ReLU,
           -- Convolution, ReLU,
           Convolution, ReLU, MaxPool]

      classifierColumn = [
           -- DropOut, FC [[1]], ReLU,
           -- DropOut, FC [[1]], ReLU,
           FC [[1]], LogSoftMax
          ]
